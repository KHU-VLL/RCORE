from itertools import product
from os.path import join as ospj
import os
import json
import cv2

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

import dataset.gtransform as gtransform
from numpy.random import randint
from random import choice
import random, pickle
from collections import Counter


import decord
from decord import VideoReader
decord.bridge.set_bridge("torch")

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def dataset_transform(phase):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use

        Returns
            transform: A list of pytorch transforms
    '''
    # mean, std = get_norm_values(norm_family=norm_family)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    if phase == 'train':
        transform = transforms.Compose([
            gtransform.GroupResize(256),
            gtransform.GroupMultiScaleCrop(224),
            # transforms.RandomHorizontalFlip(),
            gtransform.ToTensor(),
            gtransform.GroupNormalize(img_mean, img_std)
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            gtransform.GroupResize(256),
            gtransform.GroupCenterCrop(224),
            gtransform.ToTensor(),
            gtransform.GroupNormalize(img_mean, img_std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            gtransform.GroupResize(256),
            gtransform.GroupCenterCrop(224),
            gtransform.ToTensor(),
            gtransform.GroupNormalize(img_mean, img_std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img
    

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 


class CompositionVideoDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            num_negs=-1,
            frames_duration=8,
            tdn_input=False,
            aux_input=False,
            use_composed_pair_loss=False,
            ade_input=False,
            return_n_matrix=True,
            test_json='test_pairs.json',
            ex_test_json='test_pairs.json',
            given_known_pairs=False,
            split_root=None,
            config=None,
            return_ids=False,
            processor=None
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        print("Dataset Open world: ", self.open_world)
        split_root = './data_split/sth_com' if split_root is None else split_root
        print("Split root : ", split_root)
        self.splitroot = split_root
        self.test_json = test_json
        self.val_json = 'val_pairs.json'
        self.ex_test_json = ex_test_json

        self.return_idx = True if getattr(config, "use_lff", False) else False
        print("Return idx : ", self.return_idx)

        self.train_json = "train_pairs.json"
        print("***"*10)
        print("train file : ", self.train_json)
        print("***"*10)
        self.return_ids = return_ids

        self.tdn_input = tdn_input
        self.in_duration = frames_duration
        self.seg_length = 1 if not tdn_input else 5
        self.index_bias = 1
        self.total_length = self.in_duration * self.seg_length

        self.num_negs = num_negs

        self.feat_dim = None
        self.transform = dataset_transform(self.phase)

   
        self.attrs, self.objs, self.pairs, \
        self.train_pairs, self.val_pairs, \
        self.test_pairs,self.ex_test_pairs = self.parse_split()

        #!#### to align with test classifier
        self.attrs = sorted(self.attrs)
        self.objs = sorted(self.objs)
        self.pairs = sorted(self.pairs)

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        print('Number of pairs: %d' % len(self.pairs))
        print('Number of attrs: %d' % len(self.attrs))
        print('Number of objs: %d' % len(self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = self.get_sorted_pair2idx(self.pairs)

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))


        self.train_pair_to_idx = self.get_sorted_pair2idx(self.train_pairs)

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)


    def get_sorted_pair2idx(self, contents) :
        new_idx = 0
        result_dict = {}
        contents_set = set(contents)
        for pair in list(product(self.attrs, self.objs)) :
            if pair in contents_set :
                result_dict[pair] = new_idx
                new_idx += 1
        return result_dict

    def prepare_data(self):
        frame_cnts = {}
        for item in self.data:
            item_id = item[0]
            try:
                frames_path = ospj(self.root, item_id)
                frames = os.listdir(frames_path)
                n_frame = int(len(frames))
            except Exception as e:
                print(str(e))
            frame_cnts[item_id] = n_frame

        self.frame_cnts = frame_cnts

    def get_split_info(self):
        with open(ospj(self.splitroot, self.train_json), 'r') as f:
            items = json.load(f)
            id_key = 'vid' if 'vid' in items[0].keys() else 'id'
            if id_key == "vid" :
                train_data = [[item[id_key], item['verb'], item['object'], item['id']] for item in items]
            else :
                train_data = [[item['id'], item['verb'], item['object']] for item in items]

        with open(ospj(self.splitroot, self.val_json), 'r') as f:
            items = json.load(f)
            id_key = 'vid' if 'vid' in items[0].keys() else 'id'
            if id_key == "vid" :
                val_data = [[item[id_key], item['verb'], item['object'], item['id']] for item in items]
            else :
                val_data = [[item['id'], item['verb'], item['object']] for item in items]

        with open(ospj(self.splitroot, self.test_json), 'r') as f:
            items = json.load(f)
            id_key = 'vid' if 'vid' in items[0].keys() else 'id'
            if id_key == "vid" :
                test_data = [[item[id_key], item['verb'], item['object'], item['id']] for item in items]
            else :
                test_data = [[item[id_key], item['verb'], item['object']] for item in items]

        return train_data, val_data, test_data

    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                items = json.load(f)
            pairs = [[item['verb'], item['object']] for item in items]
            pairs = list(map(tuple, pairs))

            attrs, objs = zip(*pairs)
            return list(set(attrs)), list(set(objs)), list(set(pairs))

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.splitroot, self.train_json)
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.splitroot, self.val_json)
        )

        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.splitroot, self.test_json)
        )

        ex_ts_attrs, ex_ts_objs, ex_ts_pairs = parse_pairs(
            ospj(self.splitroot, self.ex_test_json)
        )

        # now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs+ex_ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs+ex_ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs + ex_ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs, ex_ts_pairs

    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        return Image.open(ospj(self.root, vid_name, '%06d.jpg' % (frame_idx))).convert('RGB')

    def _sample_indices(self, id):
        if not self.tdn_input:
            if self.frame_cnts[id] <= self.total_length:
                offsets = np.concatenate((
                    np.arange(self.frame_cnts[id]),
                    randint(self.frame_cnts[id],
                            size=self.total_length - self.frame_cnts[id])))
                offsets.sort()
                return offsets
            offsets = list()
            ticks = [i * self.frame_cnts[id] // self.in_duration
                     for i in range(self.in_duration + 1)]

            for i in range(self.in_duration):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    tick += randint(tick_len - self.seg_length + 1)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return offsets
        else:
            if ((self.frame_cnts[id] - self.seg_length + 1) < self.in_duration):
                average_duration = (self.frame_cnts[id] - 5 + 1) // (self.in_duration)
            else:
                average_duration = (self.frame_cnts[id] - self.seg_length + 1) // (self.in_duration)
            offsets = []
            if average_duration > 0:
                offsets += list(
                    np.multiply(list(range(self.in_duration)), average_duration) + randint(average_duration,
                                                                                           size=self.in_duration))
            elif self.frame_cnts[id] > self.in_duration:
                if ((self.frame_cnts[id] - self.seg_length + 1) >= self.in_duration):
                    offsets += list(np.sort(randint(self.frame_cnts[id] - self.seg_length + 1, size=self.in_duration)))
                else:
                    offsets += list(np.sort(randint(self.frame_cnts[id] - 5 + 1, size=self.in_duration)))
            else:
                offsets += list(np.zeros((self.in_duration,)))
            final_offset = []
            for i in offsets:
                for bias in range(5):
                    final_offset.append(i + bias)
            return final_offset

    def _get_val_indices(self, id):
        if not self.tdn_input:
            if self.in_duration == 1:
                return np.array([self.frame_cnts[id] // 2], dtype=np.int) + self.index_bias

            if self.frame_cnts[id] <= self.in_duration:
                return np.array([i * self.frame_cnts[id] // self.in_duration
                                 for i in range(self.in_duration)], dtype=np.int) + self.index_bias
            offset = (self.frame_cnts[id] / self.in_duration - self.seg_length) / 2.0
            return [i * self.frame_cnts[id] / self.in_duration + offset + j
                    for i in range(self.in_duration)
                    for j in range(self.seg_length)]
        else:
            if self.frame_cnts[id] > self.in_duration + self.seg_length - 1:
                tick = (self.frame_cnts[id] - self.seg_length + 1) / float(self.in_duration)
                offsets = [int(tick / 2.0 + tick * x) for x in range(self.in_duration)]
            else:
                offsets = [0 for i in range(self.in_duration)]

            final_offset = []
            for i in offsets:
                for bias in range(5):
                    final_offset.append(i + bias)
            return final_offset

    
    def _load_video(
        self,
        vid_name,
        sampling="uniform",
        clip_proposal=None,
    ):
        video_path = ospj(self.root, vid_name+".mp4")
        vr = VideoReader(uri=video_path)
        vlen = len(vr)
        fps = vr.get_avg_fps()
        if clip_proposal is None:
            start, end = 0, vlen
        else:
            start, end = int(clip_proposal[0] * fps), int(clip_proposal[1] * fps)
            if start < 0:
                start = 0
            if end > vlen:
                end = vlen

        intervals = np.linspace(start=start, stop=end, num=self.in_duration + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1]))

        if sampling == "random":
            indices = []
            for x in ranges:
                if x[0] == x[1]:
                    indices.append(x[0])
                else:
                    indices.append(random.choice(range(x[0], x[1])))
        elif sampling == "uniform":
            indices = []
            for x in ranges:
                index = (x[0] + x[1]) // 2
                if index < vlen:
                    indices.append(index)
                else:
                    indices.append(vlen - 1)

        else:
            raise NotImplementedError

        if len(indices) < self.in_duration:
            rest = [indices[-1] for i in range(self.in_duration - len(indices))]
            indices = indices + rest
        # get_batch -> T, H, W, C
        frms = vr.get_batch(indices).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        
        pil_list = []
        for frm in frms:
            if hasattr(frm, "cpu"):
                frame_np = frm.cpu().numpy()
            else:
                frame_np = np.array(frm)
            frame_np = frame_np.transpose(1, 2, 0)
            pil_img = Image.fromarray(frame_np.astype('uint8'))
            pil_list.append(pil_img)

        return pil_list


    def __len__(self):
        return len(self.data)

    def get_com_label_list(self) :
        return [self.train_pair_to_idx[(elem[1], elem[2])] for elem in self.data]


    def __getitem__(self, index):
        elem = self.data[index]
        if len(elem) == 4 :
            id, attr, obj, rid = elem
        else :
            id, attr, obj = elem

        vid = self._load_video(id)
        vid = self.transform(vid)

        if self.phase == 'train':
            if self.open_world :
                data = [
                    vid, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
                    ]
            else :
                data = [
                        vid, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
                        ]

            if self.return_idx :
                data.append(index)

        else:
            if self.return_ids:
                if "_" in id :
                    id = 0
                    # ek100, dummy
                else :
                    id = int(id)  # sth-com
                data = [
                vid, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)], id
                ]
            else :
                data = [
                    vid, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
                ]

        return data


