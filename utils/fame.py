import torch
import torch.nn as nn
import numpy as np
import kornia


class FAME(nn.Module):
    def __init__(self, crop_size=112, beta=0.5, device="cpu", eps=1e-8, sampling_fg_ratio=0.5):
        super().__init__()
        self.frame_mean = [0.485, 0.456, 0.406]
        self.frame_std = [0.229, 0.224, 0.225]
        self.crop_size = crop_size
        gauss_size = int(0.1 * crop_size) // 2 * 2 + 1
        self.gauss = kornia.filters.GaussianBlur2d(
            (gauss_size, gauss_size),
            (gauss_size / 3, gauss_size / 3),
        )
        self.device = device
        self.eps = eps
        self.beta = beta  # portion of foreground
        self.sampling_fg_ratio = sampling_fg_ratio
        print("sampling_fg_ratio: ", self.sampling_fg_ratio)

    def norm_batch(self, matrix):
        # matrix: B * H * W
        B, H, W = matrix.shape
        matrix = matrix.flatten(start_dim=1)
        matrix -= matrix.min(dim=-1, keepdim=True)[0]
        matrix /= (matrix.max(dim=-1, keepdim=True)[0] + self.eps)
        return matrix.reshape(B, H, W)

    def batched_bincount(self, x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        values = torch.ones_like(x)
        target.scatter_add_(dim, x, values)
        return target

    def getSeg(self, mask, video_clips):
        # input mask: B, H, W; video_clips: B, C, T, H, W
        # return soft seg mask: B, H, W
        B, C, T, H, W = video_clips.shape
        video_clips_ = video_clips.mean(dim=2)  # B, C, H, W
        img_hsv = kornia.color.rgb_to_hsv(video_clips_.reshape(-1, C, H, W))
        sampled_fg_index = torch.topk(mask.reshape(B, -1), k=int(self.sampling_fg_ratio * H * W), dim=-1)[1]
        sampled_bg_index = torch.topk(mask.reshape(B, -1), k=int(0.1 * H * W), dim=-1, largest=False)[1]

        dimH, dimS, dimV = 10, 10, 10
        img_hsv = img_hsv.reshape(B, -1, H, W)
        img_h = img_hsv[:, 0]
        img_s = img_hsv[:, 1]
        img_v = img_hsv[:, 2]
        hx = (img_s * torch.cos(img_h * 2 * np.pi) + 1) / 2
        hy = (img_s * torch.sin(img_h * 2 * np.pi) + 1) / 2
        h = torch.round(hx * (dimH - 1) + 1)
        s = torch.round(hy * (dimS - 1) + 1)
        v = torch.round(img_v * (dimV - 1) + 1)
        color_map = h + (s - 1) * dimH + (v - 1) * dimH * dimS
        color_map = color_map.reshape(B, -1).long()
        col_fg = color_map.gather(index=sampled_fg_index, dim=-1)
        col_bg = color_map.gather(index=sampled_bg_index, dim=-1)
        dict_fg = self.batched_bincount(col_fg, dim=1, max_value=dimH * dimS * dimV)
        dict_bg = self.batched_bincount(col_bg, dim=1, max_value=dimH * dimS * dimV)
        dict_fg = dict_fg.float()
        dict_bg = dict_bg.float() + 1
        dict_fg /= (dict_fg.sum(dim=-1, keepdim=True) + self.eps)
        dict_bg /= (dict_bg.sum(dim=-1, keepdim=True) + self.eps)

        pr_fg = dict_fg.gather(dim=1, index=color_map)
        pr_bg = dict_bg.gather(dim=1, index=color_map)
        refine_mask = pr_fg / (pr_bg + pr_fg)

        mask = self.gauss(refine_mask.reshape(-1, 1, H, W))
        mask = self.norm_batch(mask.reshape(-1, H, W))

        num_fg = int(self.beta * H * W)
        sampled_index = torch.topk(mask.reshape(B, -1), k=num_fg, dim=-1)[1]
        mask = torch.zeros_like(mask).reshape(B, -1)
        b_index = torch.LongTensor([[i] * num_fg for i in range(B)])
        mask[b_index.view(-1), sampled_index.view(-1)] = 1
        return mask.reshape(B, H, W)

    def getmask_per_frame(self, video_clips):
        # input: B, C, T, H, W
        # return list[T] of soft seg masks (B, H, W)
        B, C, T, H, W = video_clips.shape
        masks = []
        for i in range(T):
            end = i + 1 if i + 1 < T else i
            im_diff = (video_clips[:, :, i] - video_clips[:, :, end]).abs().sum(dim=1)
            mask = self.gauss(im_diff.reshape(-1, 1, H, W))
            mask = self.norm_batch(mask.reshape(-1, H, W))
            mask = self.getSeg(mask, video_clips)
            masks.append(mask)
        return masks

    def get_bboxes_from_masks(self, mask: torch.Tensor) -> torch.Tensor:
        """Project mask along H/W axes to find bbox extents per (B, T)."""
        B, T, H, W = mask.shape

        mask_exists = torch.any(mask, dim=(2, 3))
        y_indices = torch.any(mask, dim=3)
        x_indices = torch.any(mask, dim=2)

        ymin = torch.argmax(y_indices.long(), dim=2)
        ymax = (H - 1) - torch.argmax(torch.flip(y_indices, dims=[2]).long(), dim=2)
        xmin = torch.argmax(x_indices.long(), dim=2)
        xmax = (W - 1) - torch.argmax(torch.flip(x_indices, dims=[2]).long(), dim=2)

        bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=2)
        return bboxes * mask_exists.unsqueeze(-1).long()  # (B, T, 4)

    def forward(self, videos, return_bbox=True):
        assert return_bbox, "FAME only supports return_bbox=True"
        tmp_video = videos.permute(0, 2, 1, 3, 4).contiguous()

        # denormalize
        std = torch.tensor(self.frame_std, device=tmp_video.device).reshape(1, 3, 1, 1, 1)
        mean = torch.tensor(self.frame_mean, device=tmp_video.device).reshape(1, 3, 1, 1, 1)
        tmp_video = tmp_video * std + mean

        masks_per_frame = self.getmask_per_frame(tmp_video)
        masks_per_frame = torch.stack(masks_per_frame).permute(1, 0, 2, 3).to(videos.dtype)  # B, T, H, W

        bbox_per_frame = self.get_bboxes_from_masks(masks_per_frame)
        return bbox_per_frame, masks_per_frame
