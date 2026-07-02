import torch
import torch.nn as nn
from clip import clip
import json
from einops import rearrange
from copy import deepcopy
from torch.nn import functional as F
import numpy as np
import os
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

# from flash_attn.ops.rms_norm import DropoutAddRMSNorm
from .internvideo2 import InternVideo2, LLaMA, Tokenizer, TextTransformer, ClipTokenizer, Block, RMSNorm
from .custom_clip_c2c_prompts import MLP, MLP_ST, PromptLearner


class InternVideo_C2C(nn.Module):
    """
    CLIP model with InternVideo2 backbone and C2C modules
    """
    def __init__(self, clip_model_name="dummy", config=None, device=None):
        super().__init__()

        self.config = config
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load original CLIP model
        
        # Build InternVideo2 Encoder
        self.build_vision_encoder()
        self.build_text_encoder_small()
        
        self.temp = nn.parameter.Parameter(torch.ones([]) * 1 / 100.0)
        self.temp_min = 1 / 100.0

        self.load_checkpoint(
            config.iv_vision_ckpt_path, config.iv_text_ckpt_path, 
            getattr(config, "iv_extra_ckpt_path", None)
        )

        model_name = "C2C" if 'c2c' in config.method else ("AIM" if 'aim' in config.method else "BASE")
        print(f"Created {model_name} model with InternVideo2 backbone")

        self.label_index_file = getattr(config, "label_index_file", None)
        self.obj_label_index_file = getattr(config, "obj_label_index_file", None)
        self.prompt_learn_method = getattr(config, 'prompt_learn_method', None)  # class-specific context
        
        if "coop" == self.prompt_learn_method :
            ##################
            # CoOp configuration
            self.n_ctx = getattr(config, 'n_ctx', 4)  # number of context tokens
            self.ctx_init = getattr(config, 'input_template', None)  # initialization words
            self.class_specific = getattr(config, 'class_specific', False)  # class-specific context

            print(f"{self.prompt_learn_method} enabled with {self.n_ctx} context tokens")

            self.label_index_file = getattr(config, "label_index_file", None)
            self.obj_label_index_file = getattr(config, "obj_label_index_file", None)
            
            # Get class names for CoOp
            self.classnames, self.obj_classnames = self._get_classnames()

            positional_embedding_data = self.text_encoder.positional_embedding.pos_embed.pos_embed.clone()

            # Initialize CoOp modules
            # Prompt learner for verb/action classes
            self.prompt_learner_verb = PromptLearner(
                n_ctx=self.n_ctx,
                n_cls_ctx=len(self.classnames),
                ctx_init=self.ctx_init,
                classnames=self.classnames,
                token_embedding=deepcopy(self.text_encoder.embedding_layer),  # original_model.token_embedding
                positional_embedding=positional_embedding_data,  # original_model.positional_embedding
                device=self.device,
                learn_method=self.prompt_learn_method
            )
            
            # Prompt learner for object classes
            self.prompt_learner_obj = PromptLearner(
                n_ctx=self.n_ctx,
                n_cls_ctx=len(self.obj_classnames),
                ctx_init=self.ctx_init,
                classnames=self.obj_classnames,
                token_embedding=deepcopy(self.text_encoder.embedding_layer),  # original_model.token_embedding
                positional_embedding=positional_embedding_data,  # original_model.positional_embedding
                device=self.device,
                learn_method=self.prompt_learn_method
            )
            
            # Generate tokenized prompts for classes
            self.class_text_embedding = self._tokenize_prompts(self.classnames)
            self.obj_class_text_embedding = self._tokenize_prompts(self.obj_classnames)
            ##################
        else :
            # Use original text encoder
            self.class_text_embedding = self.get_class_text_embedding(self.label_index_file)
            self.obj_class_text_embedding = self.get_class_text_embedding(self.obj_label_index_file, mode="obj")
        
        self.dtype = self.vision_encoder.dtype
        self.c2c_pred_mode = getattr(config, 'c2c_pred_mode', "both")
        self.save_features = getattr(config, 'save_features', False)
        self.shared_encoder = getattr(config, 'shared_encoder', False)
        self.temporal_modeling_shuffle = getattr(config, 'temporal_modeling_shuffle', False)

        if 'c2c' in config.method :
            print(f"Created C2C model with InternVideo2 backbone")
            # C2C Components
            try:
                fc_emb = config.fc_emb.split(',')
            except:
                fc_emb = [config.fc_emb]
            layers = [int(a) for a in fc_emb]

            self.c2c_OE1 = MLP(config.emb_dim, int(config.emb_dim), relu=config.relu, 
                                num_layers=config.nlayers, dropout=False, norm=True, layers=layers)
            if not self.shared_encoder :
                self.c2c_OE2 = MLP(config.emb_dim, int(config.emb_dim), relu=config.relu, 
                                num_layers=config.nlayers, dropout=False, norm=True, layers=layers)
            self.c2c_VE1 = MLP_ST(config.emb_dim, int(config.emb_dim), relu=config.relu, 
                                    num_layers=config.nlayers, dropout=False, norm=True, layers=layers)
            if not self.shared_encoder :
                self.c2c_VE2 = MLP_ST(config.emb_dim, int(config.emb_dim), relu=config.relu, 
                                    num_layers=config.nlayers, dropout=False, norm=True, layers=layers)

            # Composition modules
            self.c2c_f_v_e_o_com = nn.Linear(2 * config.emb_dim, config.emb_dim, bias=True)
            self.c2c_f_o_e_v_com = nn.Linear(2 * config.emb_dim, config.emb_dim, bias=True)
            self.c2c_text_v = nn.Linear(config.feat_dim, config.emb_dim, bias=True)
            self.c2c_text_o = nn.Linear(config.feat_dim, config.emb_dim, bias=True)
        else :
            print(f"Created Vanilla InternVideo2")
            self.c2c_VE1 = self.c2c_OE1 = None


    def build_vision_encoder(self):
        vision_cfg = dict(
            in_chans=3,
            patch_size=14,
            img_size=224,
            qkv_bias=False,
            drop_path_rate=0.0,
            head_drop_path_rate=0.0,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4,
            init_values=0.1,
            qk_normalization=True,
            depth=12,
            use_flash_attn=getattr(self.config, "use_flash_attn", False),
            use_fused_rmsnorm=getattr(self.config, "use_flash_attn", False),
            use_fused_mlp=getattr(self.config, "use_flash_attn", False),
            fused_mlp_heuristic=1,
            attn_pool_num_heads=16,
            clip_embed_dim=768,
            layerscale_no_force_fp32=getattr(self.config, "use_flash_attn", False),
            num_frames=getattr(self.config, "num_frames", 16),
            tubelet_size=1,
            sep_pos_embed=False,
            use_checkpoint=False,
            checkpoint_num=0,
        )
        align_dim=512

        # if 'aim' in self.config.method or 'c2c' in self.config.method:
        if 'aim' in self.config.method:
            self.vision_encoder = get_internvideo_aim(self.config, vision_cfg)
        else :
            self.vision_encoder = InternVideo2(**vision_cfg)
        
        self.vision_align = nn.Sequential(
            nn.LayerNorm(vision_cfg["clip_embed_dim"]),
            nn.Linear(
                vision_cfg["clip_embed_dim"], 
                align_dim,

            ),
        )

    def build_text_encoder(self):
        text_cfg = dict(
            use_flash_attn=True,
            transformer_width=4096,
            llama_path=self.config.text_encoder_llama_path,
            use_lora=True,
        )
        self.text_encoder = LLaMA(**text_cfg)
    
    def build_text_encoder_small(self):
        iv_text_encoder_name = getattr(self.config, "iv_text_encoder_name", "mobileclip_b")
        text_cfg = json.load(open(os.path.join(
                "./model/internvideo2/mobileclip/configs/" + \
                f"{iv_text_encoder_name}.json"))
        )
        self.tokenizer = ClipTokenizer(text_cfg)
        self.text_encoder = TextTransformer(text_cfg['text_cfg'], text_cfg["embed_dim"])

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        # no weight decay for LLM if training
        ret.update(
            {"text_encoder." + k for k, _ in self.text_encoder.named_parameters()}
        )

        return ret


    def load_checkpoint(self, vision_ckpt_path=None, text_ckpt_path=None, extra_ckpt_path=None):
        assert vision_ckpt_path is not None, "No vision_encoder checkpoint"
        assert text_ckpt_path is not None, "No text_encoder checkpoint"

        new_ckpt = {}

        # load vision_encoder
        print(f"Load vision_encoder checkpoint from {vision_ckpt_path}")
        vision_ckpt = torch.load(vision_ckpt_path, map_location='cpu')

        for k, v in vision_ckpt.items():
            if k.startswith('clip_decoder.') or k.startswith('mae_decoder.') or k.startswith('final_clip_decoder.'):
                continue
            elif 'clip_pos_embed' in k or 'mae_pos_embed' in k:
                continue
            else:
                # new_k = 'vision_encoder.' + k
                if k.startswith('vision_encoder.') or 'align' in k :
                    new_ckpt[k] = v
                else:
                    new_k = 'vision_encoder.' + k
                    new_ckpt[new_k] = v

        # load text_encoder
        print(f"Load text_encoder checkpoint from {text_ckpt_path}")
        test_ckpt = torch.load(text_ckpt_path, map_location='cpu')

        for k, v in test_ckpt.items():
            if k.startswith('text_encoder.'):
                new_ckpt[k] = v

        # load extra checkpoint
        # often when post-pretrain after previous pretraining, thus the keys are same
        if extra_ckpt_path is not None:
            print(f"Load extra checkpoint from {extra_ckpt_path}")
            extra_ckpt = torch.load(extra_ckpt_path, map_location='cpu')
            if 'module' in extra_ckpt.keys():
                extra_ckpt = extra_ckpt['module']
            for k, v in extra_ckpt.items():
                if 'decoder' in k :
                    continue
                new_k = 'vision_encoder.' + k
                new_ckpt[new_k] = v
        
        msg = self.load_state_dict(new_ckpt, strict=False)
        print(msg)


    def _get_classnames(self):
        """Extract class names from label files or config"""
        # Get verb/action class names

        if self.label_index_file is not None:
            with open(self.label_index_file, 'r') as f:
                classnames = list(json.load(f).values())
        else:
            classnames = self.config.attrs if hasattr(self.config, 'attrs') else []
        
        # Get object class names
        if self.obj_label_index_file is not None:
            with open(self.obj_label_index_file, 'r') as f:
                obj_classnames = list(json.load(f).values())
        else:
            obj_classnames = self.config.objs if hasattr(self.config, 'objs') else []
        
        classnames = [classname.replace("[", "").replace("]", "") for classname in classnames]

        print("Class names:", classnames)
        print("Object class names:", obj_classnames)

        return classnames, obj_classnames
    

    def _tokenize_prompts(self, classnames):
        """Tokenize prompts with placeholder for learnable context"""
        prompts = []
        for name in classnames:
            # Create template with placeholder for context tokens
            # e.g., "X X X X [CLASS]" where X will be replaced by learnable tokens
            prompt = " ".join(["X"] * self.n_ctx) + f" {name}."
            prompts.append(prompt)
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        return tokenized_prompts
    

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask


    def get_class_text_embedding(self, label_index_file=None, mode="attr") :
        if not hasattr(self.config, "input_template") :
            self.config.input_template = self.config.input_template_obj
        print("USED template : ", self.config.input_template)

        if label_index_file is not None :
            with open(label_index_file, 'r') as f :
                labels = json.load(f).values()
            print(f"SET labels (num {len(labels)}) from : ", label_index_file)
        else :
            print(f"LOAD labels from the training dataset")
            if mode == "attr" :
                labels = self.config.attrs
            elif mode == "obj" :
                labels = self.config.objs
            print(f"SET labels (num {len(labels)}) (mode={mode})")

        labels = [classname.replace("[", "").replace("]", "") for classname in labels]
        text_prompts = [self.config.input_template.replace("x", elem) for elem in labels]
        text_embs = clip.tokenize(text_prompts).to(self.device)
        return text_embs
    

    def encode_image(self, x, return_all=False):
        B, T, C, H, W = x.shape
        use_image = True if T == 1 else False
        image = x.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

        vision_embeds = self.vision_encoder(image, use_image=use_image, return_all_features=return_all)
        if return_all :
            return vision_embeds    # (B, T, C)
        vision_embeds = self.vision_align(vision_embeds)
        return vision_embeds    # (B, C)


    def encode_text(self, text, input_prompts=None):
        text_embeds = self.text_encoder(text, input_prompts=input_prompts)
        return text_embeds

    def set_save_features_true(self) :
        self.save_features = True

    def set_save_features_false(self) :
        self.save_features = False


    def forward(self, x=None, **kwargs) :
        # x: [B, T, C, H, W]
        if x is None :
            x = kwargs['input_ids']
        B, T, C, H, W = x.shape

        # image_features: [B, T, C]
        return_all = False if self.c2c_VE1 is None else True
        image_features = self.encode_image(x, return_all=return_all)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Permute for MLP_ST: [B, C, T]
        # image_features = image_features.permute(0, 2, 1)
        
        if "coop" == self.prompt_learn_method :
            verb_prompts = self.prompt_learner_verb()
            verb_text_features = self.encode_text(verb_prompts, input_prompts=self.class_text_embedding)
            obj_prompts = self.prompt_learner_obj()
            obj_text_features = self.encode_text(obj_prompts, input_prompts=self.obj_class_text_embedding)
        else :
            verb_text_features = self.encode_text(self.class_text_embedding)
            obj_text_features = self.encode_text(self.obj_class_text_embedding)
        
        if self.c2c_VE1 is None :
            ### BASE or AIM
            verb_text_features = verb_text_features / verb_text_features.norm(dim=-1, keepdim=True)
            obj_text_features = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)

            verb_logits = image_features @ verb_text_features.t() * 0.5 + 0.5
            obj_logits = image_features @ obj_text_features.t() * 0.5 + 0.5

            com_logits = torch.einsum('bn,bm->bnm', verb_logits, obj_logits)
            com_logits = com_logits.reshape(B, -1)

            if self.training :
                return verb_logits, obj_logits, com_logits
            
            if self.save_features :
                return verb_logits, obj_logits, com_logits

            return com_logits

        verb_text_features = self.c2c_text_v(verb_text_features)
        verb_text_features = verb_text_features / verb_text_features.norm(dim=-1, keepdim=True)
        
        obj_text_features = self.c2c_text_o(obj_text_features)
        obj_text_features = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)

        #! component
        image_features = image_features.permute(0, 2, 1)  # B, C, T
        # image_features.mean(dim=-1) -> [B, C] (Global avg over time)
        o_feat = self.c2c_OE1(image_features.mean(dim=-1))  
        o_feat_normed = F.normalize(o_feat, dim=1)
        
        # MLP_ST takes [B, C, T]
        v_feat_t = self.c2c_VE1(image_features)  
        v_feat = v_feat_t.mean(dim=-1)  
        v_feat_normed = F.normalize(v_feat, dim=1)

        if self.temporal_modeling_shuffle :
            # if self.temporal_modeling_shuffle_type in ['cosine_entropy'] :
            if True:
                random_shuffle_indices = torch.rand(B, T).argsort(dim=1)
                reverse_shuffle_indices = torch.arange(T).flip(dims=[0]).expand(B, -1)
            
                random_shuffle_indices = random_shuffle_indices.unsqueeze(1).expand(-1, image_features.shape[1], -1).to(image_features.device)
                random_shuffled_x = image_features.clone().gather(2, random_shuffle_indices)
                random_shuffled_v_feat = self.c2c_VE1(random_shuffled_x)

                reverse_shuffle_indices = reverse_shuffle_indices.unsqueeze(1).expand(-1, image_features.shape[1], -1).to(image_features.device)
                reverse_shuffled_x = image_features.clone().gather(2, reverse_shuffle_indices)
                reverse_shuffled_v_feat = self.c2c_VE1(reverse_shuffled_x)

                random_shuffled_v_feat = random_shuffled_v_feat.mean(dim=-1)
                reverse_shuffled_v_feat = reverse_shuffled_v_feat.mean(dim=-1)
                
                random_shuffled_v_feat_normed = F.normalize(random_shuffled_v_feat, dim=1)
                reverse_shuffled_v_feat_normed = F.normalize(reverse_shuffled_v_feat, dim=1)

                shuffled_verb_loss = F.cosine_similarity(v_feat_normed, reverse_shuffled_v_feat_normed, dim=-1)
                shuffled_verb_logits = random_shuffled_v_feat_normed @ verb_text_features.t() * 0.5 + 0.5
                shuffled_verb_outputs = [shuffled_verb_loss, shuffled_verb_logits, shuffled_verb_loss]


        verb_logits = v_feat_normed @ verb_text_features.t() * 0.5 + 0.5
        obj_logits = o_feat_normed @ obj_text_features.t() * 0.5 + 0.5

        #! composition
        if not self.shared_encoder :
            o_feat_c = self.c2c_OE2(image_features.mean(dim=-1))
            v_feat_c = self.c2c_VE2(image_features)
        else :
            o_feat_c = self.c2c_OE1(image_features.mean(dim=-1))
            v_feat_c = self.c2c_VE1(image_features)
        v_feat_c = v_feat_c.mean(dim=-1)

        b = B
        c_dim = verb_text_features.shape[-1]
        n_v = verb_logits.shape[-1]
        n_o = obj_logits.shape[-1]

        p_v_con_o, p_o_con_v = self.condition_module(v_feat_c, o_feat_c, verb_text_features, 
                                                    obj_text_features, n_o, b, c_dim, n_v)
        p_pair_o = p_v_con_o * obj_logits.unsqueeze(1)
        p_pair_v = p_o_con_v * verb_logits.unsqueeze(-1)

        com_logits = p_pair_o + p_pair_v
        com_logits = com_logits.reshape(b, -1)

        if self.training :
            if self.temporal_modeling_shuffle :
                return verb_logits, obj_logits, com_logits, shuffled_verb_outputs
            return verb_logits, obj_logits, com_logits
        
        if self.save_features :
            # return v_feat_normed, o_feat_normed, v_feat_c, o_feat_c
            return verb_logits, obj_logits, com_logits
            # return verb_logits, obj_logits, com_logits, p_v_con_o, p_o_con_v

        # if self.c2c_pred_mode == "both" :
        return com_logits
        # elif self.c2c_pred_mode == "verb" :
        #     return verb_logits
        # elif self.c2c_pred_mode == "obj" :
        #     return obj_logits
        # else :
        #     raise NotImplementedError()
    
    def condition_module(self, v_feat_c, o_feat_c, v_emb, o_emb, n_o, b, c, n_v):
        v_emb_normed = F.normalize(v_emb, dim=1)
        o_emb_normed = F.normalize(o_emb, dim=1)

        f_v_e_o = self.c2c_f_v_e_o_com(
            torch.cat([v_feat_c.unsqueeze(1).repeat(1, n_o, 1), 
                      o_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2))
        f_v_e_o_norm = F.normalize(f_v_e_o, dim=-1)
        f_v_e_o_norm = f_v_e_o_norm.view(b, n_o, c)

        f_o_e_v = self.c2c_f_o_e_v_com(
            torch.cat([o_feat_c.unsqueeze(1).repeat(1, n_v, 1), 
                      v_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2))
        f_o_e_v_norm = F.normalize(f_o_e_v, dim=-1)
        f_o_e_v_norm = f_o_e_v_norm.view(b, n_v, c)

        p_v_con_o = torch.einsum('bnc,mc->bnm', f_v_e_o_norm, v_emb_normed) * 0.5 + 0.5
        p_v_con_o = p_v_con_o.permute(0, 2, 1)
        p_o_con_v = torch.einsum('bnc,mc->bnm', f_o_e_v_norm, o_emb_normed) * 0.5 + 0.5
        return p_v_con_o, p_o_con_v


#######################
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

# class AdaptedIV2VisionBlock(Block):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
#                  fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
#                  use_fused_rmsnorm=False, 
#                  num_frames=8, tubelet_size=1, scale=1., num_tadapter=1):
        
#         super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, init_values,
#                          drop_path, act_layer, norm_layer, use_flash_attn, use_fused_mlp,
#                          fused_mlp_heuristic, with_cp, qk_normalization, layerscale_no_force_fp32,
#                          use_fused_rmsnorm)

#         self.num_frames = num_frames
#         self.tubelet_size = tubelet_size
#         self.scale = scale
#         self.num_tadapter = num_tadapter

#         # Initialize Adapters
#         # S_Adapter: Spatial Adapter (Applied to Attention output)
#         self.S_Adapter = Adapter(dim)
        
#         # MLP_Adapter: Applied parallel to MLP
#         self.MLP_Adapter = Adapter(dim, skip_connect=False)
        
#         # T_Adapter: Temporal Adapter
#         self.T_Adapter = Adapter(dim, skip_connect=False)
#         if num_tadapter == 2:
#             self.T_Adapter_in = Adapter(dim)

#         # set True
#         self.use_t = True
#         # use_fused_rmsnorm need to maintain same residual size, but we use x_vid without CLS
#         self.norm_adapter = norm_layer(dim)
#         self.norm_mlp = norm_layer(dim)

#     def forward(self, x, residual=None):
#         # InternVideo2의 Fused RMSNorm/MLP 구조 때문에 forward를 재구성해야 합니다.
#         # Adapter 주입을 위해 Fused 연산을 일부 풀거나, Adapter용 Norm을 따로 사용합니다.

#         # --------------------------------------------------------
#         # Part 1: Attention + Spatial Adapter + Temporal Adapter
#         # --------------------------------------------------------
#         # Original: x = x + drop_path(ls1(attn(norm1(x))))

#         # internvideo2 block
#         # if self.use_fused_rmsnorm:
#         #     x, residual = self.norm1(x, residual)
#         #     x = self.drop_path1(self.ls1(self.attn(x)))
#         #     x, residual = self.norm2(x, residual)
#         #     x = self.drop_path2(self.ls2(self.mlp(x)))
#         #     return x, residual
#         # update residual state after fused_rmsnorm

#         # x: [B, N, C] where N = 1 (CLS) + T * L
#         B, N, C = x.shape
#         L_total = N - 1
#         T = self.num_frames // self.tubelet_size
#         L = L_total // T

#         if isinstance(x, tuple) and len(x) == 2:
#             x, residual = x

#         # Temporal Adapter
#         if self.use_t:
#             cls_token = x[:, :1, :]
#             x_vid = x[:, 1:, :] # [B, T*L, C]
            
#             xt = rearrange(x_vid, 'b (t l) c -> t (b l) c', t=T, b=B, c=C)
#             if self.num_tadapter == 2:
#                 if self.use_fused_rmsnorm:
#                     xt = self.norm_adapter(xt)
#                     if isinstance(xt, tuple) and len(xt) == 2:
#                         xt, _ = xt
#                     xt = self.T_Adapter(self.ls1(self.attn(self.T_Adapter_in(xt))))
#                 else :
#                     xt = self.T_Adapter(self.ls1(self.attn(self.T_Adapter_in(self.norm1(xt)))))
#             else:
#                 if self.use_fused_rmsnorm:
#                     xt = self.norm_adapter(xt)
#                     xt = self.T_Adapter(self.ls1(self.attn(xt)))
#                 xt = self.T_Adapter(self.ls1(self.attn(self.norm1(xt))))
#             xt = rearrange(xt, 't (b l) c -> b (t l) c', t=T, c=C, b=B)
#             x_vid = x_vid + self.drop_path1(xt)
#             x = torch.cat([cls_token, x_vid], dim=1)  # CLS 토큰에는 Temporal Adapter 적용 안함
#         else :
#             raise NotImplementedError()

#         # Spatial Adapter
#         if self.use_fused_rmsnorm:
#             x, residual = self.norm1(x, residual)
#             x = x + self.S_Adapter(self.ls1(self.attn(x)))
#         else :
#             x = x + self.S_Adapter(self.ls1(self.attn(self.norm1(x))))

#         # Joint Adapter
#         if self.use_fused_rmsnorm:
#             xn, residual = self.norm2(x, residual)
#         else :
#             xn = self.norm2(x)
#         x = x + self.drop_path2(self.ls2(self.mlp(xn))) + self.drop_path2(self.scale * self.MLP_Adapter(xn))

#         if residual is not None :
#             return x, residual
#         return x


class AdaptedIV2VisionBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
                 fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
                 use_fused_rmsnorm=False, 
                 num_frames=8, tubelet_size=1, scale=1., num_tadapter=1):
        
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, init_values,
                         drop_path, act_layer, norm_layer, use_flash_attn, use_fused_mlp,
                         fused_mlp_heuristic, with_cp, qk_normalization, layerscale_no_force_fp32,
                         use_fused_rmsnorm)

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.scale = scale
        self.num_tadapter = num_tadapter

        # Initialize Adapters
        # S_Adapter: Spatial Adapter (Applied to Attention output)
        self.S_Adapter = Adapter(dim)
        
        # MLP_Adapter: Applied parallel to MLP
        # self.MLP_Adapter = Adapter(dim, skip_connect=False)
        
        # T_Adapter: Temporal Adapter
        # self.T_Adapter = Adapter(dim, skip_connect=False)
        # if num_tadapter == 2:
        #     self.T_Adapter_in = Adapter(dim)

        # set True
        self.use_t = False
        # use_fused_rmsnorm need to maintain same residual size, but we use x_vid without CLS
        self.norm_adapter = norm_layer(dim)
        self.norm_mlp = norm_layer(dim)

    def forward(self, x, residual=None):
        # InternVideo2의 Fused RMSNorm/MLP 구조 때문에 forward를 재구성해야 합니다.
        # Adapter 주입을 위해 Fused 연산을 일부 풀거나, Adapter용 Norm을 따로 사용합니다.

        # --------------------------------------------------------
        # Part 1: Attention + Spatial Adapter + Temporal Adapter
        # --------------------------------------------------------
        # Original: x = x + drop_path(ls1(attn(norm1(x))))

        # internvideo2 block
        # if self.use_fused_rmsnorm:
        #     x, residual = self.norm1(x, residual)
        #     x = self.drop_path1(self.ls1(self.attn(x)))
        #     x, residual = self.norm2(x, residual)
        #     x = self.drop_path2(self.ls2(self.mlp(x)))
        #     return x, residual
        # update residual state after fused_rmsnorm

        # x: [B, N, C] where N = 1 (CLS) + T * L
        B, N, C = x.shape
        L_total = N - 1
        T = self.num_frames // self.tubelet_size
        L = L_total // T

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x

        # Temporal Adapter
        if self.use_t:
            cls_token = x[:, :1, :]
            x_vid = x[:, 1:, :] # [B, T*L, C]
            
            xt = rearrange(x_vid, 'b (t l) c -> t (b l) c', t=T, b=B, c=C)
            if self.num_tadapter == 2:
                if self.use_fused_rmsnorm:
                    xt = self.norm_adapter(xt)
                    if isinstance(xt, tuple) and len(xt) == 2:
                        xt, _ = xt
                    xt = self.T_Adapter(self.ls1(self.attn(self.T_Adapter_in(xt))))
                else :
                    xt = self.T_Adapter(self.ls1(self.attn(self.T_Adapter_in(self.norm1(xt)))))
            else:
                if self.use_fused_rmsnorm:
                    xt = self.norm_adapter(xt)
                    xt = self.T_Adapter(self.ls1(self.attn(xt)))
                xt = self.T_Adapter(self.ls1(self.attn(self.norm1(xt))))
            xt = rearrange(xt, 't (b l) c -> b (t l) c', t=T, c=C, b=B)
            x_vid = x_vid + self.drop_path1(xt)
            x = torch.cat([cls_token, x_vid], dim=1)  # CLS 토큰에는 Temporal Adapter 적용 안함
        else :
            pass
            # raise NotImplementedError()

        # Spatial Adapter
        if self.use_fused_rmsnorm:
            x, residual = self.norm1(x, residual)
            x = x + self.S_Adapter(self.ls1(self.attn(x)))
        else :
            x = x + self.S_Adapter(self.ls1(self.attn(self.norm1(x))))

        # Joint Adapter
        if self.use_fused_rmsnorm:
            xn, residual = self.norm2(x, residual)
        else :
            xn = self.norm2(x)
        x = x + self.drop_path2(self.ls2(self.mlp(xn)))
        # x = x + self.drop_path2(self.ls2(self.mlp(xn))) + self.drop_path2(self.scale * self.MLP_Adapter(xn))

        if residual is not None :
            return x, residual
        return x


class InternVideo_AIM(InternVideo2):
    def __init__(self, 
                 adapt_star_layer=0, 
                 num_tadapter=1, 
                 adapter_scale=0.5, 
                 kwargs=None):
        # InternVideo2 초기화
        super().__init__(**kwargs)
        
        self.adapt_star_layer = adapt_star_layer
        self.num_tadapter = num_tadapter
        self.adapter_scale = adapter_scale
        
        # Block 교체 로직
        # 기존 blocks를 순회하며 adapt_star_layer 이상인 경우 AdaptedBlock으로 교체
        new_blocks = []
        for i, blk in enumerate(self.blocks):
            if i >= adapt_star_layer:
                print(f"Replacing Layer {i} with AIM AdaptedBlock")
                
                # 기존 Block의 설정을 가져옴
                adapted_block = AdaptedIV2VisionBlock(
                    dim=blk.norm1.weight.shape[0],
                    num_heads=blk.attn.num_heads,
                    mlp_ratio=4.0, # InternVideo default approximation or extract from blk
                    qkv_bias=blk.attn.qkv.bias is not None,
                    drop=0., 
                    attn_drop=0.,
                    init_values=blk.ls1.gamma[0].item() if not isinstance(blk.ls1, nn.Identity) else None,
                    drop_path=blk.drop_path1.drop_prob if not isinstance(blk.drop_path1, nn.Identity) else 0.,
                    norm_layer=self.norm_layer_for_blocks, # InternVideo2에서 partial로 넘어온 것 사용
                    use_flash_attn=self.use_flash_attn,
                    use_fused_mlp=self.use_flash_attn, # kwargs나 config에서 가져와야 정확함, 여기선 True 가정
                    use_fused_rmsnorm=self.use_flash_attn,
                    qk_normalization=blk.attn.qk_normalization, # Fix: Pass qk_normalization from original block
                    # AIM Specifics
                    num_frames=kwargs.get('num_frames', 8),
                    tubelet_size=kwargs.get('tubelet_size', 1),
                    scale=adapter_scale,
                    num_tadapter=num_tadapter
                )
                new_blocks.append(adapted_block)
            else:
                new_blocks.append(blk)
        
        self.blocks = nn.ModuleList(new_blocks)
        
        # Adapter 가중치 초기화 (Zero Initialization)
        self.init_adapter_weights()

    def init_adapter_weights(self):
        for n, m in self.blocks.named_modules():
            # S_Adapter Init
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
            
            # T_Adapter Init
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

            # MLP_Adapter Init
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
        print("Initialized AIM Adapter weights to zero.")


def get_internvideo_aim(cfg, kwargs):
    vision_width = getattr(cfg, 'vision_width', 1408) # InternVideo2-6B size default
    depth = getattr(cfg, 'depth', 40)
    
    model = InternVideo_AIM(
        adapt_star_layer=cfg.adapt_star_layer,
        num_tadapter=cfg.num_tadapter,
        adapter_scale=0.5,
        # InternVideo2 Args
        kwargs=kwargs
    )
    return model

#######################
def load(clip_model_name="ViT-B/16", config=None, device=None):
    model = InternVideo_C2C(clip_model_name, config, device)
    print("==="*20)
    print(model)
    print("==="*20)
    return model