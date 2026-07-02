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
from .internvideo2 import pretrain_internvideo2_1b_patch14_224, interpolate_pos_embed_internvideo2_new
from .custom_clip_c2c_prompts import MLP, MLP_ST
from .bert.builder import build_bert
from .bert.tokenization_bert import BertTokenizer


class DictToObj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)


# class PromptLearner(nn.Module):
#     """
#     Learnable prompt module supporting both CoOp and CSP styles
#     """
#     def __init__(self, n_ctx, n_cls_ctx, ctx_init, classnames, tokenizer, 
#                  word_embedding, token_type_embeddings, positional_embedding, 
#                  prompt_config, device, learn_method='coop'):
#         super().__init__()
        
#         self.n_ctx = n_ctx  # number of context tokens
#         self.n_cls_ctx = n_cls_ctx  # number of class-specific tokens (for CSP)
#         self.device = device
#         self.dtype = torch.float32
#         self.learn_method = learn_method
#         self.n_cls = len(classnames)
        
#         hidden_size, hidden_dropout_prob, layer_norm_eps = prompt_config
#         self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.dropout = nn.Dropout(hidden_dropout_prob)
#         # Get embedding dimension
#         # token_embedding = token_embedding.to(device)
#         # ctx_dim = token_embedding.weight.shape[1]
        
#         # Tokenize class names (fixed)
#         prompts = [ctx_init + " " + name for name in classnames]
#         self.tokenized_prompts = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
               
#     def forward(self):

#         return self.tokenized_prompts


class PromptLearner(nn.Module):
    """
    Learnable prompt module supporting both CoOp and CSP styles
    """
    def __init__(self, n_ctx, n_cls_ctx, ctx_init, classnames, tokenizer, 
                 word_embedding, token_type_embeddings, positional_embedding, 
                 prompt_config, device, learn_method='coop'):
        super().__init__()
        
        self.n_ctx = n_ctx  # number of context tokens
        self.n_cls_ctx = n_cls_ctx  # number of class-specific tokens (for CSP)
        self.device = device
        self.dtype = torch.float32
        self.learn_method = learn_method
        self.n_cls = len(classnames)
        
        hidden_size, hidden_dropout_prob, layer_norm_eps = prompt_config
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # Get embedding dimension
        # token_embedding = token_embedding.to(device)
        # ctx_dim = token_embedding.weight.shape[1]
        
        if learn_method == 'coop':
            # CoOp: learnable context, fixed class tokens
            if ctx_init:
                # Use given words for initialization
                ctx_init = ctx_init.replace("_", " ")
                prompt = tokenizer(ctx_init, return_tensors="pt").to(device)
                with torch.no_grad():
                    embedding = word_embedding(prompt.input_ids).type(self.dtype)
                ctx_vectors = embedding[0, 1:1+n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # Random initialization
                raise NotImplementedError()
                        
            self.ctx = nn.Parameter(ctx_vectors)  # learnable
            
            # Tokenize class names (fixed)
            prompts = [prompt_prefix + " " + name for name in classnames]
            tokenized_prompts = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = word_embedding(tokenized_prompts.input_ids).type(self.dtype)
                input_shape = tokenized_prompts.input_ids.size()
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )
                token_type_embeddings_data = token_type_embeddings(token_type_ids)
            
            self.register_buffer("token_prefix", embedding[:, :1, :])  # [CLS]
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS + [EOS]
            self.register_buffer("token_type_embeddings", token_type_embeddings_data) 

            # trunc
            positional_embedding = positional_embedding[:input_shape[1]]
            if len(positional_embedding.shape) == 4 :
                self.positional_embedding = nn.Parameter(
                    positional_embedding.squeeze(0),
                    requires_grad=True
                )
            else  :
                self.positional_embedding = nn.Parameter(
                    positional_embedding.unsqueeze(0),
                    requires_grad=True
                )
        elif learn_method == 'zero':
            prompts = [name for name in classnames]
            self.tokenized_prompts = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        else :
            raise NotImplementedError(f"Unknown learn_method: {learn_method}") 
               
    def forward(self):
        """
        Returns contextualized prompts based on learning method
        """
        batch_size = self.n_cls
        
        if self.learn_method == 'zero':
            return self.tokenized_prompts

        if self.learn_method == 'coop':
            # CoOp: [CLS] + learnable_context + fixed_class + [EOS]
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(batch_size, -1, -1)
            
            prefix = self.token_prefix
            suffix = self.token_suffix
            
            # For variable length suffixes in CoOp
            # claude
            # prompts = []
            # for i in range(batch_size):
            #     prompt = torch.cat([prefix[i:i+1], ctx[i:i+1], suffix[i:i+1]], dim=1)
            #     prompts.append(prompt)
            # prompts = torch.stack(prompts).squeeze(1)
            # official repo
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )


        embeddings = prompts + self.token_type_embeddings
        embeddings += self.positional_embedding
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


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
        self.build_text_encoder()

        self.tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer_path)  #, local_files_only=True)  # "bert-base-uncased"
        
        self.vision_proj = nn.Linear(768, 512)
        self.text_proj = nn.Linear(1024, 512)

        self.load_checkpoint(config.iv_vision_ckpt_path)

        model_name = "C2C" if 'c2c' in config.method else ("AIM" if 'aim' in config.method else "BASE")
        print(f"Created {model_name} model with InternVideo2 backbone")


        self.label_index_file = getattr(config, "label_index_file", None)
        self.obj_label_index_file = getattr(config, "obj_label_index_file", None)
        self.prompt_learn_method = getattr(config, 'prompt_learn_method', None)  # class-specific context
        
        # if "coop" == self.prompt_learn_method :
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

        positional_embedding_data = self.text_encoder.bert.embeddings.position_embeddings.weight
        prompt_config = self.text_config.hidden_size, self.text_config.hidden_dropout_prob, self.text_config.layer_norm_eps

        # Initialize CoOp modules
        # Prompt learner for verb/action classes
        self.prompt_learner_verb = PromptLearner(
            n_ctx=self.n_ctx,
            n_cls_ctx=len(self.classnames),
            ctx_init=self.ctx_init,
            classnames=self.classnames,
            tokenizer=self.tokenizer,
            word_embedding=deepcopy(self.text_encoder.bert.embeddings.word_embeddings).to(self.device),
            token_type_embeddings=deepcopy(self.text_encoder.bert.embeddings.token_type_embeddings).to(self.device), 
            positional_embedding=positional_embedding_data.clone(), 
            prompt_config=prompt_config,
            device=self.device,
            learn_method=self.prompt_learn_method
        )
        
        # Prompt learner for object classes
        self.prompt_learner_obj = PromptLearner(
            n_ctx=self.n_ctx,
            n_cls_ctx=len(self.obj_classnames),
            ctx_init=self.ctx_init,
            classnames=self.obj_classnames,
            tokenizer=self.tokenizer,
            word_embedding=deepcopy(self.text_encoder.bert.embeddings.word_embeddings).to(self.device),
            token_type_embeddings=deepcopy(self.text_encoder.bert.embeddings.token_type_embeddings).to(self.device), 
            positional_embedding=positional_embedding_data.clone(), 
            prompt_config=prompt_config,
            device=self.device,
            learn_method=self.prompt_learn_method
        )
            
            # Generate tokenized prompts for classes
            # self.class_text_embedding = self._tokenize_prompts(self.classnames)
            # self.obj_class_text_embedding = self._tokenize_prompts(self.obj_classnames)
            ##################
        # else :
        #     raise NotImplementedError
        #     # Use original text encoder
        #     self.class_text_embedding = self.get_class_text_embedding(self.label_index_file)
        #     self.obj_class_text_embedding = self.get_class_text_embedding(self.obj_label_index_file, mode="obj")
        
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
        config_dict = dict(
            vision_encoder=dict(
                num_frames=getattr(self.config, "num_frames", 16), 
                tubelet_size=getattr(self.config, "tubelet_size", 1),
                clip_embed_dim=768,
                clip_teacher_embed_dim=3200,
                clip_teacher_final_dim=768,
                clip_norm_type='l2',
                clip_return_layer=6,
                clip_student_return_interval=1,
                use_checkpoint=getattr(self.config, "use_gradient_checkpoint", True),
                checkpoint_num=40,
                use_flash_attn=getattr(self.config, "use_flash_attn", False),
                use_fused_rmsnorm=getattr(self.config, "use_flash_attn", False),
                use_fused_mlp=getattr(self.config, "use_flash_attn", False),
                sep_image_video_pos_embed=True,
                pretrained=getattr(self.config, "iv_vision_ckpt_path", "")
            )
        )

        config = DictToObj(config_dict)
        self.vision_encoder = pretrain_internvideo2_1b_patch14_224(config)


    def build_text_encoder(self):
        encoder_name = self.config.bert_model

        if encoder_name == "bert_large" :
            config_dict = dict(
                vision_encoder=dict(
                    d_model=1408
                ),
                text_encoder=dict(
                    name="bert_large",
                    pretrained="bert-large-uncased",
                    config=self.config.bert_config_path,
                    d_model=1024,
                    fusion_layer=19,
                ),
                multimodal=dict(enable=True)
            )
        else :
            raise ValueError(f"Not implemented: {encoder_name}")
        
        config = DictToObj(config_dict)

        self.text_encoder, self.text_config = build_bert(
            config,
            True,  # self.is_pretrain
            getattr(self.config, "use_gradient_checkpoint", True),  # self.config.gradient_checkpointing,
        )
        

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


    def load_checkpoint(self, vision_ckpt_path=None):
        assert vision_ckpt_path is not None, "No vision_encoder checkpoint"

        # load vision_encoder
        print(f"Load pretrained checkpoint from {vision_ckpt_path}")
        ckpt = torch.load(vision_ckpt_path, map_location='cpu')['module']
        interpolate_pos_embed_internvideo2_new(ckpt, self.vision_encoder, orig_t_size=4)
        msg = self.load_state_dict(ckpt, strict=False)
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

    def get_text_encoder(self):
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder


    def encode_text(self, text):
        if self.prompt_learn_method == 'zero':
            text_output = self.get_text_encoder()(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state
            pooled_text_embeds = text_embeds[:, 0]
            return pooled_text_embeds

        elif self.prompt_learn_method == 'coop':
            text_output = self.get_text_encoder()(
                encoder_embeds=text,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state
            pooled_text_embeds = text_embeds[:, 0]
            return pooled_text_embeds


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
        
        # if "coop" == self.prompt_learn_method :
        verb_prompts = self.prompt_learner_verb()
        verb_text_features = self.encode_text(verb_prompts)
        obj_prompts = self.prompt_learner_obj()
        obj_text_features = self.encode_text(obj_prompts)
        # else :
            # raise NotImplementedError
            # verb_text_features = self.encode_text(self.class_text_embedding)
            # obj_text_features = self.encode_text(self.obj_class_text_embedding)
        
        if self.c2c_VE1 is None :
            ### BASE or AIM
            verb_text_features = verb_text_features / verb_text_features.norm(dim=-1, keepdim=True)
            obj_text_features = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)

            image_features = image_features.mean(dim=1)
            image_features = self.vision_proj(image_features)
            verb_text_features = self.text_proj(verb_text_features)
            obj_text_features = self.text_proj(obj_text_features)

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
            return verb_logits, obj_logits, com_logits, p_v_con_o, p_o_con_v

        return com_logits
    
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
def load(clip_model_name="ViT-B/16", config=None, device=None):
    model = InternVideo_C2C(clip_model_name, config, device)
    print("==="*20)
    print(model)
    print("==="*20)
    return model

