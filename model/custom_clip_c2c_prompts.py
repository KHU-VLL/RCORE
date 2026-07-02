# Modified CLIP VisionTransformer with CoOp-style prompt tuning
import json
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from clip.model import LayerNorm, Transformer

from model.AIM import get_aim


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True,
                 dropout=False, norm=False, layers=[]):
        super().__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            outgoing = incoming if len(layers) == 0 else layers[layer_ind]
            mod.append(nn.Linear(incoming, outgoing, bias=bias))
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))
        mod.append(nn.Linear(incoming, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)


class MLP_ST(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True,
                 dropout=False, norm=False, layers=[]):
        super().__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            outgoing = incoming if len(layers) == 0 else layers[layer_ind]
            mod.append(nn.Conv1d(incoming, outgoing, kernel_size=3, bias=bias, padding=1))
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))
        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))
        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else:
                x = o(x)
        return x


class PromptLearner(nn.Module):
    """Learnable prompt module (CoOp)."""

    def __init__(self, n_ctx, n_cls_ctx, ctx_init, classnames, token_embedding,
                 positional_embedding, device, learn_method='coop'):
        super().__init__()

        self.n_ctx = n_ctx
        self.n_cls_ctx = n_cls_ctx
        self.device = device
        self.dtype = torch.float32
        self.learn_method = learn_method
        self.n_cls = len(classnames)

        token_embedding = token_embedding.to(device)
        ctx_dim = token_embedding.weight.shape[1]

        if len(positional_embedding.shape) == 4:
            self.positional_embedding = nn.Parameter(
                positional_embedding.squeeze(0), requires_grad=True,
            )
        else:
            self.positional_embedding = nn.Parameter(
                positional_embedding.unsqueeze(0), requires_grad=True,
            )

        if learn_method != 'coop':
            raise NotImplementedError(f"Unknown learn_method: {learn_method}")

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        name_lens = [len(clip.tokenize(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])           # [CLS]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])   # class + [EOS]
        self.name_lens = name_lens

    def forward(self):
        # CoOp: [CLS] + learnable_context + fixed_class + [EOS]
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        prompts += self.positional_embedding
        return prompts


class CLIP_C2C(nn.Module):
    """CLIP wrapper with a modified vision transformer and CoOp-style prompt tuning."""

    def __init__(self, clip_model_name="ViT-B/16", config=None, device=None):
        super().__init__()

        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load original CLIP model
        original_model, _ = clip.load(clip_model_name, device=self.device)
        original_state_dict = original_model.state_dict()
        self.dtype = original_model.dtype

        config.vision_width = original_state_dict["visual.conv1.weight"].shape[0]
        config.output_dim = original_state_dict["text_projection"].shape[1]

        self.video_encoder = get_aim(config).to(self.device)
        self.video_encoder.proj = original_model.visual.proj

        self.context_length = original_model.context_length
        transformer_width = original_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        embed_dim = original_state_dict["text_projection"].shape[1]

        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = original_model.logit_scale

        self.text_encoder = Transformer(
            width=transformer_width,
            layers=12,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        # CoOp configuration
        self.n_ctx = getattr(config, 'n_ctx', 4)
        self.ctx_init = getattr(config, 'input_template', None)
        self.prompt_learn_method = getattr(config, 'prompt_learn_method', None)
        assert self.prompt_learn_method == 'coop', \
            "prompt_learn_method must be 'coop'"

        print("Created C2C model with AIM adapter")
        print(f"{self.prompt_learn_method} enabled with {self.n_ctx} context tokens")

        self.label_index_file = getattr(config, "label_index_file", None)
        self.obj_label_index_file = getattr(config, "obj_label_index_file", None)

        self.classnames, self.obj_classnames = self._get_classnames()

        self.prompt_learner_verb = PromptLearner(
            n_ctx=self.n_ctx,
            n_cls_ctx=len(self.classnames),
            ctx_init=self.ctx_init,
            classnames=self.classnames,
            token_embedding=deepcopy(original_model.token_embedding),
            positional_embedding=original_model.positional_embedding.clone(),
            device=self.device,
            learn_method=self.prompt_learn_method,
        )
        self.prompt_learner_obj = PromptLearner(
            n_ctx=self.n_ctx,
            n_cls_ctx=len(self.obj_classnames),
            ctx_init=self.ctx_init,
            classnames=self.obj_classnames,
            token_embedding=deepcopy(original_model.token_embedding),
            positional_embedding=original_model.positional_embedding.clone(),
            device=self.device,
            learn_method=self.prompt_learn_method,
        )

        self.class_text_embedding = self._tokenize_prompts(self.classnames)
        self.obj_class_text_embedding = self._tokenize_prompts(self.obj_classnames)

        self.save_features = getattr(config, 'save_features', False)
        self.save_visual_features = getattr(config, 'save_visual_features', False)
        self.temporal_modeling_shuffle = getattr(config, 'temporal_modeling_shuffle', False)

        # C2C modules
        try:
            fc_emb = config.fc_emb.split(',')
        except AttributeError:
            fc_emb = [config.fc_emb]
        layers = [int(a) for a in fc_emb]

        mlp_kwargs = dict(relu=config.relu, num_layers=config.nlayers,
                          dropout=False, norm=True, layers=layers)
        self.c2c_OE1 = MLP(config.feat_dim, int(config.emb_dim), **mlp_kwargs)
        self.c2c_OE2 = MLP(config.feat_dim, int(config.emb_dim), **mlp_kwargs)
        self.c2c_VE1 = MLP_ST(config.feat_dim, int(config.emb_dim), **mlp_kwargs)
        self.c2c_VE2 = MLP_ST(config.feat_dim, int(config.emb_dim), **mlp_kwargs)

        # Composition modules
        self.c2c_f_v_e_o_com = nn.Linear(2 * config.emb_dim, config.emb_dim, bias=True)
        self.c2c_f_o_e_v_com = nn.Linear(2 * config.emb_dim, config.emb_dim, bias=True)
        self.c2c_text_v = nn.Linear(config.feat_dim, config.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(config.feat_dim, config.emb_dim, bias=True)

    def _get_classnames(self):
        if self.label_index_file is not None:
            with open(self.label_index_file, 'r') as f:
                classnames = list(json.load(f).values())
        else:
            classnames = self.config.attrs if hasattr(self.config, 'attrs') else []

        if self.obj_label_index_file is not None:
            with open(self.obj_label_index_file, 'r') as f:
                obj_classnames = list(json.load(f).values())
        else:
            obj_classnames = self.config.objs if hasattr(self.config, 'objs') else []

        classnames = [c.replace("[", "").replace("]", "") for c in classnames]
        print("Class names:", classnames)
        print("Object class names:", obj_classnames)
        return classnames, obj_classnames

    def _tokenize_prompts(self, classnames):
        """Tokenize prompts with placeholder for learnable context."""
        prompts = [" ".join(["X"] * self.n_ctx) + f" {name}." for name in classnames]
        return torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)

    def build_attention_mask(self):
        # Causal attention mask; additive (-inf on upper triangle)
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_image(self, image, return_all=False):
        x = self.video_encoder(image, return_all=True)
        if self.video_encoder.proj is not None:
            x = x @ self.video_encoder.proj.float()
        if return_all:
            return x
        return x[:, 0]

    def encode_text(self, prompts, tokenized_prompts):
        x = prompts.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # Take features from [EOS] token
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

    def _compute_shuffled_outputs(self, image_features, v_feat_normed, verb_text_features, B, T):
        """Compute [cosine_similarity_loss, random_shuffled_logits, cosine_similarity_loss]."""
        random_indices = torch.rand(B, T).argsort(dim=1)
        reverse_indices = torch.arange(T).flip(dims=[0]).expand(B, -1)

        random_indices = random_indices.unsqueeze(1).expand(-1, image_features.shape[1], -1).to(image_features.device)
        random_shuffled_x = image_features.clone().gather(2, random_indices)
        random_shuffled_v_feat = self.c2c_VE1(random_shuffled_x).mean(dim=-1)

        reverse_indices = reverse_indices.unsqueeze(1).expand(-1, image_features.shape[1], -1).to(image_features.device)
        reverse_shuffled_x = image_features.clone().gather(2, reverse_indices)
        reverse_shuffled_v_feat = self.c2c_VE1(reverse_shuffled_x).mean(dim=-1)

        random_shuffled_v_feat_normed = F.normalize(random_shuffled_v_feat, dim=1)
        reverse_shuffled_v_feat_normed = F.normalize(reverse_shuffled_v_feat, dim=1)

        shuffled_verb_loss = F.cosine_similarity(v_feat_normed, reverse_shuffled_v_feat_normed, dim=-1)
        shuffled_verb_logits = random_shuffled_v_feat_normed @ verb_text_features.t() * 0.5 + 0.5
        return [shuffled_verb_loss, shuffled_verb_logits, shuffled_verb_loss]


    def forward(self, x):
        B, T, C, H, W = x.shape

        image_features = self.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.view(B, T, -1).permute(0, 2, 1)

        # Text features (verb + object)
        verb_prompts = self.prompt_learner_verb()
        verb_text_features = self.encode_text(verb_prompts, self.class_text_embedding)
        verb_text_features = self.c2c_text_v(verb_text_features)
        verb_text_features = verb_text_features / verb_text_features.norm(dim=-1, keepdim=True)

        obj_prompts = self.prompt_learner_obj()
        obj_text_features = self.encode_text(obj_prompts, self.obj_class_text_embedding)
        obj_text_features = self.c2c_text_o(obj_text_features)
        obj_text_features = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)

        # Component features
        o_feat_normed = F.normalize(self.c2c_OE1(image_features.mean(dim=-1)), dim=1)
        v_feat = self.c2c_VE1(image_features).mean(dim=-1)
        v_feat_normed = F.normalize(v_feat, dim=1)

        shuffled_verb_outputs = None
        if self.temporal_modeling_shuffle:
            shuffled_verb_outputs = self._compute_shuffled_outputs(
                image_features, v_feat_normed, verb_text_features, B, T,
            )

        verb_logits = v_feat_normed @ verb_text_features.t() * 0.5 + 0.5
        obj_logits = o_feat_normed @ obj_text_features.t() * 0.5 + 0.5

        # Composition features
        o_feat_c = self.c2c_OE2(image_features.mean(dim=-1))
        v_feat_c = self.c2c_VE2(image_features).mean(dim=-1)

        if self.save_visual_features:
            return (
                v_feat_normed, o_feat_normed,
                F.normalize(v_feat_c, dim=-1), F.normalize(o_feat_c, dim=-1),
            )

        b = B
        c = verb_text_features.shape[-1]
        n_v = verb_logits.shape[-1]
        n_o = obj_logits.shape[-1]

        p_v_con_o, p_o_con_v = self.condition_module(
            v_feat_c, o_feat_c, verb_text_features, obj_text_features, n_o, b, c, n_v,
        )
        p_pair_o = p_v_con_o * obj_logits.unsqueeze(1)   # b, nv, no
        p_pair_v = p_o_con_v * verb_logits.unsqueeze(-1)  # b, nv, no
        com_logits = (p_pair_o + p_pair_v).reshape(b, -1)

        if self.training:
            if self.temporal_modeling_shuffle:
                return verb_logits, obj_logits, com_logits, shuffled_verb_outputs
            return verb_logits, obj_logits, com_logits

        if self.save_features:
            return verb_logits, obj_logits, com_logits

        return com_logits

    def condition_module(self, v_feat_c, o_feat_c, v_emb, o_emb, n_o, b, c, n_v):
        v_emb_normed = F.normalize(v_emb, dim=1)
        o_emb_normed = F.normalize(o_emb, dim=1)

        f_v_e_o = self.c2c_f_v_e_o_com(
            torch.cat([
                v_feat_c.unsqueeze(1).repeat(1, n_o, 1),
                o_emb.unsqueeze(0).repeat(b, 1, 1),
            ], dim=-1).view(-1, c * 2),
        )
        f_v_e_o_norm = F.normalize(f_v_e_o, dim=-1).view(b, n_o, c)

        f_o_e_v = self.c2c_f_o_e_v_com(
            torch.cat([
                o_feat_c.unsqueeze(1).repeat(1, n_v, 1),
                v_emb.unsqueeze(0).repeat(b, 1, 1),
            ], dim=-1).view(-1, c * 2),
        )
        f_o_e_v_norm = F.normalize(f_o_e_v, dim=-1).view(b, n_v, c)

        p_v_con_o = torch.einsum('bnc,mc->bnm', f_v_e_o_norm, v_emb_normed) * 0.5 + 0.5  # b, no, nv
        p_v_con_o = p_v_con_o.permute(0, 2, 1)                                            # b, nv, no
        p_o_con_v = torch.einsum('bnc,mc->bnm', f_o_e_v_norm, o_emb_normed) * 0.5 + 0.5  # b, nv, no

        return p_v_con_o, p_o_con_v


def load(clip_model_name="ViT-B/16", config=None, device=None):
    """Load CLIP-based C2C model initialized with pretrained CLIP weights."""
    model = CLIP_C2C(clip_model_name, config, device)

    original_model, _ = clip.load(clip_model_name, device='cpu')
    original_state_dict = original_model.state_dict()
    for k, v in list(original_state_dict.items()):
        if k.startswith('visual.'):
            original_state_dict[k.replace("visual.", "video_encoder.")] = v
            del original_state_dict[k]
        elif k.startswith('transformer.'):
            original_state_dict[k.replace("transformer.", "text_encoder.")] = v
            del original_state_dict[k]

    msg = model.load_state_dict(original_state_dict, strict=False)
    print("Initialize model with pretrained CLIP weights")
    print(msg)

    return model
