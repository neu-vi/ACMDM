import torch
import torch.nn as nn
from models.ACMDM import ACMDM
from models.ACMDM import TimestepEmbedder, ACMDMTransBlock, LlamaRMSNorm
from models.ROPE import RopeND
from utils.eval_utils import eval_decorator
from utils.train_utils import lengths_to_mask


#################################################################################
#                                  ACMDM+ControlNet                             #
#################################################################################
class ACMDM_ControlNet(ACMDM):
    def __init__(self, input_dim, cond_mode, base_checkpoint, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.2, clip_dim=512,
                 diff_model='Flow', cond_drop_prob=0.1, max_length=49,
                 patch_size=(1, 22), stride_size=(1, 22),
                 clip_version='ViT-B/32', freeze_base=True, need_base=True, **kargs):
        # --------------------------------------------------------------------------
        # ACMDM
        super().__init__(input_dim, cond_mode, latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                 num_heads=num_heads, dropout=dropout, clip_dim=clip_dim,
                 diff_model=diff_model, cond_drop_prob=cond_drop_prob, max_length=max_length,
                 patch_size=patch_size, stride_size=stride_size,
                 clip_version=clip_version, **kargs)

        # --------------------------------------------------------------------------
        # ControlNet
        self.c_t_embedder = TimestepEmbedder(self.latent_dim)
        self.c_control_embedder = c_control_embedder(3, self.latent_dim, patch_size=self.patch_size,
                                                     stride_size=self.stride_size)
        self.c_x_embedder = nn.Conv2d(self.input_dim, self.latent_dim, kernel_size=self.patch_size,
                                      stride=self.stride_size, bias=True)
        self.c_y_embedder = nn.Linear(self.clip_dim, self.latent_dim)
        self.c_rope = RopeND(nd=1, nd_split=[1], max_lens=self.max_lens)
        self.ControlNet = nn.ModuleList([
            ACMDMTransBlock(self.latent_dim, num_heads, mlp_size=ff_size, rope=self.c_rope, qk_norm=True) for _ in
            range(num_layers)
        ])
        self.zero_Linear = nn.ModuleList([
            nn.Linear(self.latent_dim, self.latent_dim) for _ in range(num_layers)
        ])
        self.initialize_weights_control()
        if need_base:
            for key, value in list(base_checkpoint['ema_acmdm'].items()):
                if key.startswith('ACMDMTransformer.'):
                    new_key = key.replace('ACMDMTransformer.', 'ControlNet.')
                    base_checkpoint['ema_acmdm'][new_key] = value.clone()
            missing_keys, unexpected_keys = self.load_state_dict(base_checkpoint['ema_acmdm'], strict=False)
            assert len(unexpected_keys) == 0

        if self.cond_mode == 'text':
            print('ReLoading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        if freeze_base:
            for param in self.t_embedder.parameters():
                param.requires_grad = False
            for param in self.x_embedder.parameters():
                param.requires_grad = False
            for param in self.y_embedder.parameters():
                param.requires_grad = False
            for param in self.final_layer.parameters():
                param.requires_grad = False
            for param in self.ACMDMTransformer.parameters():
                param.requires_grad = False

    def initialize_weights_control(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.ACMDMTransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.c_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.c_t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.ControlNet:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.c_control_embedder.zero_linear.weight, 0)
        nn.init.constant_(self.c_control_embedder.zero_linear.bias, 0)

        for block in self.zero_Linear:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)

    def forward_with_control(self, x, t, conds, attention_mask, cfg1=1.0, cfg2=1.0, control=None, index=None,
                             force_mask=False):
        if not (cfg1 == 1.0 and cfg2 == 1.0):
            half = x[: len(x) // 3]
            x = torch.cat([half, half, half], dim=0)
        # controlnet
        c_t = self.c_t_embedder(t, dtype=x.dtype)
        conds = self.mask_cond(conds, force_mask=force_mask)
        c_control = self.c_control_embedder(control * index)
        if self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(c_control.shape[0], device=c_control.device) * self.cond_drop_prob).view(c_control.shape[0], 1, 1)
            c_control = c_control * (1. - mask)
        if not (cfg1 == 1.0 and cfg2 == 1.0):
            c_control = torch.cat([c_control, c_control, torch.zeros_like(c_control)], dim=0)
        c_x = self.c_x_embedder(x).flatten(2).transpose(1, 2)
        c_y = self.c_y_embedder(conds)
        c_y = c_t.unsqueeze(1) + c_y.unsqueeze(1)
        c_x = c_x + c_control
        c_position_ids = self.position_ids_precompute[:, :c_x.shape[1]]
        c_out = []
        for c_block, c_linear in zip(self.ControlNet, self.zero_Linear):
            c_x = c_block(c_x, c_y, attention_mask, position_ids=c_position_ids)
            c_out.append(c_linear(c_x))
        # main branch
        tt = self.t_embedder(t, dtype=x.dtype)
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)
        conds = self.y_embedder(conds)
        y = tt.unsqueeze(1) + conds.unsqueeze(1)
        position_ids = self.position_ids_precompute[:, :x.shape[1]]
        # merging
        for block, c in zip(self.ACMDMTransformer, c_out):
            x = block(x, y, attention_mask, position_ids=position_ids)
            x = x + c
        x = self.final_layer(x, y)
        if not (cfg1 == 1.0 and cfg2 == 1.0):
            cond_eps, uncond_eps1, uncond_eps2 = torch.split(x, len(x) // 3, dim=0)
            half_eps = cond_eps + (cfg1-1) * (cond_eps - uncond_eps1) + (cfg2-1) * (cond_eps - uncond_eps2)
            x = torch.cat([half_eps, half_eps, half_eps], dim=0)
        return x

    def forward_control_loss(self, latents, y, m_lens, original, index, ae, mean_std):
        latents = latents.permute(0, 2, 3, 1)
        b, l, j, d = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1).unsqueeze(-1), latents, torch.zeros_like(latents))

        target = latents.clone().permute(0, 3, 1, 2).detach()
        original = original.clone().detach()

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        attention_mask = non_pad_mask.unsqueeze(-1).repeat(1, 1, self.patches_per_frame).flatten(1).unsqueeze(1).unsqueeze(1)

        random_indices = torch.randint(0, len(index), (b,)).to(device)
        indexx = torch.tensor(index, device=device)[random_indices]
        mask_seq = torch.zeros((b, 3, l*4, j), device=device)
        for i in range(b):
            seq_num = torch.randint(1, m_lens[i]*4, (1,))
            choose_seq = torch.sort(torch.randperm(m_lens[i]*4)[:seq_num.item()]).values
            mask_seq[i, :, choose_seq, indexx[i]] = 1.0

        model_kwargs = dict(conds=cond_vector, attention_mask=attention_mask, control=original, index=mask_seq,
                            force_mask=force_mask, mean_std=mean_std)
        if self.diff_model == "Flow":
            loss_dict = self.train_diffusion.training_losses(self.forward_with_control, target, ae=ae,
                                                             model_kwargs=model_kwargs)
        else:
            t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
            loss_dict = self.train_diffusion.training_losses(self.forward_with_control, target, t, model_kwargs)
        loss = loss_dict["loss"]
        loss = (loss * non_pad_mask).sum() / non_pad_mask.sum()

        return loss, loss_dict["loss_control"]


    @torch.no_grad()
    @eval_decorator
    def generate_control(self,
                         conds,
                         m_lens,
                         control,
                         index,
                         density,
                         cond_scale,
                         temperature=1,
                         j=22
                         ):
        device = next(self.parameters()).device
        l = control.shape[2]//4
        b = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, l)

        noise = torch.randn(b, self.input_dim, l, j).to(device)
        control = control.clone()
        cfg1 = cond_scale[0]
        cfg2 = cond_scale[1]
        if not (cfg1 == 1.0 and cfg2 == 1.0):
            # (1) with text and with control (2) no text and with control (3) with text and no control
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector), cond_vector], dim=0)

        random_indices = torch.tensor(0, device=device).repeat(b) # no random in inference
        indexx = torch.tensor(index, device=device)[random_indices]
        mask_seq = torch.zeros((b, 3, l * 4, j), device=device)
        for i in range(b):
            if density in [1, 2, 5]:
                seq_num = density
            else:
                seq_num = int(m_lens[i] *4* density / 100)
            choose_seq = torch.sort(torch.randperm(m_lens[i] * 4)[:seq_num]).values
            mask_seq[i, :, choose_seq, indexx[i]] = 1.0

        attention_mask = (~padding_mask).unsqueeze(-1).repeat(1, 1, self.patches_per_frame).flatten(1).unsqueeze(1).unsqueeze(1)
        model_kwargs = dict(conds=cond_vector, attention_mask=attention_mask, cfg1=cfg1, cfg2=cfg2, index=mask_seq,
                            control=control)
        sample_fn = self.forward_with_control

        if not (cfg1 == 1.0 and cfg2 == 1.0):
            model_kwargs["attention_mask"] = attention_mask.repeat(3, 1, 1, 1)
            noise = torch.cat([noise, noise, noise], dim=0)

        if self.diff_model == "Flow":
            model_fn = self.gen_diffusion.sample_ode()  # default to ode sampling
            sampled_token_latent = model_fn(noise, sample_fn, **model_kwargs)[-1]
        else:
            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs,
                progress=False,
                temperature=temperature
            )
        if not (cfg1 == 1.0 and cfg2 == 1.0):
            sampled_token_latent, _, _ = sampled_token_latent.chunk(3, dim=0)
        sampled_token_latent = sampled_token_latent.permute(0, 2, 3, 1)

        latents = torch.where(padding_mask.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(sampled_token_latent),
                              sampled_token_latent)
        return latents.permute(0, 3, 1, 2), mask_seq

#################################################################################
#                                     ACMDM Zoos                                #
#################################################################################
def acmdm_raw_flow_s_ps22_control(**kwargs):
    layer = 8
    return ACMDM_ControlNet(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=49,
                 patch_size=(1, 22), stride_size=(1, 22), freeze_base=True, **kwargs)


ACMDM_ControlNet_Models = {
    'ACMDM-Flow-S-PatchSize22-ControlNet': acmdm_raw_flow_s_ps22_control,
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class c_control_embedder(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features,
            patch_size,
            stride_size,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_features, hidden_features, kernel_size=(4,patch_size[1]), stride=(4,stride_size[1]), bias=True)
        self.norm = LlamaRMSNorm(hidden_features, eps=1e-6)
        self.zero_linear = nn.Linear(hidden_features, hidden_features)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.zero_linear(x)
        return x