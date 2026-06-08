import math
import torch
from torch import nn
import torch.nn.functional as F


def _sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2  = nn.GroupNorm(8, out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        return self.conv2(h) + self.skip(x)


class _AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, -1).transpose(1, 2)   # (B, HW, C)
        k = k.reshape(B, C, -1).transpose(1, 2)
        v = v.reshape(B, C, -1).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) * (C ** -0.5), dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class _UNet(nn.Module):
    """
    Two-level U-Net for 28x28 MNIST: 28->14->7->14->28.
    Conditioned on diffusion timestep via sinusoidal embeddings.
    ~1.6M parameters with base_ch=32.
    """
    def __init__(self, base_ch=32, t_dim=128):
        super().__init__()
        ch = base_ch
        td = t_dim * 2
        self._t_dim = t_dim

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, td),
            nn.SiLU(),
            nn.Linear(td, td),
        )

        # encoder
        self.conv_in = nn.Conv2d(1, ch, 3, padding=1)
        self.enc1a   = _ResBlock(ch,     ch,     td)
        self.enc1b   = _ResBlock(ch,     ch,     td)
        self.down1   = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)    # 28 -> 14
        self.enc2a   = _ResBlock(ch * 2, ch * 2, td)
        self.enc2b   = _ResBlock(ch * 2, ch * 2, td)
        self.down2   = nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1) # 14 -> 7

        # bottleneck with self-attention
        self.mid1      = _ResBlock(ch * 4, ch * 4, td)
        self.mid_attn  = _AttnBlock(ch * 4)
        self.mid2      = _ResBlock(ch * 4, ch * 4, td)

        # decoder with skip connections
        self.up2   = nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1)  # 7 -> 14
        self.dec2a = _ResBlock(ch * 4, ch * 2, td)   # ch*4: upsampled + skip
        self.dec2b = _ResBlock(ch * 2, ch * 2, td)
        self.up1   = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)      # 14 -> 28
        self.dec1a = _ResBlock(ch * 2, ch,     td)   # ch*2: upsampled + skip
        self.dec1b = _ResBlock(ch,     ch,     td)

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.t_mlp(_sinusoidal_embedding(t, self._t_dim))

        h  = self.conv_in(x)
        s1 = self.enc1b(self.enc1a(h, t_emb), t_emb)   # (B, ch,   28, 28)
        h  = self.down1(s1)
        s2 = self.enc2b(self.enc2a(h, t_emb), t_emb)   # (B, ch*2, 14, 14)
        h  = self.down2(s2)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)                          # (B, ch*4, 7,  7)

        h = torch.cat([self.up2(h), s2], dim=1)          # (B, ch*4, 14, 14)
        h = self.dec2b(self.dec2a(h, t_emb), t_emb)      # (B, ch*2, 14, 14)
        h = torch.cat([self.up1(h), s1], dim=1)           # (B, ch*2, 28, 28)
        h = self.dec1b(self.dec1a(h, t_emb), t_emb)      # (B, ch,   28, 28)

        return self.conv_out(F.silu(self.norm_out(h)))    # (B, 1,    28, 28)


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020) for 28x28 MNIST.
    Linear beta schedule; U-Net predicts noise epsilon at each diffusion step.
    Data is mapped to [-1, 1] internally for the diffusion process.
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, base_ch=32):
        super().__init__()
        self.T   = T
        self.net = _UNet(base_ch=base_ch)

        betas     = torch.linspace(beta_start, beta_end, T)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, 0)

        self.register_buffer('betas',             betas)
        self.register_buffer('alphas',            alphas)
        self.register_buffer('alpha_bar',         alpha_bar)
        self.register_buffer('sqrt_ab',           alpha_bar.sqrt())
        self.register_buffer('sqrt_one_minus_ab', (1.0 - alpha_bar).sqrt())

    def _scale(self, x):
        return 2.0 * x - 1.0          # [0, 1] -> [-1, 1]

    def _unscale(self, x):
        return (x.clamp(-1.0, 1.0) + 1.0) * 0.5  # [-1, 1] -> [0, 1]

    def forward(self, x):
        """
        Simplified DDPM training loss: MSE between predicted and actual noise.
        Returned loss is mean over batch and pixels (not summed like ELBO).
        """
        x = self._scale(x)
        B = x.size(0)
        t = torch.randint(0, self.T, (B,), device=x.device)
        eps = torch.randn_like(x)
        x_t = (self.sqrt_ab[t, None, None, None] * x
               + self.sqrt_one_minus_ab[t, None, None, None] * eps)
        return F.mse_loss(self.net(x_t, t), eps)

    @torch.no_grad()
    def sample(self, n, device, return_steps=False):
        """
        DDPM reverse-process sampling.  Returns images in [0, 1].
        If return_steps=True, also returns a list of ~10 intermediate states
        ordered from pure noise (index 0) to final clean image (last index).
        """
        x = torch.randn(n, 1, 28, 28, device=device)
        record_every = max(1, self.T // 10)
        record_at    = set(range(0, self.T, record_every)) | {self.T - 1}
        steps        = []

        for i in reversed(range(self.T)):
            t_vec    = torch.full((n,), i, device=device, dtype=torch.long)
            eps_pred = self.net(x, t_vec)

            ab_t    = self.alpha_bar[i]
            alpha_t = self.alphas[i]
            beta_t  = self.betas[i]
            ab_prev = self.alpha_bar[i - 1] if i > 0 else x.new_ones(1)

            # estimate x_0
            x0_pred = (x - self.sqrt_one_minus_ab[i] * eps_pred) / self.sqrt_ab[i]
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i > 0:
                mean = ((ab_prev.sqrt() * beta_t) / (1.0 - ab_t)) * x0_pred \
                     + ((alpha_t.sqrt() * (1.0 - ab_prev)) / (1.0 - ab_t)) * x
                var  = beta_t * (1.0 - ab_prev) / (1.0 - ab_t)
                x    = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = x0_pred

            if return_steps and i in record_at:
                # record from noisy (high i) to clean (i=0)
                steps.append(self._unscale(x).view(n, 1, 28, 28).cpu())

        final = self._unscale(x).view(n, 1, 28, 28)
        if return_steps:
            return final, steps   # steps[0] = near-noise, steps[-1] = clean
        return final
