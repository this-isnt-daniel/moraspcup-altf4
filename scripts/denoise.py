#!/usr/bin/env python3
"""
denoise.py  — Mora SP Cup 2026 submission inference script.
Architecture: NAFNet-style Gated Residual U-Net (custom implementation).
Checkpoint  : sweep_outputs/run_lr3e4_ssim/checkpoints/run_lr3e4_ssim_best.pth
Alpha blend : 0.0 (pure neural, no wavelet blend)

USAGE:
    python scripts/denoise.py --noise_dir INPUT_DIR --denoised_dir OUTPUT_DIR
    python scripts/denoise.py --noise_dir INPUT_DIR --denoised_dir OUTPUT_DIR --device cpu
"""

import argparse, math, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn


# ── Architecture (NAFNet-style Gated Residual U-Net) ──────────────────────
class ChannelNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))
    def forward(self, x):
        dtype = x.dtype; z = x.float(); mu = z.mean(1, keepdim=True)
        z = (z - mu) * torch.rsqrt((z - mu).square().mean(1, keepdim=True) + 1e-6)
        return (z * self.w + self.b).to(dtype)

class GatedBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = ChannelNorm(c); self.n2 = ChannelNorm(c)
        self.expand = nn.Conv2d(c, 2*c, 1)
        self.depth  = nn.Conv2d(2*c, 2*c, 3, padding=1, groups=2*c)
        self.attn   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.project = nn.Conv2d(c, c, 1)
        self.f1 = nn.Conv2d(c, 2*c, 1); self.f2 = nn.Conv2d(c, c, 1)
        self.beta  = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
    def forward(self, x):
        a, b = self.depth(self.expand(self.n1(x))).chunk(2, 1); z = a * b
        x = x + self.beta * self.project(z * self.attn(z))
        a, b = self.f1(self.n2(x)).chunk(2, 1)
        return x + self.gamma * self.f2(a * b)

class Denoiser(nn.Module):
    def __init__(self, channels=3, width=24, blocks=2):
        super().__init__(); self.channels = channels
        self.head = nn.Conv2d(channels, width, 3, padding=1)
        self.enc  = nn.ModuleList(); self.down = nn.ModuleList()
        self.up   = nn.ModuleList(); self.dec  = nn.ModuleList()
        c = width
        for _ in range(3):
            self.enc.append(nn.Sequential(*[GatedBlock(c) for _ in range(blocks)]))
            self.down.append(nn.Conv2d(c, 2*c, 2, stride=2)); c *= 2
        self.mid = nn.Sequential(*[GatedBlock(c) for _ in range(blocks + 2)])
        for _ in range(3):
            self.up.append(nn.Sequential(nn.Conv2d(c, 2*c, 1), nn.PixelShuffle(2))); c //= 2
            self.dec.append(nn.Sequential(*[GatedBlock(c) for _ in range(blocks)]))
        self.tail = nn.Conv2d(width, channels, 3, padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)
    def forward(self, x):
        h, w = x.shape[-2:]
        p = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode='replicate')
        z = self.head(p); skips = []
        for enc, down in zip(self.enc, self.down): z = enc(z); skips.append(z); z = down(z)
        z = self.mid(z)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)): z = dec(up(z) + s)
        return (p + self.tail(z))[..., :h, :w]


# ── Inference helpers ──────────────────────────────────────────────────────
def tiled(model, x, tile=384, overlap=64):
    h, w = x.shape[-2:]
    if h <= tile and w <= tile:
        return model(x).float()
    def starts(n):
        if n <= tile: return [0]
        return sorted(set(list(range(0, n - tile + 1, tile - overlap)) + [n - tile]))
    out = torch.zeros_like(x, dtype=torch.float32)
    wts = torch.zeros_like(x[:, :1], dtype=torch.float32)
    for top in starts(h):
        for left in starts(w):
            patch = x[..., top:top+tile, left:left+tile]; ph, pw = patch.shape[-2:]
            wy = torch.hann_window(ph, periodic=False, device=x.device).clamp_min(.05)
            wx = torch.hann_window(pw, periodic=False, device=x.device).clamp_min(.05)
            wt = (wy[:, None] * wx[None, :])[None, None]
            out[..., top:top+ph, left:left+pw] += model(patch).float() * wt
            wts[..., top:top+ph, left:left+pw] += wt
    return out / wts

@torch.inference_mode()
def predict(model, array, device, tile=384, overlap=64):
    x = torch.from_numpy(np.ascontiguousarray(
        array.transpose(2, 0, 1))).float()[None].to(device) / 255
    model.eval()
    result = tiled(model, x, tile, overlap).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return result

def read_image(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 2: a = a[..., None]
    return np.array(a, copy=True)

def write_image(path, arr):
    out = np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out[..., 0] if out.shape[-1] == 1 else out).save(path)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--noise_dir',    required=True,  type=Path)
    ap.add_argument('--denoised_dir', required=True,  type=Path)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('WARNING: CUDA unavailable, falling back to CPU.', file=sys.stderr)
        device = torch.device('cpu')
    print(f'Device: {device}', flush=True)

    # Resolve checkpoint relative to this script's location
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent
    ckpt_path  = repo_root / 'sweep_outputs/run_lr3e4_ssim/checkpoints/run_lr3e4_ssim_best.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    c     = state['config']
    model = Denoiser(c['channels'], c['width'], c['blocks'])
    model.load_state_dict(state['model'])
    model = model.to(device).eval()
    print(f'Loaded: {ckpt_path.name} (channels={c["channels"]}, '
          f'width={c["width"]}, blocks={c["blocks"]})', flush=True)

    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(args.noise_dir.glob('*_noise.png'))
    if not inputs:
        raise FileNotFoundError(f'No *_noise.png files found in {args.noise_dir}')

    t0 = time.perf_counter()
    for p in inputs:
        img_id  = p.stem[:-len('_noise')]   # e.g. "461_noise" -> "461"
        out_name = f'{img_id}.png'
        arr  = read_image(p)
        pred = predict(model, arr, device, tile=384, overlap=64)
        write_image(args.denoised_dir / out_name, pred)
        print(f'  {p.name} -> {out_name}  '
              f'({(time.perf_counter()-t0)*1000/max(1,inputs.index(p)+1):.0f} ms/img avg)',
              flush=True)
    total = time.perf_counter() - t0
    print(f'Done: {len(inputs)} images in {total:.1f}s '
          f'({total/len(inputs)*1000:.0f} ms/image, device={device})', flush=True)

if __name__ == '__main__':
    main()
