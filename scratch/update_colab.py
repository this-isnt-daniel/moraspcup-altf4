import json
with open('Mora_SP_Cup_Colab_Denoising.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

NEW_CORE_IO = """import math, re
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

def read_image(path):
    with Image.open(path) as im:
        a=np.asarray(im)
        if im.mode not in ('RGB','L') or a.dtype!=np.uint8:
            raise ValueError(f'Only uint8 RGB/L supported: {path}, {im.mode}, {a.dtype}')
    if a.ndim==2: a=a[...,None]
    return np.array(a,copy=True)

def quantize(a):
    return np.rint(np.clip(a,0,1)*255).astype(np.uint8)

def write_image(path,a):
    a=quantize(a)
    Image.fromarray(a[...,0] if a.shape[-1]==1 else a).save(path)

def _fast_ssim(a, b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.ndim == 2: a = a[..., None]
    if b.ndim == 2: b = b[..., None]
    a_t = torch.from_numpy(a).float().permute(2,0,1).unsqueeze(0).to(device)
    b_t = torch.from_numpy(b).float().permute(2,0,1).unsqueeze(0).to(device)
    C = a_t.shape[1]
    weight = torch.ones((C, 1, 7, 7), device=device) / 49.0
    a_pad = F.pad(a_t, (3,3,3,3), mode='reflect')
    b_pad = F.pad(b_t, (3,3,3,3), mode='reflect')
    mu_a = F.conv2d(a_pad, weight, groups=C)
    mu_b = F.conv2d(b_pad, weight, groups=C)
    sigma_a_sq = (F.conv2d(a_pad**2, weight, groups=C) - mu_a**2) * (49.0/48.0)
    sigma_b_sq = (F.conv2d(b_pad**2, weight, groups=C) - mu_b**2) * (49.0/48.0)
    sigma_ab = (F.conv2d(a_pad*b_pad, weight, groups=C) - mu_a*mu_b) * (49.0/48.0)
    C1 = (0.01 * 1.0)**2; C2 = (0.03 * 1.0)**2
    ssim_map = ((2 * mu_a*mu_b + C1) * (2 * sigma_ab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2))
    return ssim_map.mean().item()

def _fast_psnr(a, b):
    mse = np.mean((a - b)**2)
    return float('inf') if mse == 0 else 10 * math.log10(1.0 / mse)

def metric_row(noisy,clean,pred):
    x=clean.astype(np.float64)/255
    y=noisy.astype(np.float64)/255
    z=quantize(pred).astype(np.float64)/255
    def pair(a,b):
        return (_fast_psnr(a,b), _fast_ssim(a,b))
    pn,sn=pair(x,y); pp,sp=pair(x,z)
    dp=pp-pn if math.isfinite(pn) else (0.0 if not math.isfinite(pp) else -math.inf)
    score=.6*np.clip(dp/15,0,1)+.4*max(sp-sn,0)
    return dict(psnr=pp,ssim=sp,noisy_psnr=pn,noisy_ssim=sn,
                delta_psnr=dp,delta_ssim=sp-sn,score=float(score),
                mse=float(np.mean((x-z)**2)),mae=float(np.mean(np.abs(x-z))))
"""

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'CORE_IO =' in src:
            new_lines = [
                "CORE_IO = \"\"\"" + NEW_CORE_IO + "\"\"\"\n",
                "(SCRIPTS/'denoise_core.py').write_text(CORE_IO)\n",
                "import sys\n",
                "sys.path.insert(0,str(SCRIPTS))\n",
                "from denoise_core import read_image, quantize, write_image, metric_row\n",
                "print('IO and diagnostic metrics ready (with CUDA SSIM).')\n"
            ]
            cell['source'] = new_lines
            print('Updated CORE_IO cell!')
            break

with open('Mora_SP_Cup_Colab_Denoising.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
