"""
Generate gated_unet_hparam_sweep.ipynb — a hyperparameter sweep notebook
that reuses the exact GatedUNet architecture and data pipeline from nafnet_script.ipynb,
sweeps training hyperparameters, evaluates with the official evaluate.py, and
benchmarks CPU inference.
"""

import json
from pathlib import Path

def cell(source, cell_type='code'):
    if isinstance(source, list):
        src = source
    else:
        src = [line + '\n' for line in source.split('\n')]
        if src and src[-1] == '\n':
            src.pop()
    if cell_type == 'markdown':
        return {'cell_type': 'markdown', 'metadata': {}, 'source': src}
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}

def md(text):
    return cell(text, cell_type='markdown')

def code(text):
    return cell(text, cell_type='code')

cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(md("""# Gated U-Net Hyperparameter Sweep
Systematically sweeps training hyperparameters (learning rate, loss weighting) while
keeping the **exact same GatedUNet architecture** and data pipeline as `nafnet_script.ipynb`.

Each run is evaluated with:
1. The notebook's own fast diagnostic score (CUDA SSIM)
2. The official `evaluation/evaluate.py` (skimage SSIM, the real competition metric)
3. A clean CPU-only inference benchmark

Results are appended to `hparam_sweep_results.csv` — never overwritten."""))

# ── Cell 1: Paths & Config ──────────────────────────────────────────────────
cells.append(md("## Cell 1 — Paths, Packages & Device"))
cells.append(code("""import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'pillow', 'numpy', 'pandas', 'matplotlib', 'scikit-image', 'tqdm'])

from pathlib import Path
import os, json, random, time, hashlib, math, copy, platform
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
# Resolve repo root robustly: this notebook lives in scripts/, so go up one level.
# Works whether you run from the repo root OR from inside the scripts/ folder.
_nb_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
REPO_ROOT = next(
    (p for p in [_nb_dir] + list(_nb_dir.parents)
     if (p / 'competition_data').is_dir()),
    _nb_dir
)
DATASET_ROOT   = REPO_ROOT / 'competition_data'
EVAL_SCRIPT    = REPO_ROOT / 'evaluation' / 'evaluate.py'

TRAIN_NOISY_DIR = DATASET_ROOT / 'public' / 'noisy'
TRAIN_CLEAN_DIR = DATASET_ROOT / 'public' / 'ground_truth'
TEST_DIR        = DATASET_ROOT / 'submissions' / 'noisy'

SWEEP_ROOT = REPO_ROOT / 'sweep_outputs'
SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = SWEEP_ROOT / 'hparam_sweep_results.csv'

# ── Shared constants (match original notebook) ──────────────────────────────
BASE_CFG = dict(
    seed=42, width=24, blocks=2, patch=128, batch=4,
    steps_per_epoch=150, epochs=25,
    weight_decay=1e-4, ssim_weight=0.0,
    ema_decay=0.995, val_fraction=0.10,
    patience=8, tile=384, overlap=64, amp=True,
    cache_images=True, num_workers=0,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
else:
    raise RuntimeError('Select a GPU runtime before running the sweep — training without CUDA will take days.')
print('Outputs:', SWEEP_ROOT)"""))

# ── Cell 2: Image loading ───────────────────────────────────────────────────
cells.append(md("## Cell 2 — Image Indexer & Dataset Integrity Check"))
cells.append(code("""from PIL import Image
import re

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

def index_folder(folder, noisy):
    result = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in EXTS or not p.is_file(): continue
        m = re.fullmatch(r'(\\d+)_noise' if noisy else r'(\\d+)', p.stem)
        if not m: raise ValueError(f'Unexpected filename: {p.name}')
        i = int(m.group(1))
        if i in result: raise ValueError(f'Duplicate ID: {i}')
        result[i] = p
    return result

noisy_paths = index_folder(TRAIN_NOISY_DIR, True)
clean_paths = index_folder(TRAIN_CLEAN_DIR, False)
test_paths  = index_folder(TEST_DIR, True) if TEST_DIR.exists() else {}

for label, mapping, expected in [('noisy', noisy_paths, set(range(1,461))),
                                  ('clean', clean_paths, set(range(1,461)))]:
    if set(mapping) != expected:
        raise ValueError(f'{label}: missing={sorted(expected-set(mapping))}')
    for i, p in mapping.items():
        exp_name = f'{i:03d}_noise.png' if label != 'clean' else f'{i:03d}.png'
        if p.name != exp_name: raise ValueError(f'Expected {exp_name}, found {p.name}')

# Infer channels from first image
with Image.open(clean_paths[1]) as im:
    CHANNELS = 3 if im.mode == 'RGB' else 1
BASE_CFG['channels'] = CHANNELS
print(f'Dataset: {len(noisy_paths)} noisy / {len(clean_paths)} clean pairs | Channels: {CHANNELS}')"""))

# ── Cell 3: I/O helpers ─────────────────────────────────────────────────────
cells.append(md("## Cell 3 — I/O Helpers & Fast Diagnostic Metrics"))
cells.append(code("""def read_image(path):
    with Image.open(path) as im:
        a = np.asarray(im)
        if im.mode not in ('RGB','L') or a.dtype != np.uint8:
            raise ValueError(f'Only uint8 RGB/L supported: {path}')
    if a.ndim == 2: a = a[..., None]
    return np.array(a, copy=True)

def quantize(a):
    return np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)

def write_image(path, a):
    a = quantize(a)
    Image.fromarray(a[..., 0] if a.shape[-1] == 1 else a).save(path)

def _fast_psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * math.log10(1.0 / mse)

def _fast_ssim(a, b):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.ndim == 2: a = a[..., None]
    if b.ndim == 2: b = b[..., None]
    a_t = torch.from_numpy(a).float().permute(2,0,1).unsqueeze(0).to(dev)
    b_t = torch.from_numpy(b).float().permute(2,0,1).unsqueeze(0).to(dev)
    C = a_t.shape[1]
    w = torch.ones((C,1,7,7), device=dev) / 49.0
    ap = F.pad(a_t, (3,3,3,3), mode='reflect')
    bp = F.pad(b_t, (3,3,3,3), mode='reflect')
    mu_a = F.conv2d(ap, w, groups=C); mu_b = F.conv2d(bp, w, groups=C)
    sa = (F.conv2d(ap**2, w, groups=C) - mu_a**2) * (49.0/48.0)
    sb = (F.conv2d(bp**2, w, groups=C) - mu_b**2) * (49.0/48.0)
    sab = (F.conv2d(ap*bp, w, groups=C) - mu_a*mu_b) * (49.0/48.0)
    C1 = 0.0001; C2 = 0.0009
    ssim_map = ((2*mu_a*mu_b+C1)*(2*sab+C2)) / ((mu_a**2+mu_b**2+C1)*(sa+sb+C2))
    return ssim_map.mean().item()

def metric_row(noisy, clean, pred):
    x = clean.astype(np.float64)/255
    y = noisy.astype(np.float64)/255
    z = quantize(pred).astype(np.float64)/255
    pn,sn = _fast_psnr(x,y), _fast_ssim(x.astype(np.float32), y.astype(np.float32))
    pp,sp = _fast_psnr(x,z), _fast_ssim(x.astype(np.float32), z.astype(np.float32))
    dp = pp - pn if math.isfinite(pn) else -math.inf
    score = 0.6*np.clip(dp/15,0,1) + 0.4*max(sp-sn, 0)
    return dict(psnr=pp, ssim=sp, noisy_psnr=pn, noisy_ssim=sn,
                delta_psnr=dp, delta_ssim=sp-sn, score=float(score))

print('I/O helpers and fast CUDA SSIM ready.')"""))

# ── Cell 4: Dedup split ─────────────────────────────────────────────────────
cells.append(md("""## Cell 4 — Duplicate-Grouping Train / Val Split
Exact copy of the grouping logic from `nafnet_script.ipynb`. The split is shared across
all sweep runs (same seed, same val fraction) so results are directly comparable."""))
cells.append(code("""GROUPS_CSV = ''   # optional path to a CSV with columns [id, group]

random.seed(BASE_CFG['seed']); np.random.seed(BASE_CFG['seed']); torch.manual_seed(BASE_CFG['seed'])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(BASE_CFG['seed'])

parent = {i:i for i in clean_paths}
def root(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
def union(a,b): parent[root(b)] = root(a)

seen = {}
for i, p in tqdm(clean_paths.items(), desc='Grouping exact duplicates'):
    digest = hashlib.sha256(read_image(p).tobytes()).hexdigest()
    if digest in seen: union(i, seen[digest])
    else: seen[digest] = i

if GROUPS_CSV:
    groups = pd.read_csv(GROUPS_CSV)
    for _, g in groups.groupby('group'):
        ids = g['id'].astype(int).tolist()
        for i in ids[1:]: union(ids[0], i)

clusters = {}
for i in clean_paths: clusters.setdefault(root(i), []).append(i)
keys = list(clusters); random.Random(BASE_CFG['seed']).shuffle(keys)

val_ids = []
for key in keys:
    if len(val_ids) >= round(len(clean_paths) * BASE_CFG['val_fraction']): break
    val_ids.extend(clusters[key])
val_ids = sorted(val_ids)
train_ids = sorted(set(clean_paths) - set(val_ids))
print(f'Train: {len(train_ids)}  Val: {len(val_ids)}')
SPLIT = {'train': train_ids, 'val': val_ids}
(SWEEP_ROOT / 'split.json').write_text(json.dumps(SPLIT, indent=2))"""))

# ── Cell 5: Model Architecture ──────────────────────────────────────────────
cells.append(md("""## Cell 5 — Model Architecture
Verbatim copy of `ChannelNorm`, `GatedBlock`, and `Denoiser` from `nafnet_script.ipynb`.
**Do NOT modify** — we are sweeping training hyperparameters only, not architecture."""))
cells.append(code("""class ChannelNorm(nn.Module):
    def __init__(self,c):
        super().__init__(); self.w=nn.Parameter(torch.ones(1,c,1,1)); self.b=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        dtype=x.dtype; z=x.float(); mu=z.mean(1,keepdim=True)
        z=(z-mu)*torch.rsqrt((z-mu).square().mean(1,keepdim=True)+1e-6)
        return (z*self.w+self.b).to(dtype)

class GatedBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.n1=ChannelNorm(c); self.n2=ChannelNorm(c)
        self.expand=nn.Conv2d(c,2*c,1); self.depth=nn.Conv2d(2*c,2*c,3,padding=1,groups=2*c)
        self.attn=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,c,1))
        self.project=nn.Conv2d(c,c,1); self.f1=nn.Conv2d(c,2*c,1); self.f2=nn.Conv2d(c,c,1)
        self.beta=nn.Parameter(torch.zeros(1,c,1,1)); self.gamma=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        a,b=self.depth(self.expand(self.n1(x))).chunk(2,1); z=a*b
        x=x+self.beta*self.project(z*self.attn(z))
        a,b=self.f1(self.n2(x)).chunk(2,1)
        return x+self.gamma*self.f2(a*b)

class Denoiser(nn.Module):
    def __init__(self,channels=3,width=24,blocks=2):
        super().__init__(); self.channels=channels
        self.head=nn.Conv2d(channels,width,3,padding=1)
        self.enc=nn.ModuleList(); self.down=nn.ModuleList(); self.up=nn.ModuleList(); self.dec=nn.ModuleList()
        c=width
        for _ in range(3):
            self.enc.append(nn.Sequential(*[GatedBlock(c) for _ in range(blocks)]))
            self.down.append(nn.Conv2d(c,2*c,2,stride=2)); c*=2
        self.mid=nn.Sequential(*[GatedBlock(c) for _ in range(blocks+2)])
        for _ in range(3):
            self.up.append(nn.Sequential(nn.Conv2d(c,2*c,1),nn.PixelShuffle(2))); c//=2
            self.dec.append(nn.Sequential(*[GatedBlock(c) for _ in range(blocks)]))
        self.tail=nn.Conv2d(width,channels,3,padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)
    def forward(self,x):
        h,w=x.shape[-2:]; p=F.pad(x,(0,(-w)%8,0,(-h)%8),mode='replicate')
        z=self.head(p); skips=[]
        for enc,down in zip(self.enc,self.down): z=enc(z); skips.append(z); z=down(z)
        z=self.mid(z)
        for up,dec,s in zip(self.up,self.dec,reversed(skips)): z=dec(up(z)+s)
        return (p+self.tail(z))[...,:h,:w]

# Sanity check
with torch.no_grad():
    probe = torch.rand(1, CHANNELS, 65, 71).to(DEVICE)
    _tmp = Denoiser(CHANNELS, BASE_CFG['width'], BASE_CFG['blocks']).to(DEVICE)
    assert _tmp(probe).shape == probe.shape
    assert torch.allclose(_tmp(probe), probe)
    del _tmp, probe
print('PASS: Architecture shape and initial identity checks.')
print('Parameters:', sum(p.numel() for p in Denoiser(CHANNELS, BASE_CFG['width'], BASE_CFG['blocks']).parameters()))"""))

# ── Cell 6: Inference helpers ───────────────────────────────────────────────
cells.append(md("## Cell 6 — Tiled Inference & Wavelet Fallback"))
cells.append(code("""def tiled(model, x, tile=384, overlap=64):
    if tile <= overlap or overlap < 0: raise ValueError('Require 0 <= overlap < tile')
    h, w = x.shape[-2:]
    if h <= tile and w <= tile: return model(x).float()
    def starts(n):
        if n <= tile: return [0]
        return sorted(set(list(range(0, n-tile+1, tile-overlap)) + [n-tile]))
    out = torch.zeros_like(x, dtype=torch.float32)
    weights = torch.zeros_like(x[:,:1], dtype=torch.float32)
    for top in starts(h):
        for left in starts(w):
            patch = x[..., top:top+tile, left:left+tile]; ph, pw = patch.shape[-2:]
            wy = torch.hann_window(ph, periodic=False, device=x.device).clamp_min(.05)
            wx = torch.hann_window(pw, periodic=False, device=x.device).clamp_min(.05)
            weight = (wy[:,None]*wx[None,:])[None,None]
            out[..., top:top+ph, left:left+pw] += model(patch).float() * weight
            weights[..., top:top+ph, left:left+pw] += weight
    return out / weights

@torch.inference_mode()
def predict(model, array, device, tile=384, overlap=64):
    x = torch.from_numpy(np.ascontiguousarray(array.transpose(2,0,1))).float()[None].to(device) / 255
    model.eval()
    result = tiled(model, x, tile, overlap).clamp(0,1)[0].permute(1,2,0).cpu().numpy()
    if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite prediction')
    return result

def wavelet_prediction(array):
    from skimage.restoration import denoise_wavelet
    x = array.astype(np.float32) / 255
    if x.shape[-1] == 1:
        out = denoise_wavelet(x[...,0], method='BayesShrink', mode='soft',
                              rescale_sigma=True, channel_axis=None)[..., None]
    else:
        out = denoise_wavelet(x, method='BayesShrink', mode='soft',
                              rescale_sigma=True, channel_axis=-1, convert2ycbcr=True)
    return np.clip(out, 0, 1).astype(np.float32)

print('Inference helpers ready.')"""))

# ── Cell 7: Dataset loader ──────────────────────────────────────────────────
cells.append(md("## Cell 7 — Training Dataset & DataLoader"))
cells.append(code("""from torch.utils.data import Dataset, DataLoader

class PairedPatches(Dataset):
    def __init__(self, ids, patch_size, cache=True):
        self.ids = ids; self.patch_size = patch_size; self.cache = {} if cache else None
    def __len__(self): return BASE_CFG['steps_per_epoch'] * BASE_CFG['batch']
    def __getitem__(self, _):
        i = random.choice(self.ids)
        if self.cache is not None and i in self.cache:
            y, x = self.cache[i]
        else:
            y, x = read_image(noisy_paths[i]), read_image(clean_paths[i])
            if self.cache is not None: self.cache[i] = (y, x)
        p = self.patch_size; h, w = y.shape[:2]
        top = random.randrange(h-p+1); left = random.randrange(w-p+1)
        y = y[top:top+p, left:left+p]; x = x[top:top+p, left:left+p]
        k = random.randrange(4); y = np.rot90(y,k); x = np.rot90(x,k)
        if random.random() < .5: y = y[:,::-1]; x = x[:,::-1]
        return tuple(torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).float()/255 for a in (y,x))

print('Dataset class ready.')"""))

# ── Cell 8: Sweep definition ────────────────────────────────────────────────
cells.append(md("""## Cell 8 — Hyperparameter Grid
Defines the sweep configurations. Max 4 training runs (2 LRs × 2 loss weightings).
Alpha blending is swept at evaluation time only — no extra training runs needed."""))
cells.append(code("""# ── Sweep grid ──────────────────────────────────────────────────────────────
SWEEP_GRID = [
    dict(run_id='run_lr1e4_charb',  lr=1e-4, loss_weighting='charbonnier_only',    patch_size=128),
    dict(run_id='run_lr1e4_ssim',   lr=1e-4, loss_weighting='charbonnier_plus_ssim', patch_size=128),
    dict(run_id='run_lr3e4_charb',  lr=3e-4, loss_weighting='charbonnier_only',    patch_size=128),
    dict(run_id='run_lr3e4_ssim',   lr=3e-4, loss_weighting='charbonnier_plus_ssim', patch_size=128),
]

ALPHA_CANDIDATES = [0.0, 0.1, 0.2]  # swept at eval time only

print(f'Grid has {len(SWEEP_GRID)} training runs × {len(ALPHA_CANDIDATES)} alpha values at eval time.')
print('Estimated training time: 30-50 min per run on a T4 GPU (25 epochs × ~150 steps).')
for cfg in SWEEP_GRID:
    print(f"  {cfg['run_id']:30s} lr={cfg['lr']:.0e}  loss={cfg['loss_weighting']}")"""))

# ── Cell 9: Loss functions ──────────────────────────────────────────────────
cells.append(md("## Cell 9 — Loss Functions (Charbonnier + optional SSIM)"))
cells.append(code("""def charbonnier_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred.float() - target.float())**2 + eps**2).mean()

def structural_loss(a, b):
    a, b = a.float(), b.float()
    mu_a = F.avg_pool2d(a,7,1); mu_b = F.avg_pool2d(b,7,1)
    va = F.avg_pool2d(a*a,7,1) - mu_a*mu_a
    vb = F.avg_pool2d(b*b,7,1) - mu_b*mu_b
    cov = F.avg_pool2d(a*b,7,1) - mu_a*mu_b
    s = ((2*mu_a*mu_b+.01**2)*(2*cov+.03**2)) / ((mu_a.square()+mu_b.square()+.01**2)*(va+vb+.03**2))
    return 1 - s.mean()

def make_loss_fn(loss_weighting):
    if loss_weighting == 'charbonnier_only':
        return lambda pred, tgt: charbonnier_loss(pred, tgt)
    elif loss_weighting == 'charbonnier_plus_ssim':
        return lambda pred, tgt: charbonnier_loss(pred, tgt) + 0.15 * structural_loss(pred, tgt)
    else:
        raise ValueError(f'Unknown loss_weighting: {loss_weighting}')

print('Loss functions ready.')"""))

# ── Cell 10: Training runner ────────────────────────────────────────────────
cells.append(md("## Cell 10 — Training Runner & Diagnostic Evaluator"))
cells.append(code("""def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)

def diagnostic_evaluate(net, val_ids_list, tile, overlap, device):
    \"\"\"Fast diagnostic evaluation on val set using GPU SSIM.\"\"\"
    rows = []
    for i in tqdm(val_ids_list, desc='Val eval', leave=False):
        y = read_image(noisy_paths[i]); x = read_image(clean_paths[i])
        pred = predict(net, y, device, tile, overlap)
        if device.type == 'cuda': torch.cuda.synchronize()
        rows.append(metric_row(y, x, pred))
    frame = pd.DataFrame(rows)
    return frame, frame.mean().to_dict()

def run_training(run_cfg, seed=42):
    \"\"\"Train one configuration from scratch. Returns (history, best_ckpt_path).\"\"\"
    run_id = run_cfg['run_id']
    lr = run_cfg['lr']
    loss_weighting = run_cfg['loss_weighting']
    patch_size = run_cfg['patch_size']

    run_dir = SWEEP_ROOT / run_id
    ckpt_dir = run_dir / 'checkpoints'
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Seed per-run
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)

    cfg = dict(BASE_CFG)
    cfg.update(lr=lr, patch=patch_size)

    model = Denoiser(CHANNELS, cfg['width'], cfg['blocks']).to(DEVICE)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)

    loss_fn = make_loss_fn(loss_weighting)

    def lr_factor(epoch):
        if epoch < 3: return (epoch+1)/3
        progress = min(1, max(0, (epoch-3)/max(1, cfg['epochs']-3)))
        return .05 + .95*.5*(1+math.cos(math.pi*progress))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    AMP = cfg['amp'] and DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)

    dataset = PairedPatches(train_ids, patch_size, cache=True)
    loader = DataLoader(dataset, batch_size=cfg['batch'], num_workers=0,
                        pin_memory=DEVICE.type == 'cuda')

    history = []
    best_score = -float('inf')
    best_ckpt = ckpt_dir / f'{run_id}_best.pth'
    stale = 0

    print(f'\\n{"="*60}')
    print(f'Starting: {run_id}')
    print(f'  lr={lr:.0e}  loss={loss_weighting}  patch={patch_size}  epochs={cfg["epochs"]}')
    print(f'{"="*60}')

    for epoch in range(cfg['epochs']):
        model.train(); losses = []; started = time.perf_counter()
        if DEVICE.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        bar = tqdm(loader, desc=f'[{run_id}] Epoch {epoch+1}/{cfg["epochs"]}')
        nan_detected = False
        for noisy_b, clean_b in bar:
            noisy_b = noisy_b.to(DEVICE, non_blocking=True)
            clean_b = clean_b.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                    pred = model(noisy_b); loss = loss_fn(pred, clean_b)
                if not torch.isfinite(loss):
                    print(f'WARNING: NaN/Inf loss at epoch {epoch+1}, step {len(losses)}. Skipping run.')
                    nan_detected = True; break
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                stepped = scaler.get_scale() >= old_scale
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                print(f'WARNING: GPU OOM on run {run_id}. Skipping.')
                nan_detected = True; break
            if stepped:
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), model.parameters()): ep.lerp_(p, 1-cfg['ema_decay'])
            losses.append(float(loss.detach())); bar.set_postfix(loss=np.mean(losses[-20:]))

        if nan_detected:
            print(f'Run {run_id} aborted due to NaN/OOM.')
            return history, None

        _, raw_metrics = diagnostic_evaluate(model, val_ids, cfg['tile'], cfg['overlap'], DEVICE)
        _, ema_metrics = diagnostic_evaluate(ema, val_ids, cfg['tile'], cfg['overlap'], DEVICE)
        candidates = [('raw', model, raw_metrics), ('ema', ema, ema_metrics)]
        kind, winner, metrics = max(candidates, key=lambda item: item[2]['score'])

        improved = metrics['score'] > best_score
        if improved:
            best_score = metrics['score']
            state = {'model': {k:v.detach().cpu() for k,v in winner.state_dict().items()},
                     'config': dict(cfg), 'epoch': epoch, 'metrics': metrics, 'split': SPLIT}
            atomic_save(state, best_ckpt)
            stale = 0
        else:
            stale += 1

        elapsed = time.perf_counter() - started
        row = dict(epoch=epoch+1, loss=float(np.mean(losses)),
                   lr=optimizer.param_groups[0]['lr'],
                   raw_score=raw_metrics['score'], ema_score=ema_metrics['score'],
                   selected_score=metrics['score'], selected=kind,
                   psnr=metrics['psnr'], ssim=metrics['ssim'], seconds=elapsed)
        history.append(row)
        scheduler.step()

        still_improving = epoch >= 3 and history[-1]['selected_score'] - history[-2]['selected_score'] > 0.001
        suffix = ' ↑' if improved else (' ≈' if still_improving else '')
        print(f"  Epoch {epoch+1:2d} | loss={row['loss']:.4f} | score={metrics['score']:.4f}{suffix} | "
              f"psnr={metrics['psnr']:.2f} | {elapsed:.0f}s")

        pd.DataFrame(history).to_csv(run_dir / 'training_history.csv', index=False)

        if stale >= cfg['patience']:
            print(f'Early stopping at epoch {epoch+1}. Best score: {best_score:.4f}')
            break

        # After first epoch, print time estimate for the whole sweep
        if epoch == 0:
            remaining_epochs = cfg['epochs'] - 1
            remaining_runs = len(SWEEP_GRID) - list(r['run_id'] for r in SWEEP_GRID).index(run_id) - 1
            total_remaining_s = elapsed * (remaining_epochs + remaining_runs * cfg['epochs'])
            print(f'\\n  ⏱  Estimated remaining sweep time: {total_remaining_s/3600:.1f} h '
                  f'({remaining_runs} more runs + {remaining_epochs} more epochs this run)\\n')

    still_improving_at_end = (len(history) == cfg['epochs'] and
                               history[-1]['selected_score'] - history[-5]['selected_score'] > 0.01)
    if still_improving_at_end:
        print(f'  ⚠️  Run {run_id} was still clearly improving at epoch {cfg["epochs"]} — consider more epochs.')

    return history, best_ckpt

print('Training runner ready.')"""))

# ── Cell 11: Official evaluator ─────────────────────────────────────────────
cells.append(md("""## Cell 11 — Official Evaluator (via evaluation/evaluate.py)
Writes predicted images to disk then calls the official script via subprocess.
This gives the **true competition score** using skimage SSIM with win_size=7,
gaussian_weights=False — matching the official evaluation exactly."""))
cells.append(code("""def official_eval_on_full_set(model_or_ckpt_path, run_id, alpha=0.0,
                               tile=384, overlap=64):
    \"\"\"
    Run the official evaluate.py on all 460 public images.
    Returns dict with official mean composite score, psnr, ssim.
    \"\"\"
    pred_dir = SWEEP_ROOT / run_id / f'preds_alpha{alpha:.2f}'
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Load model if path given
    if isinstance(model_or_ckpt_path, Path):
        state = torch.load(model_or_ckpt_path, map_location='cpu', weights_only=False)
        c = state['config']
        net = Denoiser(c['channels'], c['width'], c['blocks'])
        net.load_state_dict(state['model'])
        net = net.to(DEVICE).eval()
    else:
        net = model_or_ckpt_path

    all_ids = sorted(noisy_paths.keys())
    print(f'  Running inference on {len(all_ids)} images (alpha={alpha})...')
    for i in tqdm(all_ids, desc='Official inference', leave=False):
        y = read_image(noisy_paths[i])
        pred = predict(net, y, DEVICE, tile, overlap)
        if alpha > 0:
            wav = wavelet_prediction(y)
            pred = (1 - alpha) * pred + alpha * wav
        pred = np.clip(pred, 0, 1)
        write_image(pred_dir / f'{i:03d}.png', pred)

    # Call official evaluate.py
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         '--noisy_dir', str(TRAIN_NOISY_DIR),
         '--pred_dir',  str(pred_dir),
         '--gt_dir',    str(TRAIN_CLEAN_DIR)],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    print(result.stdout)

    # Parse output
    def extract(key):
        import re
        m = re.search(key + r'\\s*([\\d.]+)', output)
        return float(m.group(1)) if m else float('nan')

    return dict(
        composite=extract('Composite Score:'),
        psnr=extract('Mean PSNR:'),
        ssim=extract('Mean SSIM:'),
    )

print('Official evaluator wrapper ready.')
print('Evaluator script:', EVAL_SCRIPT, '(exists:', EVAL_SCRIPT.exists(), ')')"""))

# ── Cell 12: CPU benchmark ──────────────────────────────────────────────────
cells.append(md("## Cell 12 — CPU-Only Inference Benchmark"))
cells.append(code("""def cpu_benchmark(ckpt_path, alpha=0.0, tile=384, overlap=64, n_warmup=3, n_measure=10):
    \"\"\"
    Clean CPU-only benchmark. Model loaded OUTSIDE timing loop.
    3 warmup passes discarded, then 10 measured passes.
    alpha-blend cost included if alpha > 0 (realistic inference cost).
    \"\"\"
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    c = state['config']
    net = Denoiser(c['channels'], c['width'], c['blocks'])
    net.load_state_dict(state['model'])
    net = net.cpu().eval()

    # Pick 10 random images for measurement
    rng = random.Random(42)
    measure_ids = rng.sample(sorted(noisy_paths.keys()), n_measure + n_warmup)

    times = []
    for idx, i in enumerate(measure_ids):
        y = read_image(noisy_paths[i])
        t0 = time.perf_counter()
        pred = predict(net, y, torch.device('cpu'), tile, overlap)
        if alpha > 0:
            wav = wavelet_prediction(y)
            pred = (1 - alpha) * pred + alpha * wav
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        if idx >= n_warmup:
            times.append(elapsed)
        print(f'  {"warmup" if idx < n_warmup else "MEASURE"} {idx+1}/{n_warmup+n_measure}: {elapsed:.0f} ms')

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    print(f'\\nCPU benchmark: {mean_ms:.1f} ± {std_ms:.1f} ms/image (alpha={alpha})')
    return mean_ms, std_ms

print('CPU benchmark ready.')"""))

# ── Cell 13: THE SWEEP ──────────────────────────────────────────────────────
cells.append(md("""## Cell 13 — Run the Full Sweep
This is the main cell. Run it after all setup cells above. Each run trains a fresh model
from scratch, picks the best checkpoint (raw vs EMA), sweeps alpha at eval time,
runs the official evaluator, benchmarks CPU inference, then appends results to CSV."""))
cells.append(code("""import datetime

all_results = []

for run_cfg in SWEEP_GRID:
    run_id = run_cfg['run_id']
    print(f'\\n{"#"*65}')
    print(f'# Sweep run: {run_id}')
    print(f'{"#"*65}')

    # ── Train ──────────────────────────────────────────────────────────────
    try:
        history, best_ckpt = run_training(run_cfg)
    except Exception as exc:
        print(f'ERROR in training {run_id}: {exc}')
        continue

    if best_ckpt is None or not best_ckpt.exists():
        print(f'Skipping {run_id} — no valid checkpoint saved.')
        continue

    epochs_trained = len(history)
    last5_gain = history[-1]['selected_score'] - history[max(0,len(history)-6)]['selected_score']
    still_improving = last5_gain > 0.01

    # ── Diagnostic score (already computed during training) ────────────────
    best_diag_score = max(r['selected_score'] for r in history)
    diag_psnr = max(r['psnr'] for r in history)
    diag_ssim = max(r['ssim'] for r in history)

    # ── Load best checkpoint model ─────────────────────────────────────────
    state = torch.load(best_ckpt, map_location='cpu', weights_only=False)
    c = state['config']
    net = Denoiser(c['channels'], c['width'], c['blocks'])
    net.load_state_dict(state['model'])
    net = net.to(DEVICE).eval()

    # ── Alpha sweep at eval time ───────────────────────────────────────────
    print('\\nSweeping alpha...')
    best_alpha = 0.0; best_official = -float('inf'); best_psnr = 0.0; best_ssim_val = 0.0
    alpha_results = []
    for alpha in ALPHA_CANDIDATES:
        try:
            official = official_eval_on_full_set(best_ckpt, run_id, alpha=alpha,
                                                  tile=BASE_CFG['tile'], overlap=BASE_CFG['overlap'])
            print(f'  alpha={alpha:.1f}: official_composite={official[\"composite\"]:.4f} '
                  f'psnr={official[\"psnr\"]:.2f} ssim={official[\"ssim\"]:.4f}')
            alpha_results.append((alpha, official))
            if official['composite'] > best_official:
                best_official = official['composite']
                best_alpha = alpha
                best_psnr = official['psnr']
                best_ssim_val = official['ssim']
        except Exception as exc:
            print(f'  WARNING: official eval failed for alpha={alpha}: {exc}')

    print(f'  Best alpha: {best_alpha} → composite={best_official:.4f}')

    # ── Diagnostic vs official gap ─────────────────────────────────────────
    gap = best_diag_score - best_official
    print(f'\\n  Diagnostic score: {best_diag_score:.4f}')
    print(f'  Official score:   {best_official:.4f}')
    print(f'  Gap (diag-official): {gap:+.4f}')

    # ── CPU benchmark ──────────────────────────────────────────────────────
    print('\\nRunning CPU benchmark...')
    try:
        cpu_mean, cpu_std = cpu_benchmark(best_ckpt, alpha=best_alpha,
                                           tile=BASE_CFG['tile'], overlap=BASE_CFG['overlap'])
    except Exception as exc:
        print(f'  WARNING: CPU benchmark failed: {exc}')
        cpu_mean, cpu_std = float('nan'), float('nan')

    # ── Append to results ──────────────────────────────────────────────────
    row = dict(
        run_id=run_id,
        learning_rate=run_cfg['lr'],
        loss_weighting=run_cfg['loss_weighting'],
        patch_size=run_cfg['patch_size'],
        epochs_trained=epochs_trained,
        still_improving_at_end=still_improving,
        best_alpha=best_alpha,
        notebook_diagnostic_score=round(best_diag_score, 6),
        official_evaluate_score=round(best_official, 6),
        diag_vs_official_gap=round(gap, 6),
        mean_psnr=round(best_psnr, 4),
        mean_ssim=round(best_ssim_val, 6),
        cpu_ms_per_image=round(cpu_mean, 2),
        cpu_ms_std=round(cpu_std, 2),
        timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
    )
    all_results.append(row)

    # Append-only CSV
    df_row = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        df_row.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
    else:
        df_row.to_csv(RESULTS_CSV, index=False)

    print(f'\\nResult saved: {row}')

print('\\n' + '='*65)
print('ALL RUNS COMPLETE')
print('='*65)"""))

# ── Cell 14: Summary ────────────────────────────────────────────────────────
cells.append(md("## Cell 14 — Final Summary Table & Conclusions"))
cells.append(code("""df = pd.read_csv(RESULTS_CSV).sort_values('official_evaluate_score', ascending=False)

print('\\n=== HYPERPARAMETER SWEEP RESULTS (sorted by official score) ===\\n')
display_cols = ['run_id', 'learning_rate', 'loss_weighting', 'epochs_trained',
                'best_alpha', 'notebook_diagnostic_score', 'official_evaluate_score',
                'diag_vs_official_gap', 'mean_psnr', 'mean_ssim',
                'cpu_ms_per_image', 'cpu_ms_std', 'still_improving_at_end']
print(df[display_cols].to_string(index=False))

winner = df.iloc[0]
baseline_score = 0.5233  # best alpha=0.1 score from original nafnet_script run

print(f'\\n=== CONCLUSIONS ===')
print(f'Winner:          {winner[\"run_id\"]}')
print(f'Official score:  {winner[\"official_evaluate_score\"]:.4f}')
print(f'Baseline score:  {baseline_score:.4f}  (from nafnet_script, alpha=0.1)')
delta = winner[\"official_evaluate_score\"] - baseline_score
if abs(delta) < 0.005:
    print(f'Verdict: ±{abs(delta):.4f} — original hyperparameters were ALREADY near-optimal.')
elif delta > 0:
    print(f'Verdict: +{delta:.4f} improvement — sweep found a MEANINGFULLY BETTER config!')
else:
    print(f'Verdict: {delta:.4f} — original hyperparameters were BETTER than all sweep configs.')

print(f'\\nDiagnostic vs Official gap across runs:')
for _, r in df.iterrows():
    trusted = 'TRUSTED (gap < 0.01)' if abs(r['diag_vs_official_gap']) < 0.01 else 'SUSPECT'
    print(f\"  {r['run_id']:30s}  gap={r['diag_vs_official_gap']:+.4f}  → {trusted}\")"""))

# ── Build notebook ───────────────────────────────────────────────────────────
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.0'},
        'accelerator': 'GPU',
    },
    'cells': cells,
}

out_path = Path('scripts/gated_unet_hparam_sweep.ipynb')
out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Written: {out_path}  ({out_path.stat().st_size//1024} KB)')
