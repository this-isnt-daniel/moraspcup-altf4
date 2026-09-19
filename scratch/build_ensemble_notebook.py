"""
Generate gated_unet_ensemble.ipynb — an ensemble notebook
that combines the top 3 models from the sweep with a new, larger capacity model.
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
cells.append(md("""# Gated U-Net Ensemble & Scaling
This notebook builds an ensemble of the best checkpoints from the hyperparameter sweep
and trains a new, larger-capacity model (`width=32`, `blocks=4`) to add to the ensemble."""))

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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tempfile, shutil

# ── Paths ──────────────────────────────────────────────────────────────────
_nb_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
REPO_ROOT = next(
    (p for p in [_nb_dir] + list(_nb_dir.parents)
     if (p / 'competition_data').is_dir()),
    _nb_dir
)
DATASET_ROOT   = REPO_ROOT / 'competition_data'
EVAL_SCRIPT    = REPO_ROOT / 'evaluation' / 'evaluate.py'
DENOISE_SCRIPT = REPO_ROOT / 'scripts' / 'denoise.py'

TRAIN_NOISY_DIR = DATASET_ROOT / 'public' / 'noisy'
TRAIN_CLEAN_DIR = DATASET_ROOT / 'public' / 'ground_truth'
TEST_DIR        = DATASET_ROOT / 'submissions' / 'noisy'

SWEEP_ROOT = REPO_ROOT / 'sweep_outputs'
SPLIT_JSON = SWEEP_ROOT / 'split.json'
ENSEMBLE_ROOT = REPO_ROOT / 'ensemble_outputs'
ENSEMBLE_ROOT.mkdir(parents=True, exist_ok=True)

# ── Shared constants ───────────────────────────────────────────────────────
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
else:
    raise RuntimeError('Select a GPU runtime before running the ensemble — training without CUDA will take days.')
print('Ensemble outputs:', ENSEMBLE_ROOT)"""))

# ── Cell 2: Model Architecture ──────────────────────────────────────────────
cells.append(md("""## Cell 2 — Model Architecture
Verbatim copy of `ChannelNorm`, `GatedBlock`, and `Denoiser`. Existing checkpoints must load into this exact class."""))
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
        return (p+self.tail(z))[...,:h,:w]"""))

# ── Cell 3: Data Loading & Split ────────────────────────────────────────────
cells.append(md("""## Cell 3 — Data Loading & Split (Reusing Sweep Split)
Crucial: We load `split.json` from the sweep to ensure we use the **exact same 414/46 train/val split**. This prevents data leakage and ensures ensemble val scores are directly comparable to the sweep."""))
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

with Image.open(clean_paths[1]) as im:
    CHANNELS = 3 if im.mode == 'RGB' else 1
BASE_CFG['channels'] = CHANNELS

# Load exact split from sweep
if not SPLIT_JSON.exists():
    raise FileNotFoundError(f"Sweep split.json not found at {SPLIT_JSON}. You must run the sweep first!")
    
with open(SPLIT_JSON, 'r') as f:
    SPLIT = json.load(f)
    
train_ids = SPLIT['train']
val_ids = SPLIT['val']
print(f'Train: {len(train_ids)} images | Val: {len(val_ids)} images (loaded from sweep split.json)')"""))

# ── Cell 4: Inference Utilities ─────────────────────────────────────────────
cells.append(md("## Cell 4 — Inference Utilities"))
cells.append(code("""def read_image(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 2: a = a[..., None]
    return np.array(a, copy=True)

def quantize(a):
    return np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)

def write_image(path, a):
    a = quantize(a)
    Image.fromarray(a[..., 0] if a.shape[-1] == 1 else a).save(path)

def tiled(model, x, tile=384, overlap=64):
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
    return result

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

print('Inference utilities ready.')"""))

# ── Cell 5: Load Sweep Checkpoints ──────────────────────────────────────────
cells.append(md("""## Cell 5 — Load Sweep Checkpoints (Top 3)
We'll load the top 3 models from the sweep and evaluate them individually on the val set."""))
cells.append(code("""ENSEMBLE_MEMBERS = [
    ("run_lr3e4_ssim_p192", SWEEP_ROOT),
    ("run_lr3e4_ssim",       SWEEP_ROOT),
    ("run_lr3e4_charb",      SWEEP_ROOT),
]

models = {}
for run_id, root in ENSEMBLE_MEMBERS:
    ckpt = root / run_id / 'checkpoints' / f'{run_id}_best.pth'
    if not ckpt.exists():
        print(f'WARNING: Checkpoint missing -> {ckpt}')
        continue
    
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    c = state['config']
    net = Denoiser(c['channels'], c['width'], c['blocks'])
    net.load_state_dict(state['model'])
    models[run_id] = net.to(DEVICE).eval()
    print(f'Loaded {run_id}: width={c["width"]} blocks={c["blocks"]}')

# Individual evaluation
individual_scores = {}
for run_id, net in models.items():
    rows = []
    for i in tqdm(val_ids, desc=f'Eval {run_id}', leave=False):
        y = read_image(noisy_paths[i]); x = read_image(clean_paths[i])
        pred = predict(net, y, DEVICE, BASE_CFG['tile'], BASE_CFG['overlap'])
        rows.append(metric_row(y, x, pred))
    df = pd.DataFrame(rows)
    m = df.mean()
    individual_scores[run_id] = m
    print(f\"{run_id:25s} | Score: {m['score']:.4f} | PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f}\")
"""))

# ── Cell 6: Incremental Ensemble Eval ───────────────────────────────────────
cells.append(md("""## Cell 6 — Incremental Ensemble Evaluation
We'll evaluate the ensemble by adding one model at a time, simply averaging their predictions."""))
cells.append(code("""ensemble_keys = list(models.keys())
print("Incremental Ensemble Performance (Val-46):\\n")

best_ensemble_score = 0
best_ensemble_preds = {}

# Evaluate incrementally: 1 model, then 2, then 3
for n in range(1, len(ensemble_keys) + 1):
    current_keys = ensemble_keys[:n]
    rows = []
    
    for i in tqdm(val_ids, desc=f'Ensemble {n} models', leave=False):
        y = read_image(noisy_paths[i])
        x = read_image(clean_paths[i])
        
        preds = []
        for k in current_keys:
            preds.append(predict(models[k], y, DEVICE, BASE_CFG['tile'], BASE_CFG['overlap']))
            
        # Simple average
        avg_pred = np.mean(preds, axis=0)
        if n == len(ensemble_keys):
            best_ensemble_preds[i] = avg_pred
            
        rows.append(metric_row(y, x, avg_pred))
        
    m = pd.DataFrame(rows).mean()
    print(f\"Top {n} Models | Score: {m['score']:.4f} | PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f}\")
    if m['score'] > best_ensemble_score:
        best_ensemble_score = m['score']
"""))

# ── Cell 7: Official Evaluate.py ────────────────────────────────────────────
cells.append(md("""## Cell 7 — Official Evaluator (Top-3 Ensemble)
We'll dump the predictions for the top-3 ensemble to disk and run the official `evaluate.py` script on the **val-46 images only** to avoid data leakage."""))
cells.append(code("""import subprocess

def official_eval(preds_dict, run_name):
    pred_dir = ENSEMBLE_ROOT / run_name / 'preds'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    for i, pred in preds_dict.items():
        write_image(pred_dir / f'{i:03d}.png', pred)
        
    # Copy ground truth and noisy for ONLY these IDs to temp dirs to avoid scoring all 460
    tmp_gt = Path(tempfile.mkdtemp())
    tmp_noisy = Path(tempfile.mkdtemp())
    for i in preds_dict.keys():
        shutil.copy2(clean_paths[i], tmp_gt / f'{i:03d}.png')
        shutil.copy2(noisy_paths[i], tmp_noisy / f'{i:03d}_noise.png')
        
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         '--noisy_dir', str(tmp_noisy),
         '--pred_dir',  str(pred_dir),
         '--gt_dir',    str(tmp_gt)],
        capture_output=True, text=True
    )
    
    shutil.rmtree(tmp_gt); shutil.rmtree(tmp_noisy)
    
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:\\n", result.stderr)

    def extract(key):
        import re
        m = re.search(key + r'\\s*([\\d.]+)', result.stdout)
        return float(m.group(1)) if m else float('nan')

    return dict(composite=extract('Composite Score:'), psnr=extract('Mean PSNR:'), ssim=extract('Mean SSIM:'))

official_top3 = official_eval(best_ensemble_preds, 'top3_ensemble')
"""))

# ── Cell 8: Train Larger Model ──────────────────────────────────────────────
cells.append(md("""## Cell 8 — Train Larger Model (Method 4)
We will train a single wider & deeper model (`width=32`, `blocks=4`) which has ~2.5x the capacity of the base models."""))
cells.append(code("""from torch.utils.data import Dataset, DataLoader

# Dataset class
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

def charbonnier_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred.float() - target.float())**2 + eps**2).mean()

def structural_loss(a, b):
    a, b = a.float(), b.float()
    mu_a = F.avg_pool2d(a,7,1); mu_b = F.avg_pool2d(b,7,1)
    va = F.avg_pool2d(a*a,7,1) - mu_a*mu_a
    vb = F.avg_pool2d(b*b,7,1) - mu_b*mu_b
    cov = F.avg_pool2d(a*b,7,1) - mu_a*mu_b
    s = ((2*mu_a*mu_b+.01**2)*(2*cov+.03**2)) / ((mu_a.square()+mu_b.square()+.01**2)*(va+vb+.03**2))
    return 1 - s.mean()

def loss_fn(pred, tgt): return charbonnier_loss(pred, tgt) + 0.15 * structural_loss(pred, tgt)

def diagnostic_evaluate(net, val_ids_list, tile, overlap, device):
    rows = []
    for i in val_ids_list:
        y = read_image(noisy_paths[i]); x = read_image(clean_paths[i])
        pred = predict(net, y, device, tile, overlap)
        if device.type == 'cuda': torch.cuda.synchronize()
        rows.append(metric_row(y, x, pred))
    frame = pd.DataFrame(rows)
    return frame, frame.mean().to_dict()

# Train larger model
run_id = "larger_model_w32_b4"
lr = 3e-4
patch_size = 192
width = 32
blocks = 4
epochs = 30

run_dir = ENSEMBLE_ROOT / run_id
ckpt_dir = run_dir / 'checkpoints'
run_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)

model_large = Denoiser(CHANNELS, width, blocks).to(DEVICE)
ema = copy.deepcopy(model_large).eval()
for p in ema.parameters(): p.requires_grad_(False)

print(f"Larger Model Params: {sum(p.numel() for p in model_large.parameters())}")

optimizer = torch.optim.AdamW(model_large.parameters(), lr=lr, weight_decay=BASE_CFG['weight_decay'])
def lr_factor(epoch):
    if epoch < 3: return (epoch+1)/3
    progress = min(1, max(0, (epoch-3)/max(1, epochs-3)))
    return .05 + .95*.5*(1+math.cos(math.pi*progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
AMP = BASE_CFG['amp'] and DEVICE.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=AMP)
dataset = PairedPatches(train_ids, patch_size, cache=True)
loader = DataLoader(dataset, batch_size=BASE_CFG['batch'], num_workers=0, pin_memory=DEVICE.type == 'cuda')

history = []
best_score = -float('inf')
best_ckpt_large = ckpt_dir / f'{run_id}_best.pth'

print(f"\\nTraining Larger Model ({run_id})")
for epoch in range(epochs):
    model_large.train(); losses = []; started = time.perf_counter()
    bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
    for noisy_b, clean_b in bar:
        noisy_b = noisy_b.to(DEVICE, non_blocking=True); clean_b = clean_b.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=AMP):
            pred = model_large(noisy_b); loss = loss_fn(pred, clean_b)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_large.parameters(), 1.0)
        old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale() >= old_scale:
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model_large.parameters()): ep.lerp_(p, 1-BASE_CFG['ema_decay'])
        losses.append(float(loss.detach()))

    _, raw_metrics = diagnostic_evaluate(model_large, val_ids, BASE_CFG['tile'], BASE_CFG['overlap'], DEVICE)
    _, ema_metrics = diagnostic_evaluate(ema, val_ids, BASE_CFG['tile'], BASE_CFG['overlap'], DEVICE)
    kind, winner, metrics = max([('raw', model_large, raw_metrics), ('ema', ema, ema_metrics)], key=lambda x:x[2]['score'])
    
    improved = metrics['score'] > best_score
    if improved:
        best_score = metrics['score']
        state = {'model': {k:v.detach().cpu() for k,v in winner.state_dict().items()},
                 'config': {'width': width, 'blocks': blocks, 'channels': CHANNELS}, 'epoch': epoch}
        torch.save(state, best_ckpt_large)
    
    elapsed = time.perf_counter() - started
    print(f"Epoch {epoch+1:2d} | loss={np.mean(losses):.4f} | score={metrics['score']:.4f}{' ↑' if improved else ''} | psnr={metrics['psnr']:.2f} | {elapsed:.0f}s")
    scheduler.step()
"""))

# ── Cell 9: Evaluate Larger Model ───────────────────────────────────────────
cells.append(md("""## Cell 9 — Evaluate Larger Model Individually
We load the best checkpoint of the new model and evaluate it on the val-46 set."""))
cells.append(code("""state = torch.load(best_ckpt_large, map_location='cpu', weights_only=False)
c = state['config']
best_model_large = Denoiser(c['channels'], c['width'], c['blocks'])
best_model_large.load_state_dict(state['model'])
best_model_large = best_model_large.to(DEVICE).eval()

large_preds = {}
rows = []
for i in tqdm(val_ids, desc='Eval Larger Model', leave=False):
    y = read_image(noisy_paths[i]); x = read_image(clean_paths[i])
    pred = predict(best_model_large, y, DEVICE, BASE_CFG['tile'], BASE_CFG['overlap'])
    large_preds[i] = pred
    rows.append(metric_row(y, x, pred))

m = pd.DataFrame(rows).mean()
print(f\"Larger Model | Score: {m['score']:.4f} | PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f}\")

official_large = official_eval(large_preds, 'larger_model')
"""))

# ── Cell 10: 4-Model Ensemble ───────────────────────────────────────────────
cells.append(md("""## Cell 10 — Full 4-Model Ensemble
We now average the Top-3 sweep models + the new Larger model."""))
cells.append(code("""full_ensemble_preds = {}
rows = []
for i in tqdm(val_ids, desc='Eval 4-Model Ensemble', leave=False):
    y = read_image(noisy_paths[i]); x = read_image(clean_paths[i])
    pred_top3 = best_ensemble_preds[i]
    pred_large = large_preds[i]
    
    # Simple average of 4 models: 3 from sweep, 1 large
    avg_pred = (pred_top3 * 3 + pred_large) / 4
    full_ensemble_preds[i] = avg_pred
    
    rows.append(metric_row(y, x, avg_pred))
    
m = pd.DataFrame(rows).mean()
print(f\"4-Model Ensemble | Score: {m['score']:.4f} | PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f}\")

official_4model = official_eval(full_ensemble_preds, '4model_ensemble')
"""))

# ── Cell 11: CPU Benchmark ──────────────────────────────────────────────────
cells.append(md("""## Cell 11 — CPU Benchmark (Full Ensemble)
We benchmark all 4 models running sequentially on the CPU to ensure we are still fast enough."""))
cells.append(code("""# Prepare all 4 models on CPU
cpu_models = []
for k in ensemble_keys:
    cpu_models.append(models[k].cpu().eval())
cpu_models.append(best_model_large.cpu().eval())

measure_ids = random.Random(42).sample(val_ids, 8)
n_warmup = 2

times = []
for idx, i in enumerate(measure_ids):
    y = read_image(noisy_paths[i])
    t0 = time.perf_counter()
    
    for net in cpu_models:
        _ = predict(net, y, torch.device('cpu'), BASE_CFG['tile'], BASE_CFG['overlap'])
        
    elapsed = (time.perf_counter() - t0) * 1000
    if idx >= n_warmup:
        times.append(elapsed)
    print(f'  {"warmup" if idx < n_warmup else "MEASURE"} {idx+1}/{len(measure_ids)}: {elapsed:.0f} ms')

cpu_mean = np.mean(times)
print(f'\\nCPU benchmark (4 models): {cpu_mean:.1f} ± {np.std(times):.1f} ms/image')
"""))

# ── Cell 12: Write scripts/denoise.py ───────────────────────────────────────
cells.append(md("## Cell 12 — Write scripts/denoise.py for Ensemble Submission"))
cells.append(code("""ensemble_configs = []
for run_id, root in ENSEMBLE_MEMBERS:
    ckpt = root / run_id / 'checkpoints' / f'{run_id}_best.pth'
    ensemble_configs.append(str(ckpt.relative_to(REPO_ROOT)).replace('\\\\', '/'))
ensemble_configs.append(str(best_ckpt_large.relative_to(REPO_ROOT)).replace('\\\\', '/'))

ckpt_paths_str = str(ensemble_configs)

DENOISE_SRC = f'''#!/usr/bin/env python3
\"\"\"
denoise.py  — Mora SP Cup 2026 submission inference script.
Architecture: Ensemble of 4 NAFNet-style Gated Residual U-Nets (3 base, 1 large).
\"\"\"

import argparse, math, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn

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

def tiled(model, x, tile=384, overlap=64):
    h, w = x.shape[-2:]
    if h <= tile and w <= tile: return model(x).float()
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
    x = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()[None].to(device) / 255
    model.eval()
    return tiled(model, x, tile, overlap).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise_dir', required=True, type=Path)
    ap.add_argument('--denoised_dir', required=True, type=Path)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = ap.parse_args()

    device = torch.device('cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    print(f'Device: {{device}}', flush=True)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    ckpt_paths = {ckpt_paths_str}
    
    models = []
    for rel_path in ckpt_paths:
        ckpt_path = repo_root / rel_path
        if not ckpt_path.exists(): raise FileNotFoundError(f'Missing: {{ckpt_path}}')
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        c = state['config']
        model = Denoiser(c['channels'], c.get('width', 24), c.get('blocks', 2))
        model.load_state_dict(state['model'])
        models.append(model.to(device).eval())
    print(f'Loaded {{len(models)}} ensemble models.', flush=True)

    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(args.noise_dir.glob('*_noise.png'))

    t0 = time.perf_counter()
    for p in inputs:
        out_name = f'{{p.stem[:-6]}}.png'
        arr = np.asarray(Image.open(p))
        if arr.ndim == 2: arr = arr[..., None]
        
        preds = []
        for model in models:
            preds.append(predict(model, arr, device, 384, 64))
            
        avg_pred = np.mean(preds, axis=0)
        out = np.rint(np.clip(avg_pred, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(out[..., 0] if out.shape[-1] == 1 else out).save(args.denoised_dir / out_name)
        
        print(f'  {{p.name}} -> {{out_name}}', flush=True)

if __name__ == '__main__':
    main()
'''
DENOISE_SCRIPT.write_text(DENOISE_SRC, encoding='utf-8')
print("Written scripts/denoise.py")
"""))

# ── Cell 13: Summary Table ──────────────────────────────────────────────────
cells.append(md("## Cell 13 — Summary"))
cells.append(code("""print("=== ENSEMBLE SUMMARY (Val-46) ===\\n")
print(f"{'Configuration':<25} | {'Official Score':<15} | {'PSNR':<10} | {'SSIM':<10}")
print("-" * 70)
print(f"{'Top-3 Sweep Ensemble':<25} | {official_top3['composite']:<15.4f} | {official_top3['psnr']:<10.2f} | {official_top3['ssim']:<10.4f}")
print(f"{'Larger Model Alone':<25} | {official_large['composite']:<15.4f} | {official_large['psnr']:<10.2f} | {official_large['ssim']:<10.4f}")
print(f"{'4-Model Full Ensemble':<25} | {official_4model['composite']:<15.4f} | {official_4model['psnr']:<10.2f} | {official_4model['ssim']:<10.4f}")
print("-" * 70)
print(f"\\nEstimated CPU Time per image: {cpu_mean:.0f} ms")
"""))

# ── Build Notebook ──────────────────────────────────────────────────────────
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.0'},
    },
    'cells': cells,
}

out_path = Path('scripts/gated_unet_ensemble.ipynb')
out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Written: {out_path}  ({out_path.stat().st_size//1024} KB)')
