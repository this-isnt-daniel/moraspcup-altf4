import pandas as pd
import json
from pathlib import Path
import hashlib
import numpy as np

run_dir = Path('colab_outputs/runs/gated_unet_01')

def to_md_table(df, float_fmt='{:.4f}'):
    headers = '| ' + ' | '.join(df.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(df.columns)) + '|'
    rows = []
    for _, row in df.iterrows():
        row_strs = []
        for val in row:
            if isinstance(val, (float, np.float64, np.float32)):
                row_strs.append(float_fmt.format(val))
            else:
                row_strs.append(str(val))
        rows.append('| ' + ' | '.join(row_strs) + ' |')
    return '\n'.join([headers, sep] + rows)

out = []
out.append('# Mora SP Cup 2026 - Denoising Model Analysis Report\n')
out.append('*Auto-generated read-only analysis of existing outputs*\n\n')

# 1. PER-IMAGE FULL-460 EVALUATION SUMMARY
out.append('## 1. Per-Image Full-460 Evaluation Summary\n')
try:
    df_eval = pd.read_csv(run_dir / 'full_public_evaluation.csv')
    df_noise = pd.read_csv(run_dir / 'noise_brightness.csv') if (run_dir / 'noise_brightness.csv').exists() else None
    
    out.append('### Global Metrics (460 images)\n')
    summary = df_eval[['psnr', 'ssim', 'delta_psnr', 'delta_ssim', 'score']].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    summary.rename(columns={'index': 'metric'}, inplace=True)
    out.append(to_md_table(summary))
    out.append('\n\n')
    
    out.append('### Score Distribution\n')
    scores = df_eval['score']
    out.append(f'- **Below 0.3**: {(scores < 0.3).sum()} images\n')
    out.append(f'- **0.3 to 0.5**: {((scores >= 0.3) & (scores <= 0.5)).sum()} images\n')
    out.append(f'- **Above 0.5**: {(scores > 0.5).sum()} images\n\n')
    
    out.append('### 10 Worst-Scoring Images\n')
    worst10 = df_eval.nsmallest(10, 'score')
    out.append(to_md_table(worst10[['id', 'psnr', 'ssim', 'delta_psnr', 'delta_ssim', 'score']]))
    out.append('\n\n')
    
    out.append('### 10 Best-Scoring Images\n')
    best10 = df_eval.nlargest(10, 'score')
    out.append(to_md_table(best10[['id', 'psnr', 'ssim', 'delta_psnr', 'delta_ssim', 'score']]))
    out.append('\n\n')
    
    if df_noise is not None:
        out.append('### Worst-Scoring Analysis vs Noise/Brightness\n')
        worst_ids = worst10['id'].tolist()
        worst_noise = df_noise[df_noise['id'].isin(worst_ids)]
        out.append('Brightness and Noise levels for the worst 10 images:\n')
        out.append(to_md_table(worst_noise[['id', 'brightness', 'noise_std']]))
        out.append('\n\n')
        out.append(f'*(Overall dataset mean brightness: {df_noise["brightness"].mean():.4f}, mean noise_std: {df_noise["noise_std"].mean():.4f})*\n\n')
except Exception as e:
    out.append(f'Error loading full_public_evaluation.csv: {e}\n\n')


# 2. TRAINING HISTORY SUMMARY
out.append('## 2. Training History Summary\n')
try:
    df_hist = pd.read_csv(run_dir / 'training_history.csv')
    out.append(to_md_table(df_hist[['epoch', 'loss', 'raw_score', 'ema_score', 'psnr', 'ssim', 'selected']]))
    out.append('\n\n')
    
    first_10_gain = df_hist['raw_score'].iloc[min(9, len(df_hist)-1)] - df_hist['raw_score'].iloc[0]
    last_10_gain = df_hist['raw_score'].iloc[-1] - df_hist['raw_score'].iloc[max(0, len(df_hist)-11)]
    
    out.append(f'- **Improvement first 10 epochs**: +{first_10_gain:.4f}\n')
    out.append(f'- **Improvement last 10 epochs**: +{last_10_gain:.4f}\n')
    
    plateau_epoch = None
    for i in range(1, len(df_hist)):
        if (df_hist['raw_score'].iloc[i] - df_hist['raw_score'].iloc[i-1]) < 0.005:
            plateau_epoch = df_hist['epoch'].iloc[i]
            break
    if plateau_epoch:
        out.append(f'- **Plateau onset (gain < 0.005)**: Epoch {plateau_epoch}\n')
    
    best_row = df_hist.loc[df_hist[['raw_score', 'ema_score']].max(axis=1).idxmax()]
    best_score = max(best_row['raw_score'], best_row['ema_score'])
    out.append(f'- **Final Best Epoch**: {best_row["epoch"]} (Score: {best_score:.4f})\n\n')
except Exception as e:
    out.append(f'Error loading training_history.csv: {e}\n\n')


# 3. TRAIN/VAL SPLIT AND DUPLICATE-GROUPING INTEGRITY
out.append('## 3. Train/Val Split and Duplicate-Grouping Integrity\n')
try:
    with open(run_dir / 'split.json', 'r') as f:
        split_data = json.load(f)
    train_ids = split_data['train']
    val_ids = split_data['val']
    
    out.append(f'- **Train count**: {len(train_ids)}\n')
    out.append(f'- **Val count**: {len(val_ids)}\n')
    out.append('*(Note: Duplicate-grouping arrays were not exported to JSON/CSV in the Colab run, so overlap assertions cannot be computed offline without rerunning the duplicate detection script. However, the notebook natively enforces disjoint sets during splitting.)*\n\n')
except Exception as e:
    out.append(f'Error loading split data: {e}\n\n')


# 4. TTA / ALPHA-BLEND SELECTION SUMMARY
out.append('## 4. TTA / Alpha-Blend Selection Summary\n')
try:
    df_exp = pd.read_csv(run_dir / 'experiment_table.csv')
    out.append(to_md_table(df_exp))
    out.append('\n\n')
    
    best_exp = df_exp.loc[df_exp['score'].idxmax()]
    out.append(f'- **Selected Setting**: TTA={best_exp["tta"]}, Alpha={best_exp["alpha"]} (Score: {best_exp["score"]:.4f})\n')
    worst_score = df_exp['score'].min()
    out.append(f'- **Maximum gain over worst setting**: +{best_exp["score"] - worst_score:.4f}\n\n')
except Exception as e:
    out.append(f'Error loading experiment_table.csv: {e}\n\n')


# 5. CPU/GPU INFERENCE TIMING
out.append('## 5. CPU/GPU Inference Timing\n')
try:
    with open(run_dir / 'cpu_smoke.json', 'r') as f:
        cpu_data = json.load(f)
    
    out.append('| Device | Time/Image (ms) | Config |\n')
    out.append('|---|---|---|\n')
    for key, val in cpu_data.items():
        if isinstance(val, (float, int)):
            out.append(f'| Extracted | {val*1000:.2f} ms | {key} |\n')
        elif isinstance(val, dict) and 'seconds' in val:
            out.append(f'| Benchmark | {val["seconds"]*1000:.2f} ms | {key} |\n')
    
    out.append('\n*(The final exported `denoise.py` supports `--device auto` choosing CUDA if available, else CPU fallback.)*\n\n')
except Exception as e:
    out.append(f'Error loading cpu_smoke.json: {e}\n\n')


# 6. CHECKPOINT / FILE MANIFEST
out.append('## 6. Checkpoint / File Manifest\n')
try:
    sha256_path = run_dir / 'scripts' / 'sha256.json'
    
    with open(sha256_path, 'r') as f:
        sha_data = json.load(f)
        
    out.append('| Checkpoint Filename | Size (MB) | SHA-256 Checksum | Required Local Path |\n')
    out.append('|---|---|---|---|\n')
    
    for filename in sorted(sha_data.keys()):
        actual_path = run_dir / 'scripts' / filename
        if actual_path.exists():
            size_mb = actual_path.stat().st_size / (1024*1024)
            sha = sha_data[filename]
            out.append(f'| `{Path(filename).name}` | {size_mb:.2f} | `{sha}` | `scripts/{filename}` |\n')
            
    out.append('\n')
except Exception as e:
    out.append(f'Error compiling manifest: {e}\n\n')

Path('scratch/analysis_report.md').write_text(''.join(out), encoding='utf-8')
