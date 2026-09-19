import json

with open('scripts/gated_unet_hparam_sweep.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

extra_run_md_lines = [
    "## Cell 13b — Extra Run: patch=192 (original notebook patch size)\n",
    "Run this cell **after Cell 13 finishes** to add a 5th data point.\n\n",
    "> **patch=192** gives the model more spatial context per training step — this is\n",
    "> the main thing different from the original notebook vs the current sweep (patch=128)."
]

extra_run_code_lines = [
    "# Extra Run: patch=192, lr=3e-4, charbonnier_plus_ssim\n",
    "# Run this cell after Cell 13 finishes.\n",
    "import datetime\n",
    "\n",
    "extra_cfg = dict(run_id='run_lr3e4_ssim_p192', lr=3e-4,\n",
    "                 loss_weighting='charbonnier_plus_ssim', patch_size=192)\n",
    "\n",
    "print(f'Running extra config: {extra_cfg[\"run_id\"]}')\n",
    "BASE_CFG['patch'] = 192  # override patch size\n",
    "\n",
    "try:\n",
    "    history, best_ckpt = run_training(extra_cfg)\n",
    "except Exception as exc:\n",
    "    print(f'ERROR in training {extra_cfg[\"run_id\"]}: {exc}')\n",
    "    best_ckpt = None\n",
    "\n",
    "if best_ckpt and best_ckpt.exists():\n",
    "    run_id = extra_cfg['run_id']\n",
    "    epochs_trained = len(history)\n",
    "    last5_gain = history[-1]['selected_score'] - history[max(0,len(history)-6)]['selected_score']\n",
    "    still_improving = last5_gain > 0.01\n",
    "    best_diag_score = max(r['selected_score'] for r in history)\n",
    "    best_alpha = 0.0; best_official = -float('inf'); best_psnr = 0.0; best_ssim_val = 0.0\n",
    "    for alpha in ALPHA_CANDIDATES:\n",
    "        try:\n",
    "            official = official_eval_on_full_set(best_ckpt, run_id, alpha=alpha,\n",
    "                                                  tile=BASE_CFG['tile'], overlap=BASE_CFG['overlap'])\n",
    "            print(f'  alpha={alpha:.1f}: composite={official[\"composite\"]:.4f}')\n",
    "            if official['composite'] > best_official:\n",
    "                best_official = official['composite']\n",
    "                best_alpha = alpha; best_psnr = official['psnr']; best_ssim_val = official['ssim']\n",
    "        except Exception as exc:\n",
    "            print(f'  WARNING: official eval failed for alpha={alpha}: {exc}')\n",
    "    try:\n",
    "        cpu_mean, cpu_std = cpu_benchmark(best_ckpt, alpha=best_alpha,\n",
    "                                           tile=BASE_CFG['tile'], overlap=BASE_CFG['overlap'])\n",
    "    except Exception as exc:\n",
    "        print(f'WARNING: CPU benchmark failed: {exc}')\n",
    "        cpu_mean, cpu_std = float('nan'), float('nan')\n",
    "    gap = best_diag_score - best_official\n",
    "    row = dict(\n",
    "        run_id=run_id, learning_rate=extra_cfg['lr'],\n",
    "        loss_weighting=extra_cfg['loss_weighting'], patch_size=extra_cfg['patch_size'],\n",
    "        epochs_trained=epochs_trained, still_improving_at_end=still_improving,\n",
    "        best_alpha=best_alpha, notebook_diagnostic_score=round(best_diag_score, 6),\n",
    "        official_evaluate_score=round(best_official, 6),\n",
    "        diag_vs_official_gap=round(gap, 6),\n",
    "        mean_psnr=round(best_psnr, 4), mean_ssim=round(best_ssim_val, 6),\n",
    "        cpu_ms_per_image=round(cpu_mean, 2), cpu_ms_std=round(cpu_std, 2),\n",
    "        timestamp=datetime.datetime.now().isoformat(timespec='seconds')\n",
    "    )\n",
    "    import pandas as pd\n",
    "    df_row = pd.DataFrame([row])\n",
    "    if RESULTS_CSV.exists():\n",
    "        df_row.to_csv(RESULTS_CSV, mode='a', header=False, index=False)\n",
    "    else:\n",
    "        df_row.to_csv(RESULTS_CSV, index=False)\n",
    "    print(f'Done! official_score={best_official:.4f}  diag_gap={gap:+.4f}')\n",
    "    print(f'Result appended to {RESULTS_CSV}')\n",
    "\n",
    "BASE_CFG['patch'] = 128  # restore default\n",
]

md_cell = {'cell_type': 'markdown', 'metadata': {}, 'source': extra_run_md_lines}
code_cell = {
    'cell_type': 'code', 'execution_count': None,
    'metadata': {}, 'outputs': [],
    'source': extra_run_code_lines
}

# Insert before the last summary cell (Cell 14)
summary_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Cell 14' in ''.join(cell['source']):
        summary_idx = i
        break

if summary_idx is None:
    summary_idx = len(nb['cells'])

nb['cells'].insert(summary_idx, code_cell)
nb['cells'].insert(summary_idx, md_cell)

with open('scripts/gated_unet_hparam_sweep.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Added Cell 13b to notebook.')
