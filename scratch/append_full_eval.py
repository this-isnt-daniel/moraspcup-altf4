import json

with open('Mora_SP_Cup_Colab_Denoising.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new markdown and code cells to evaluate the entire public folder
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Evaluate Output Metrics on Full Public Set\n",
        "Calculates the final PSNR, SSIM, and Composite Score over all 460 images in the public folder to match the official evaluator."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "from tqdm.auto import tqdm\n",
        "import time\n",
        "import torch\n",
        "\n",
        "def evaluate_full_public(net):\n",
        "    all_ids = sorted(list(noisy_paths.keys()))\n",
        "    rows = []\n",
        "    for i in tqdm(all_ids, desc='Full Public Eval (460 imgs)'):\n",
        "        y = read_image(noisy_paths[i])\n",
        "        x = read_image(clean_paths[i])\n",
        "        begin = time.perf_counter()\n",
        "        \n",
        "        # Predict using best model settings\n",
        "        pred = predict(net, y, DEVICE, CFG['tile'], CFG['overlap'], tta=USE_TTA)\n",
        "        \n",
        "        # Blend with wavelet if CLASSICAL_ALPHA > 0\n",
        "        if CLASSICAL_ALPHA > 0:\n",
        "            classical = wavelet_prediction(y)\n",
        "            pred = (1 - CLASSICAL_ALPHA) * pred + CLASSICAL_ALPHA * classical\n",
        "            \n",
        "        pred = np.clip(pred, 0, 1)\n",
        "        \n",
        "        if DEVICE.type == 'cuda': torch.cuda.synchronize()\n",
        "        \n",
        "        metrics = metric_row(y, x, pred)\n",
        "        rows.append(dict(id=i, seconds=time.perf_counter()-begin, **metrics))\n",
        "        \n",
        "    frame = pd.DataFrame(rows)\n",
        "    return frame, frame.drop(columns='id').mean().to_dict()\n",
        "\n",
        "# Load the best selected model and evaluate on the full 460 images\n",
        "print('Loading best selected weights...')\n",
        "best_net = load_model(checkpoint_paths[0], DEVICE)\n",
        "\n",
        "print('Evaluating on ALL 460 public images...')\n",
        "full_frame, full_metrics = evaluate_full_public(best_net)\n",
        "\n",
        "full_frame.to_csv(RUN / 'full_public_evaluation.csv', index=False)\n",
        "\n",
        "print('\\n' + '='*50)\n",
        "print('FULL PUBLIC SET METRICS (460 images)')\n",
        "print('='*50)\n",
        "print(f\"Composite Score : {full_metrics['score']:.6f}\")\n",
        "print(f\"Mean PSNR       : {full_metrics['psnr']:.4f} dB\")\n",
        "print(f\"Mean SSIM       : {full_metrics['ssim']:.6f}\")\n",
        "print(f\"Delta PSNR      : {full_metrics['delta_psnr']:+.4f} dB\")\n",
        "print(f\"Delta SSIM      : {full_metrics['delta_ssim']:+.6f}\")\n",
        "print(f\"Avg Time/Image  : {full_metrics['seconds']*1000:.1f} ms\")\n",
        "print('='*50)\n"
    ]
}

# Insert it before the cell that tests standalone CLI (around index 39 or so)
# Let's find the Markdown cell with "15 — Test offline standalone CLI" (which might be numbered 14 or 15)
insert_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        if 'Test offline standalone CLI' in src:
            insert_idx = i
            break

nb['cells'].insert(insert_idx, code_cell)
nb['cells'].insert(insert_idx, markdown_cell)

with open('Mora_SP_Cup_Colab_Denoising.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
