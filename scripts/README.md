# Mora SP Cup 2026 Submission

This directory contains the final inference solution for the Mora SP Cup 2026.

## Architecture
The submitted model is a **Heavyweight Gated Residual U-Net** (NAFNet-style) featuring:
- **Width:** 32 base channels
- **Depth:** 4 Gated Blocks per stage
- **Parameters:** ~760,000

## Instructions

1. **Environment Setup:**
   Ensure you have installed the required dependencies from this folder:
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Test Images:** 
   Ensure the blind test images (e.g., `461_noise.png` to `480_noise.png`) are placed inside `../competition_data/submissions/noisy/`.

3. **Run Inference:**
   To denoise the directory of images, run the following command from within the `scripts/` folder:

```bash
python denoise.py --noise_dir ../competition_data/submissions/noisy --denoised_dir ../competition_data/submissions/denoised --device auto
```

### Options
- `--device auto`: Automatically uses CUDA if available, falling back to CPU if not.
- `--device cpu`: Forces strict CPU execution.

The script expects the trained weights to be present at `weights/best_model.pth`.
