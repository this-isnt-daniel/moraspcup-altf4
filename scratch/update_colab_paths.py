import json
with open('Mora_SP_Cup_Colab_Denoising.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 3
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'drive.mount(' in src and 'import tempfile' not in src:
            cell['source'] = [
                "# Local Execution Mode\n",
                "# from google.colab import drive\n",
                "# drive.mount('/content/drive')\n",
                "print('Running locally without Google Drive!')\n"
            ]

# Update Cell 9
NEW_CELL_9 = """from pathlib import Path
import json
import os

# Use local repository paths!
REPO_ROOT = Path.cwd().absolute()
DATASET_ROOT = REPO_ROOT / "competition_data"

TRAIN_NOISY_DIR = DATASET_ROOT / "public" / "noisy"
TRAIN_CLEAN_DIR = DATASET_ROOT / "public" / "ground_truth"
TEST_DIR = DATASET_ROOT / "submissions" / "noisy"

paths = {
    "train_noisy": TRAIN_NOISY_DIR,
    "train_clean": TRAIN_CLEAN_DIR,
    "test": TEST_DIR,
}

for name, folder in paths.items():
    if not folder.is_dir():
        raise FileNotFoundError(f"Cannot access: {folder}")
    count = sum(1 for p in folder.glob("*.png") if p.is_file())
    print(f"{name}: {count} PNG images")

# Update ALL output paths to local runs folder
DRIVE_ROOT = REPO_ROOT / "colab_outputs"
SEARCH_ROOT = REPO_ROOT
RUN = DRIVE_ROOT / "runs" / RUN_NAME
SCRIPTS = RUN / "scripts"
CKPT = RUN / "checkpoints"

for folder in (RUN, SCRIPTS, CKPT):
    folder.mkdir(parents=True, exist_ok=True)

(RUN / "paths.json").write_text(
    json.dumps({name: str(path) for name, path in paths.items()}, indent=2)
)

print("\\nOutputs will be saved to:", RUN)
print("SUCCESS — continue with Cell 5.")
"""

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'tempfile.mkdtemp' in src and 'mora_drive_' in src:
            # We found Cell 9
            cell['source'] = [line + '\n' for line in NEW_CELL_9.split('\n')]
            print('Updated dataset paths cell!')

with open('Mora_SP_Cup_Colab_Denoising.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
