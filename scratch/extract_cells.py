import json

with open('scripts/nafnet_script.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

key_cells = {
    'imports_and_cfg': 7,
    'paths': 9,
    'image_indexer': 11,
    'core_io': 15,
    'dedup_split': 19,
    'core_model': 21,
    'core_inference': 23,
    'dataset_loader': 25,
    'loss_fn': 27,
    'evaluate_fn': 29,
    'training_loop': 31,
}

extracted = {}
for name, idx in key_cells.items():
    src = ''.join(nb['cells'][idx]['source'])
    extracted[name] = src

with open('scratch/extracted_cells.json', 'w', encoding='utf-8') as f:
    json.dump(extracted, f, indent=2)

print('Done')
