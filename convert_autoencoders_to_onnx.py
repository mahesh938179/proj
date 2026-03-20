# -*- coding: utf-8 -*-
"""
convert_autoencoders_to_onnx.py
--------------------------------
Converts the Keras encoder inside each *_autoencoder.pkl to an ONNX file,
then replaces the StackedAutoencoder's .autoencoder and .encoder Keras models
with None, re-saves the pkl, and saves an ONNX file for the encoder.

After running this:
  - outputs/models/autoencoder/{SYMBOL}_encoder.onnx  <- new ONNX encoder
  - outputs/models/autoencoder/{SYMBOL}_autoencoder.pkl <- Keras models stripped

engine.py's _run_feature_pipeline() calls self.autoencoder.transform(feat_matrix)
which calls self.encoder.predict(...). We patch the StackedAutoencoder class
to use the ONNX session when the Keras encoder is None.

Run once from the project root:
    python convert_autoencoders_to_onnx.py
"""

import sys
import os
import pickle
import subprocess
import shutil
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ml_dir = str(BASE_DIR / 'ml_modules')
if ml_dir not in sys.path:
    sys.path.insert(0, ml_dir)

ae_dir = BASE_DIR / 'outputs' / 'models' / 'autoencoder'
tmp_dir = BASE_DIR / 'tmp_conversion'
tmp_dir.mkdir(exist_ok=True)

pkl_files = sorted(ae_dir.glob('*_autoencoder.pkl'))
if not pkl_files:
    print("No *_autoencoder.pkl files found in", ae_dir)
    sys.exit(1)

for pkl_path in pkl_files:
    symbol = pkl_path.stem.replace('_autoencoder', '')
    onnx_path = ae_dir / f'{symbol}_encoder.onnx'

    print(f"\n{'='*50}")
    print(f"Processing: {pkl_path.name}  ->  {onnx_path.name}")

    # ── 1. Load the pkl ────────────────────────────────────────────────────
    try:
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f)
        print(f"  Loaded OK  (type: {type(obj).__name__})")
    except Exception as e:
        print(f"  [ERROR] Could not load: {e}")
        print(f"  Skipping {symbol}")
        continue

    # ── 2. Export encoder as SavedModel ────────────────────────────────────
    if not onnx_path.exists():
        encoder = obj.encoder
        if encoder is None:
            print(f"  encoder is already None -- skipping ONNX conversion")
        else:
            saved_path = str(tmp_dir / f'encoder_{symbol}')
            print(f"  Exporting encoder to SavedModel: {saved_path}")
            try:
                # Try keras 3.x export
                encoder.export(saved_path)
                print(f"  Exported via .export() [OK]")
            except Exception as e1:
                try:
                    import tensorflow as tf
                    tf.saved_model.save(encoder, saved_path)
                    print(f"  Exported via tf.saved_model.save() [OK]")
                except Exception as e2:
                    print(f"  [ERROR] Export failed: {e1} / {e2}")
                    continue

            # Convert SavedModel -> ONNX
            print(f"  Converting to ONNX: {onnx_path}")
            result = subprocess.run([
                'python', '-m', 'tf2onnx.convert',
                '--saved-model', saved_path,
                '--output', str(onnx_path),
                '--opset', '13'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  [ERROR] tf2onnx failed:\n{result.stderr[-500:]}")
                continue
            print(f"  ONNX saved: {onnx_path.name} [OK]")

            # Cleanup tmp
            shutil.rmtree(saved_path, ignore_errors=True)
    else:
        print(f"  ONNX already exists, skipping conversion")

    # ── 3. Strip Keras objects from pkl ────────────────────────────────────
    had_keras = False
    if obj.encoder is not None or obj.autoencoder is not None:
        had_keras = True
        obj.encoder = None
        obj.autoencoder = None

    if had_keras:
        bak = pkl_path.with_suffix('.pkl.bak')
        if not bak.exists():
            shutil.copy2(pkl_path, bak)
            print(f"  Backed up to: {bak.name}")
        with open(pkl_path, 'wb') as f:
            pickle.dump(obj, f, protocol=4)
        print(f"  [OK] Stripped Keras models and re-saved: {pkl_path.name}")
    else:
        print(f"  Keras models already None -- no change to pkl")

print("\n\n[DONE] All autoencoder pkl files processed.")
print("  Next: update StackedAutoencoder.transform() to use ONNX when encoder is None.")
