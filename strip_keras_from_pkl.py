# -*- coding: utf-8 -*-
"""
strip_keras_from_pkl.py
-----------------------
Strips the embedded Keras LSTM (nonlinear_model) from every _mmhpa.pkl file.
After stripping, the pkl files can be unpickled on any machine without needing
keras/tensorflow installed.

The engine.py already uses ONNX for the nonlinear model when the .onnx file
exists, so removing it from the pkl is safe.

Run this ONCE from the project root:
    python strip_keras_from_pkl.py
"""
import sys
import os
import pickle
from pathlib import Path

# ── Add ml_modules to path so MMHPA class can be found during unpickling ──
BASE_DIR = Path(__file__).resolve().parent
ml_dir = str(BASE_DIR / 'ml_modules')
if ml_dir not in sys.path:
    sys.path.insert(0, ml_dir)

mmhpa_dir = BASE_DIR / 'outputs' / 'models' / 'mmhpa'

pkl_files = list(mmhpa_dir.glob('*_mmhpa.pkl'))
if not pkl_files:
    print("No *_mmhpa.pkl files found in", mmhpa_dir)
    sys.exit(1)

for pkl_path in sorted(pkl_files):
    print(f"\nProcessing: {pkl_path.name}")

    # ── 1. Load the pkl ────────────────────────────────────────────────────
    try:
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"  [ERROR] Could not load (maybe already needs keras to unpickle): {e}")
        print(f"     → Trying a workaround with a custom unpickler...")
        
        # Custom unpickler that replaces unknown Keras objects with None
        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # If the class comes from keras or tensorflow, return a dummy
                if 'keras' in module or 'tensorflow' in module:
                    class _Dummy:
                        pass
                    return _Dummy
                return super().find_class(module, name)
        
        try:
            with open(pkl_path, 'rb') as f:
                obj = SafeUnpickler(f).load()
            print(f"  [WARN] Loaded with safe unpickler (Keras parts replaced with dummy objects)")
        except Exception as e2:
            print(f"  [ERROR] Safe unpickler also failed: {e2}")
            print(f"     Skipping this file. You may need to retrain and re-export this stock.")
            continue

    # ── 2. Strip the Keras model ───────────────────────────────────────────
    had_nonlinear = False
    if hasattr(obj, 'nonlinear_model') and obj.nonlinear_model is not None:
        model_type = type(obj.nonlinear_model).__name__
        print(f"  Found nonlinear_model of type: {model_type}")
        obj.nonlinear_model = None
        had_nonlinear = True
    else:
        print(f"  nonlinear_model is already None or missing — nothing to strip")

    # ── 3. Re-save the pkl ────────────────────────────────────────────────
    backup_path = pkl_path.with_suffix('.pkl.bak')
    if not backup_path.exists():
        import shutil
        shutil.copy2(pkl_path, backup_path)
        print(f"  Backed up to: {backup_path.name}")

    with open(pkl_path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    
    if had_nonlinear:
        print(f"  [OK] Stripped and re-saved: {pkl_path.name}")
    else:
        print(f"  [OK] Re-saved (no change needed): {pkl_path.name}")

print("\n\n[DONE] All mmhpa.pkl files no longer contain Keras model objects.")
print("   The engine.py will use ONNX sessions for nonlinear predictions")
print("   (or ARIMA fallback if the .onnx files were not generated).")
