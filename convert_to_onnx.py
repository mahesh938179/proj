import numpy as np
import os, subprocess
import tensorflow as tf
from tensorflow import keras
import onnxruntime as rt
import glob
from pathlib import Path

# Batch conversion utility for Django Stock Predictor
# Converts Keras .keras and MMHPA LSTM models to ONNX for production

BASE_DIR = Path(__file__).resolve().parent
GAN_MODELS_DIR = BASE_DIR / 'outputs' / 'models' / 'gan'
SEQ_LEN    = 20
N_FEATURES = 7 # This is for TCS, but we can detect it

def convert_generator(keras_path_or_model, onnx_path, is_keras_model=False):
    if is_keras_model:
        original = keras_path_or_model
        print(f"Using provided Keras model object")
    else:
        print(f"Opening model: {keras_path_or_model}")
        # ── Step 1: Load original model & extract weights ──────────────
        original = keras.models.load_model(keras_path_or_model)
    
    weights  = original.get_weights()
    print(f'Loaded original model, {len(weights)} weight tensors')

    # ── Step 2: Rebuild a clean Sequential model with same architecture ──
    # Using detected input shape
    detected_shape = original.input_shape
    # If it's a list, take first
    if isinstance(detected_shape, list):
        detected_shape = detected_shape[0]
    
    # detected_shape might be (None, 20, 7) or [(None, 20, 7)]
    input_shape = detected_shape[1:]
    print(f"Detected input shape: {input_shape}")

    inp = keras.Input(shape=input_shape, name='input')
    x   = inp
    for layer in original.layers:
        x = layer(x)
    clean_model = keras.Model(inputs=inp, outputs=x, name='generator_clean')
    clean_model.set_weights(weights)
    print('Clean model built [OK]')

    # ── Step 3: Export as SavedModel ───────────────────────────────
    # We use a path in the project instead of /tmp for portability and robustness
    temp_dir = BASE_DIR / "tmp_conversion"
    temp_dir.mkdir(exist_ok=True)
    
    # If it's a path/str, use basename, otherwise use a generic name
    if isinstance(keras_path_or_model, (str, Path)):
        base_name = os.path.basename(str(keras_path_or_model)).replace('.keras', '')
    else:
        base_name = "model_obj"
        
    saved_path = str(temp_dir / f"clean_{base_name}")
    
    # Try keras 3.x clean export first, if fails try tf.saved_model.save
    try:
        clean_model.export(saved_path)
        print(f'Exported to {saved_path} via keras.export [OK]')
    except AttributeError:
        tf.saved_model.save(clean_model, saved_path)
        print(f'Exported to {saved_path} via tf.saved_model.save [OK]')
    except Exception as e:
        print(f"Failed to export: {e}")
        return False

    # ── Step 4: Convert SavedModel -> ONNX via CLI ─────────────────
    onnx_path_str = str(onnx_path)

    result = subprocess.run([
        'python', '-m', 'tf2onnx.convert',
        '--saved-model', saved_path,
        '--output',      onnx_path_str,
        '--opset',        '13'
    ], capture_output=True, text=True)

    # print(result.stdout[-800:]) # Slicing might fail if empty
    if result.returncode != 0:
        print('STDERR:', result.stderr if result.stderr else "No stderr")
        return False

    print(f'Done saved: {onnx_path_str}')
    return True

if __name__ == "__main__":
    # --- Generators ---
    keras_files = glob.glob(str(GAN_MODELS_DIR / "*_generator.keras"))
    for kf in keras_files:
        of = kf.replace(".keras", ".onnx")
        if os.path.exists(of):
            print(f"Skipping {kf}, {of} already exists")
            continue
        print(f"Converting {kf} to {of}...")
        try:
            success = convert_generator(kf, of)
            if success:
                print(f"Successfully converted {kf}")
            else:
                print(f"Failed to convert {kf}")
        except Exception as e:
            print(f"Error during conversion of {kf}: {e}")

    # --- MMHPA models ---
    mmhpa_dir = BASE_DIR / 'outputs' / 'models' / 'mmhpa'
    mmhpa_files = glob.glob(str(mmhpa_dir / "*.pkl"))
    for mf in mmhpa_files:
        print(f"Checking {mf} for nonlinear Keras model...")
        try:
            import pickle
            import sys
            ml_dir = str(BASE_DIR / 'ml_modules')
            if ml_dir not in sys.path:
                sys.path.insert(0, ml_dir)
            
            with open(mf, 'rb') as f:
                m = pickle.load(f)
            
            if hasattr(m, 'nonlinear_model') and m.nonlinear_model is not None:
                if 'keras' in str(type(m.nonlinear_model)):
                    onf = mf.replace(".pkl", "_nonlinear.onnx")
                    if os.path.exists(onf):
                        print(f"Skipping nonlinear model from {mf}, {onf} already exists")
                        continue
                    print(f"Converting nonlinear model from {mf} to {onf}...")
                    success = convert_generator(m.nonlinear_model, onf, is_keras_model=True)
                    if success:
                        print(f"Successfully converted nonlinear model from {mf}")
                    else:
                        print(f"Failed to convert nonlinear model from {mf}")
                else:
                    print(f"Nonlinear model in {mf} is not Keras (type: {type(m.nonlinear_model)})")
        except Exception as e:
            print(f"Error checking {mf}: {e}")
