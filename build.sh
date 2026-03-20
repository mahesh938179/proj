#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Convert ML models to ONNX if needed (optional, depends on your workflow)
# python convert_to_onnx.py

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate

# Initialize App Data
python manage.py setup_stocks
python manage.py generate_token
python manage.py create_admin 
python manage.py set_role MAHESH superadmin
# python manage.py run_predictions # Optional: Only if this is fast!
