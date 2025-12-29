#!/bin/bash
set -e

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Build/prepare your app (adjust as needed for your app structure)
echo "Dependencies installed successfully"