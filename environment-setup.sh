#!/bin/bash

# Exit on error
set -e

# Create .venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
else
    echo ".venv already exists, skipping creation."
fi

# Activate venv
echo "Activating .venv..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Jupyter and kernel tools
echo "Installing Jupyter and ipykernel..."
pip install jupyter ipykernel

# Register .venv as a Jupyter kernel
echo "Registering .venv as Jupyter kernel..."
python -m ipykernel install --user --name=.venv --display-name ".venv"

echo "✅ Setup complete! Open VS Code and select the '.venv' kernel for notebooks."