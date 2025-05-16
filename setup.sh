#!/bin/bash

# Create a virtual environment (optional)
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete! You can now run the app with: streamlit run app.py"
