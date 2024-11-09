#!/bin/bash
echo "Starting setup.sh..."

# Install dependencies
echo "Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
  echo "Error installing dependencies from requirements.txt"
  exit 1
fi

# Install Streamlit if not included
echo "Installing Streamlit..."
python -m pip install streamlit
if [ $? -ne 0 ]; then
  echo "Error installing Streamlit"
  exit 1
fi

# Run the Streamlit app
echo "Running Streamlit app..."
streamlit run app.py &
if [ $? -ne 0 ]; then
  echo "Error running Streamlit app"
  exit 1
fi

echo "Setup complete."
