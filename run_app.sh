#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/isaacs_mac/datascienceproject"
PYTHON_BIN="/Users/isaacs_mac/anaconda3/bin/python"
STREAMLIT_BIN="/Users/isaacs_mac/anaconda3/bin/streamlit"
DATA_FILE="$PROJECT_DIR/data/raw/online_shoppers_intention.csv"

cd "$PROJECT_DIR"

if [ ! -f "$DATA_FILE" ]; then
  echo "Dataset not found at: $DATA_FILE"
  echo "Put online_shoppers_intention.csv into data/raw/ and run again."
  exit 1
fi

echo "Training model artifacts..."
"$PYTHON_BIN" scripts/train_model.py --data-path "$DATA_FILE"

echo "Starting Streamlit on http://127.0.0.1:8501"
exec "$STREAMLIT_BIN" run app/app.py --server.address 127.0.0.1 --server.port 8501
