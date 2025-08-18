
# Superstore DS Project (Full Lifecycle)

**EDA**,**Dashboard**, and training a **Prediction model** on the Classic Superstore dataset (10k rows approx).

## How to use (using UV)

1. Full CSV present at: `data/superstore.csv`  

2. Create a virtual env and install requirements:
   ```bash
   uv venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   uv pip install -r requirements.txt
   ```

3. Run EDA (saves figures in `reports/`):
   ```bash
   uv run notebooks/01_eda.py --csv data/superstore.csv
   ```

4. Train models and evaluate (saves model + metrics):
   ```bash
   uv run src/train_model.py --csv data/superstore.csv --target Profit
   ```

5. Launch the interactive dashboard:
   ```bash
   uv run streamlit run app/streamlit_app.py
   ```

## Targets you can predict
- Regression: `Profit` (default) or `Sales`  
- Classification: `IsProfitable` (1 if Profit>0 else 0) with `--task classify`

## Files
- `notebooks/01_eda.py`: Quick, reproducible EDA script (matplotlib only)
- `src/train_model.py`: End-to-end ML pipeline (sklearn)
- `src/features.py`: Feature engineering utilities
- `app/streamlit_app.py`: Streamlit dashboard
- `requirements.txt`: Python dependencies

---
## Dataset Info
Superstore dataset by https://github.com/VivekChowdhury23
Link to dataset : [https://github.com/VivekChowdhury23](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

## Note :
Models included are not tuned and may produce undesirable results.
They serve as a boilerplate for further improvements.
The dashboard may have bugs, as some parts were written with the help of AI and are not yet thoroughly tested.

