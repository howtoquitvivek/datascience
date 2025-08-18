
# Superstore DS Project (Full Lifecycle)

This starter kit helps you run **EDA**, build a **dashboard**, and train a **prediction model** on the Superstore dataset (10k+ rows).

## How to use

1. Put your full CSV at: `data/superstore.csv`  
   - If you don't have it yet, test with `data/superstore_sample.csv` (5 rows).

2. Create a virtual env and install requirements:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

3. Run EDA (saves figures in `reports/`):
   ```bash
   python notebooks/01_eda.py --csv data/superstore.csv  # or superstore_sample.csv
   ```

4. Train models and evaluate (saves model + metrics):
   ```bash
   python src/train_model.py --csv data/superstore.csv --target Profit
   ```

5. Launch the interactive dashboard:
   ```bash
   streamlit run app/streamlit_app.py
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

**Tip:** If your dates look like `11/8/2016`, they are month/day/year (US). The code handles this automatically.
