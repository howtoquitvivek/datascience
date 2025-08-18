###
###

# 📊 Superstore DS Project (Full Lifecycle)

End-to-end **Data Science workflow** on the Classic *Superstore* dataset (\~10k rows):

* 🔍 **EDA**
* 📈 **Dashboard**
* 🤖 **Prediction Model Training**

---

## 🚀 How to Use (with UV)

1. **Dataset**
   Ensure the full CSV is present at:

   ```
   data/superstore.csv
   ```

2. **Setup Environment**
   Create a virtual environment and install requirements:

   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

   *(no manual activation required — `uv run` handles it automatically)*

3. **Run EDA** (saves figures in `reports/`):

   ```bash
   uv run notebooks/01_eda.py --csv data/superstore.csv
   ```

4. **Train Models & Evaluate** (saves model + metrics):

   ```bash
   uv run src/train_model.py --csv data/superstore.csv --target Profit
   ```

5. **Launch Dashboard**

   ```bash
   uv run streamlit run app/streamlit_app.py
   ```

---

## 🎯 Targets You Can Predict

* **Regression**:

  * `Profit` *(default)*
  * `Sales`
* **Classification**:

  * `IsProfitable` *(1 if Profit > 0 else 0)*

    ```bash
    uv run src/train_model.py --csv data/superstore.csv --target IsProfitable --task classify
    ```

---

## 📂 Project Structure

* `notebooks/01_eda.py` → Quick, reproducible EDA script *(matplotlib only)*
* `src/train_model.py` → End-to-end ML pipeline *(sklearn)*
* `src/features.py` → Feature engineering utilities
* `app/streamlit_app.py` → Streamlit dashboard
* `requirements.txt` → Python dependencies

---

## 📑 Dataset Info

* Source: [Superstore dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
* Credits: [VivekChowdhury23](https://github.com/VivekChowdhury23)

---

## ⚠️ Notes

* Models are **not tuned** and may produce **undesirable results**.
  
  ➝ They serve as a **boilerplate** for further improvements.
* The **dashboard may contain bugs**, as some parts were generated with AI and are **not yet thoroughly tested**.


