---
title: Diamond Price Predictor
emoji: 💎
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.7.1"
python_version: 3.11
app_file: app.py
pinned: false
---
<div align="center">

# 💎 Diamond Price Appraiser AI
### *Precision Valuations. Data-Driven Insights. Fair Prices.*

[![Gradio](https://img.shields.io/badge/Gradio-FF4B4B?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/AbdullahKS-Devhub/diamond-price-prediction)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

<br/>

> **Enter physical dimensions and GIA grades. Let AI assess the market value. Know the true price.**

<br/>

🚀 **[Try the Live Demo →](https://huggingface.co/spaces/AbdullahKS-Devhub/diamond-price-prediction)**

</div>

---

## ✨ Features

- 💎 **Instant Valuations** — Highly accurate price predictions based on market data
- 🧠 **Automated ML Pipeline** — End-to-end MLOps from data ingestion to model training
- 🎛️ **Dual Interfaces** — Beautiful Gradio UI for humans, REST FastAPI for developers
- 📈 **Experiment Tracking** — Integrated with MLflow to continuously track model metrics (R², RMSE) 
- ⚡ **Cached Inference** — `@lru_cache` ensures model and preprocessor artifacts load instantly
- 🌐 **Zero Setup for Users** — Fully deployed on Hugging Face Spaces

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Gradio + Custom CSS (Clean Light Theme) |
| **API Backend** | FastAPI + Uvicorn |
| **ML Models** | Scikit-learn (Random Forest, XGBoost, Linear Regression) |
| **Experiment Tracking**| MLflow |
| **Data Processing** | Pandas, NumPy, ColumnTransformer |
| **Model Persistence** | Pickle (`.pkl` files) |
| **Deployment** | Hugging Face Spaces |
| **Version Control** | Git |

---

## 🧠 How It Works

```text
User enters diamond properties in Gradio / FastAPI
              ↓
Inputs validated via Pandas & Pydantic
              ↓
Preprocessor.pkl applies standard scaling & categorical encoding
              ↓
Best performing ML Model predicts the price
              ↓
Result rendered elegantly on UI alongside a Market Tier (e.g. Premium)
```

---

## 📥 Input Parameters (The 4 C's & Dimensions)

<details>
<summary>⚖️ <strong>Physical Properties & Quality</strong></summary>
<br/>

| Parameter | Type | Description |
|---|---|---|
| **Carat** | Numeric | Weight of the diamond (0.2 - 5.0) |
| **Cut** | Categorical | Fair, Good, Very Good, Premium, Ideal |
| **Color** | Categorical | D (Best) through J (Warm) |
| **Clarity** | Categorical | IF (Flawless) down to I1 (Included) |

</details>

<details>
<summary>📐 <strong>Dimensions</strong></summary>
<br/>

| Parameter | Type | Description |
|---|---|---|
| **Depth %** | Numeric | Total depth percentage |
| **Table %** | Numeric | Width of the top relative to widest point |
| **X** | Numeric | Length in mm |
| **Y** | Numeric | Width in mm |
| **Z** | Numeric | Height in mm |

</details>

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/abdullahks-devhub/diamond-price-prediction-with-mlops-pipeline.git
cd diamond-price-prediction-with-mlops-pipeline
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

**3. Run the Gradio App**
```bash
python app.py
```

**4. Run the FastAPI Endpoint**
```bash
uvicorn src.api.app:app --reload
```

---

## 📁 Project Structure

```text
.
├── notebooks/                # Exploratory Data Analysis (EDA)
│   └── eda.ipynb
├── src/                      # Source Code
│   ├── api/                  # FastAPI Application
│   │   └── app.py
│   ├── components/           # Core ML Logic
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/             # Training orchestration
│   ├── config.py             # Centralized configuration
│   └── logger.py             # Custom logging format
├── upload_model.py           # HF Hub upload script
├── main.py                   # CLI entry point for training & upload
├── app.py                    # Gradio Web UI
├── Dockerfile                # API Containerization
└── requirements.txt          # Python dependencies
```

---

## 📊 Dataset

| Dataset | Source | Records | Features |
|---|---|---|---|
| Diamonds Dataset | Kaggle | ~54,000 | 9 predicting features, 1 target |

The dataset undergoes robust preprocessing: numerical features are standardized utilizing `StandardScaler` and categorical features are mapped via One-Hot Encoding to ensure optimum algorithm performance.

---

## ⚠️ Disclaimer

> This application is built **for educational and demonstration purposes only**.
> It is **not** a substitute for professional certified gemology appraisals.
> Always consult a qualified jeweller before buying or selling diamonds.

---

<div align="center">

Made with ❤️ by **[Abdullah Khan](https://github.com/abdullahks-devhub)**

⭐ Star this repo if you found it useful!

</div>
