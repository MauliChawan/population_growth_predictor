# 🇺🇸 USA Population Growth Rate Predictor

A Machine Learning project that predicts future **US population growth rates** using historical data from 1961–2022.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📊 Dataset

| Field | Details |
|-------|---------|
| Source | [FRED — Federal Reserve Economic Data](https://fred.stlouisfed.org/) |
| Column | `SPPOPGROWUSA` — Annual US Population Growth Rate (%) |
| Period | 1961 – 2022 (62 years) |
| Rows | 62 |

---

## 🤖 Models Trained

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline — fits the best straight line |
| Random Forest | Ensemble of decision trees |
| **Gradient Boosting** | Boosted trees (typically best performer) |

---

## 📁 Project Structure

```
car-mileage/
│
├── data/
│   └── Population-Growth.csv       ← Raw dataset
│
├── models/
│   ├── best_model.pkl              ← Saved trained model
│   └── analysis_plots.png          ← Charts generated during training
│
├── notebooks/
│   └── population_growth_predictor.ipynb  ← Full walkthrough (Google Colab ready)
│
├── webapp/
│   └── predict.py                  ← Run predictions from command line
│
├── train_model.py                  ← Main training script
├── requirements.txt                ← Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Balagangadhar-Dev/car-mileage.git
cd car-mileage
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

### 4. Make a prediction
```bash
python webapp/predict.py --year 2027
```

---

## 📓 Google Colab

Run the full notebook directly in your browser — no installation needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Balagangadhar-Dev/car-mileage/blob/main/notebooks/population_growth_predictor.ipynb)

> **Steps in Colab:**
> 1. Click the badge above
> 2. Upload `data/Population-Growth.csv` using the Files panel (left sidebar)
> 3. Run all cells (Runtime → Run all)

---

## 📈 Results

The models predict US population growth rate (%) using:
- **Lag features** — growth rates from 1, 2, and 3 years prior
- **Rolling averages** — 3-year and 5-year trend averages
- **Year** — captures long-term trends

### 🔮 Sample Forecast (2023–2032)

| Year | Predicted Growth Rate |
|------|----------------------|
| 2023 | ~0.40% |
| 2025 | ~0.45% |
| 2030 | ~0.50% |

---

## 🛠 Tech Stack

- **Python 3.8+**
- **pandas** — data manipulation
- **scikit-learn** — machine learning
- **matplotlib** — visualisation
- **Jupyter Notebook** — interactive analysis

---

## 👤 Author

**Balagangadhar-Dev**  
GitHub: [@maulichawan](https://github.com/maulichawan)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
