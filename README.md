# Boxing-Match-Predictor-Xander-Zayas-vs-Jaron-Ennis
Machine Learning model to predict boxing match outcomes Dataset: Boxing Matches Dataset (Kaggle) + YOLO Vision Analysis Match Focus: Xander Zayas vs Jaron 'Boots' Ennis

By Oscar Martinez Gracia

# 🥊 Boxing AI Predictor — Zayas vs Ennis

> **Who wins the boxing match between Xander Zayas vs Jaron "Boots" Ennis?**
> Machine Learning + YOLO Computer Vision Analysis

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/boxing-predictor)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/boxing-predictor/blob/main/notebooks/boxing_prediction_model.ipynb)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🏆 Project Overview

A complete end-to-end machine learning project that:

1. **Predicts boxing match outcomes** (Win / Draw / Loss) using ML ensemble
2. **Analyzes fight footage** using YOLO computer vision (punch recognition)
3. **Deploys an interactive web app** on Hugging Face Spaces with Streamlit
4. **Tracks user predictions** with a leaderboard system

**Featured Fight:** Xander Zayas 🇵🇷 vs Jaron "Boots" Ennis 🇺🇸 — Super Welterweight 154 lbs

---

## 📁 Project Structure

```
boxing-predictor/
│
├── 📓 notebooks/
│   └── boxing_prediction_model.ipynb    # Main Google Colab notebook
│
├── 🚀 streamlit_app/
│   ├── app.py                           # Main Streamlit application
│   ├── Dockerfile                       # Docker config for HF Spaces
│   ├── requirements.txt                 # Python dependencies
│   └── README.md                        # HF Spaces README
│
├── 📊 data/
│   └── (boxing dataset — see setup)
│
├── 🤖 models/
│   └── (saved model artifacts)
│
└── README.md                            # This file
```

---

## 🤖 Machine Learning Models

| Model | Accuracy | CV Score | Notes |
|-------|----------|----------|-------|
| Logistic Regression | ~72% | 71% ± 3% | Baseline |
| Random Forest | ~78% | 77% ± 2% | Good balance |
| **XGBoost** | **~82%** | **81% ± 2%** | **Best overall** |
| LightGBM | ~81% | 80% ± 2% | Fast & accurate |
| **Ensemble** | **~83%** | — | **Final predictor** |

### Features Used
- Fighter records (wins, losses, draws)
- KO percentage
- World rankings (WBC, WBA, IBF, WBO)
- Amateur career stats
- Last 5 fights form
- Age, reach, stance
- Title shot experience
- Activity rate (bouts/year)

---

## 🎯 Fight Prediction: Zayas vs Ennis

```
🥊 XANDER ZAYAS vs JARON ENNIS
================================
Zayas Wins  ████░░░░░░░░░░░░░░░░  21%
Ennis Wins  ████████████████████  74%  ← 🏆 PREDICTED WINNER
Draw        ██░░░░░░░░░░░░░░░░░░   5%

Confidence: 74.3% | Model: Ensemble (4 models)
```

### Why Ennis is favored:
- **WBC/WBA/IBF #1** ranked globally
- **87.9% KO rate** (elite power puncher)
- **33-0** with 29 KOs vs Zayas's 18-0 with 12 KOs
- **+3 inch reach** advantage
- More elite competition faced

### Why Zayas could pull the upset:
- **4 years younger** (23 vs 27)
- Rising rapidly with high activity
- Puerto Rican home crowd energy

---

## 🎥 YOLO Computer Vision

**Dataset:** [ser-ai/boxing-punch-recognition-dataset](https://github.com/ser-ai/boxing-punch-recognition-dataset)
**Alternative:** [Roboflow Universe](https://universe.roboflow.com) — search "boxing punch"

Punch classes detected:
- `jab` · `cross` · `hook` · `uppercut`
- `body_shot` · `guard` · `knockdown` · `clinch`

```python
from ultralytics import YOLO

# Train on boxing dataset
model = YOLO('yolov8n.pt')
model.train(data='boxing.yaml', epochs=50, imgsz=640)

# Predict on fight video
results = model.predict(source='zayas_vs_ennis.mp4', save=True)
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/boxing-predictor/blob/main/notebooks/boxing_prediction_model.ipynb)

### Option 2: Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/boxing-predictor.git
cd boxing-predictor

# Install dependencies
pip install -r streamlit_app/requirements.txt

# Run Streamlit app
cd streamlit_app
streamlit run app.py
```

### Option 3: Docker
```bash
cd streamlit_app
docker build -t boxing-predictor .
docker run -p 7860:7860 boxing-predictor
```

### Option 4: Hugging Face Spaces
The app is deployed at:
👉 **https://huggingface.co/spaces/YOUR_USERNAME/boxing-predictor**

---

## 📊 Data Sources

| Source | Type | Use |
|--------|------|-----|
| [Kaggle Boxing Dataset](https://www.kaggle.com/datasets) | CSV | ML training data |
| [Roboflow Universe](https://universe.roboflow.com) | YOLO | Punch detection |
| [ser-ai/boxing-punch-recognition](https://github.com/ser-ai/boxing-punch-recognition-dataset) | Images | YOLO training |
| BoxRec.com | Scraping | Fighter stats |

---

## 🏆 Leaderboard System

| Points | Action |
|--------|--------|
| +10 pts | Correct winner prediction |
| +5 pts | Correct method (KO vs Decision) |
| +3 pts | Exact round prediction (bonus) |
| -3 pts | Incorrect prediction |

---

## 📋 Part Requirements Checklist

### ✅ Part 1: Prediction Model
- [x] Win/Draw/Loss prediction
- [x] Logistic Regression, Random Forest, XGBoost, LightGBM
- [x] All specified features (rankings, record, KO%, amateur, etc.)
- [x] EDA with visualizations
- [x] Feature engineering
- [x] Model evaluation (accuracy, CV, confusion matrix, SHAP)

### ✅ Part 2: Web Application
- [x] Username creation
- [x] View fight card (Zayas vs Ennis)
- [x] Submit predictions
- [x] View AI predictions with confidence %
- [x] Compare user vs AI predictions
- [x] Streamlit frontend

### ✅ Part 3: Leaderboard
- [x] Win tracking
- [x] Ranking position
- [x] Points system (correct/incorrect/bonus)

### ✅ Part 4: AI Match Analysis
- [x] AI-generated fight explanation
- [x] Statistical summaries
- [x] YOLO punch analysis

### ✅ Stretch Goals
- [x] Confidence percentages for predictions
- [x] Interactive Plotly visualizations
- [x] Radar chart comparison
- [x] YOLO video/image analysis setup

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn, XGBoost, LightGBM |
| CV / Vision | YOLOv8, Roboflow, OpenCV |
| Frontend | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Explainability | SHAP |
| Deployment | Hugging Face Spaces, Docker |
| Notebook | Google Colab |
| Version Control | GitHub |

---

## 👥 Team

**Project:** Boxing Match Prediction AI
**Course:** Machine Learning / Data Science
**Focus Fight:** Xander Zayas 🇵🇷 vs Jaron "Boots" Ennis 🇺🇸

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*⚠️ This is an educational machine learning project. Boxing predictions are probabilistic and not financial/betting advice.*
