# 🚀 Deployment Guide

## Step-by-Step: From Zero to Deployed

---

## 📓 PART A: Google Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload Notebook**
3. Upload `notebooks/boxing_prediction_model.ipynb`
4. Click **Runtime → Change runtime type → GPU (T4)**
5. Click **Runtime → Run All**

The notebook will:
- Install all dependencies
- Generate the boxing dataset
- Train 4 ML models
- Predict Zayas vs Ennis
- Run YOLO setup
- Save artifacts

---

## 🐙 PART B: GitHub Repository

```bash
# 1. Create new repo on github.com (name: boxing-predictor)

# 2. Clone and push
git clone https://github.com/YOUR_USERNAME/boxing-predictor.git
cd boxing-predictor

# Copy all project files here, then:
git add .
git commit -m "🥊 Initial commit: Boxing AI Predictor - Zayas vs Ennis"
git push origin main
```

### Add GitHub Secrets (for CI/CD):
1. Go to repo Settings → Secrets and Variables → Actions
2. Add `HF_TOKEN` = your Hugging Face write token

---

## 🤗 PART C: Hugging Face Spaces

### Option 1: Manual Upload (Easiest)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Settings:
   - **Name:** `boxing-predictor`
   - **SDK:** `Docker` ← Important!
   - **Visibility:** Public
4. Upload these files:
   - `streamlit_app/app.py` → `app.py`
   - `streamlit_app/Dockerfile` → `Dockerfile`
   - `streamlit_app/requirements.txt` → `requirements.txt`
   - `streamlit_app/README.md` → `README.md`
5. Space builds automatically! (~3-5 min)

### Option 2: Git Push

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create space
huggingface-cli repo create boxing-predictor --type space --space-sdk docker

# Clone space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/boxing-predictor
cd boxing-predictor

# Copy files
cp ../streamlit_app/app.py .
cp ../streamlit_app/Dockerfile .
cp ../streamlit_app/requirements.txt .
cp ../streamlit_app/README.md .

# Push
git add .
git commit -m "🥊 Deploy Boxing AI Predictor"
git push
```

### Option 3: GitHub → HF Auto-sync
- Update `.github/workflows/deploy.yml` with your username
- Add `HF_TOKEN` secret to GitHub
- Every push to `main` auto-deploys to HF Spaces

---

## 🎯 YOLO Dataset Setup

### From Roboflow Universe:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("boxing-punch-detection")
dataset = project.version(1).download("yolov8")
```

### From GitHub:
```bash
git clone https://github.com/ser-ai/boxing-punch-recognition-dataset
```

### Train YOLO:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='boxing.yaml', epochs=50, imgsz=640, device='cuda')
model.export(format='onnx')  # Export for deployment
```

---

## ✅ Deployment Checklist

- [ ] Colab notebook runs end-to-end
- [ ] GitHub repo created and code pushed
- [ ] README with badges and documentation
- [ ] HF Space created with Docker SDK
- [ ] app.py, Dockerfile, requirements.txt uploaded
- [ ] Space builds successfully (check Logs tab)
- [ ] All 4 tabs work: Fight Card, AI Prediction, Analytics, Leaderboard
- [ ] Username input works
- [ ] Prediction submission works
- [ ] Charts render correctly

---

## 🔗 Your Links (fill in after deployment)

| Resource | URL |
|----------|-----|
| Google Colab | `https://colab.research.google.com/github/YOUR_USERNAME/...` |
| GitHub Repo | `https://github.com/YOUR_USERNAME/boxing-predictor` |
| HF Space | `https://huggingface.co/spaces/YOUR_USERNAME/boxing-predictor` |
