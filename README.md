# VeritasAI - Fake News Detection System

An end-to-end Machine Learning web application designed to detect and classify whether a given news article or headline is real or fake using Natural Language Processing (NLP) and Support Vector Machines.

## Overview & Performance

- **Trained on:** 44,689 validated news articles from Reuters and investigative political archives.
- **NLP Vectorization:** TF-IDF with 100,000 sublinear unigram & bigram features (`ngram_range=(1,2)`).
- **Selected Model:** Linear Support Vector Classifier (`LinearSVC`).
- **Test Set Accuracy:** **99.80%**
- **Test Set F1-Score:** **99.79%**

### Model Benchmark Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Support Vector Machine (Selected)** | **99.80%** | **99.72%** | **99.86%** | **99.79%** |
| Logistic Regression | 99.24% | 98.90% | 99.50% | 99.20% |
| Multinomial Naive Bayes | 96.05% | 95.76% | 95.92% | 95.84% |

---

## Project Structure

```text
Fake-News-Detection/
├── backend/
│   └── main.py              # FastAPI REST API server
├── frontend/
│   ├── index.html           # Modern glassmorphic web UI
│   ├── style.css            # Responsive dark-theme styling
│   └── app.js               # Client controller & dynamic API resolution
├── model/
│   ├── best_model.joblib    # Serialized trained LinearSVC model
│   ├── tfidf_vectorizer.joblib # Serialized TF-IDF feature extractor
│   ├── preprocessing.py     # Reusable text cleaning & normalization
│   ├── train_model.py       # ML training and evaluation script
│   ├── evaluate_model.py    # Metric reporting and confusion matrix utilities
│   ├── model_comparison.csv # Benchmark results
│   └── confusion_matrix.png # Confusion matrix visualization
├── dataset/                 # Raw datasets (True.csv & Fake.csv)
├── render.yaml              # Render Blueprint configuration
├── requirements.txt         # Production dependencies
└── README.md                # Documentation
```

---

## Local Development Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Web Application: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Interactive API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Endpoint: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Deployment on Render

This project is pre-configured for deployment on [Render](https://render.com) using `render.yaml`:
- **Backend:** Render Web Service (FastAPI running on Uvicorn)
- **Frontend:** Render Static Site (HTML5/CSS3/ES6+ SPA)
