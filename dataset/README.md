# Dataset Directory

The raw dataset CSV files (`True.csv` and `Fake.csv`, ~116 MB total) are excluded from Git to keep the deployment lightweight and under GitHub file size limits.

The pre-trained models (`best_model.joblib` and `tfidf_vectorizer.joblib`) are already serialized in the `model/` folder for inference and deployment.

If retraining models locally:
1. Place `True.csv` and `Fake.csv` inside this folder.
2. Run `python model/train_model.py`.
