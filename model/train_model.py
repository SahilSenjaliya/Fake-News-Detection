"""
Machine Learning Training Pipeline for Fake News Detection.

This script executes the complete ML training lifecycle:
1. Loads raw datasets (True.csv and Fake.csv) without modifying original files.
2. Assigns target labels (Fake=0, True=1).
3. Combines datasets into a unified DataFrame.
4. Removes duplicate records before splitting to prevent data leakage.
5. Merges 'title' and 'text' into a single textual feature (excluding 'subject' and 'date').
6. Applies preprocessing/cleaning (lowercase, strip URLs, HTML, punctuation, normalize whitespace).
7. Splits data into 80% train and 20% test sets (stratified, random_state=42).
8. Extracts features using TF-IDF (1-2 ngrams, english stop words, max 100,000 features, sublinear tf).
9. Trains and benchmarks 3 classifiers:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Linear Support Vector Machine (LinearSVC)
10. Evaluates models on Accuracy, Precision, Recall, and F1-score.
11. Saves model comparison table to 'model/model_comparison.csv'.
12. Selects the best performing model based on test-set F1-score.
13. Persists the best model and vectorizer with joblib for FastAPI integration.
14. Saves the confusion matrix heatmap to 'model/confusion_matrix.png'.
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path for direct imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Ensure stdout handles UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from model.preprocessing import clean_text
from model.evaluate_model import calculate_metrics, save_confusion_matrix_plot, print_detailed_report


def run_pipeline():
    start_total_time = time.time()
    print("=" * 60)
    print(">>> Starting Fake News Detection ML Pipeline")
    print("=" * 60)

    # 1. Define file paths
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    model_dir = os.path.join(BASE_DIR, "model")
    true_path = os.path.join(dataset_dir, "True.csv")
    fake_path = os.path.join(dataset_dir, "Fake.csv")

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(f"Missing dataset files in {dataset_dir}")

    # 2. Load datasets without modifying originals
    print("\n[Step 1/8] Loading raw datasets...")
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    print(f"  - Loaded True.csv: {len(df_true):,} records")
    print(f"  - Loaded Fake.csv: {len(df_fake):,} records")

    # 3. Assign labels (Fake = 0, True/Real = 1)
    df_fake["label"] = 0
    df_true["label"] = 1

    # 4. Combine both datasets into one DataFrame
    print("\n[Step 2/8] Combining datasets...")
    df = pd.concat([df_fake, df_true], ignore_index=True)
    raw_count = len(df)
    print(f"  - Combined total records: {raw_count:,}")

    # 5. Remove duplicate articles BEFORE train/test splitting
    print("\n[Step 3/8] Removing duplicates...")
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_count = len(df)
    print(f"  - Duplicate rows removed: {duplicates_count:,}")
    print(f"  - Remaining records after deduplication: {dedup_count:,}")

    # 6. Feature Engineering: Combine title + text (ignore subject and date to avoid leakage)
    print("\n[Step 4/8] Merging 'title' and 'text' and applying NLP text cleaning...")
    # Safe combination handling any possible nulls
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Clean the combined text using the reusable preprocessing function
    t0_clean = time.time()
    df["cleaned_content"] = df["content"].apply(clean_text)
    t1_clean = time.time()
    print(f"  - Preprocessing completed in {t1_clean - t0_clean:.2f} seconds.")

    # 7. Stratified Train / Test Split (80% Train, 20% Test)
    print("\n[Step 5/8] Splitting dataset (80% Train, 20% Test, stratify=y, random_state=42)...")
    X = df["cleaned_content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"  - Training set size: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  - Testing set size : {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
    print(f"  - Train label distribution: {dict(y_train.value_counts())}")
    print(f"  - Test label distribution : {dict(y_test.value_counts())}")

    # 8. TF-IDF Feature Extraction
    print("\n[Step 6/8] Extracting TF-IDF features...")
    print("  - Configuration: ngram_range=(1,2), stop_words='english', max_features=100000, sublinear_tf=True")
    t0_tfidf = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=100000,
        sublinear_tf=True
    )

    # Fit on training data only to prevent data leakage
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    t1_tfidf = time.time()
    print(f"  - TF-IDF extraction completed in {t1_tfidf - t0_tfidf:.2f} seconds.")
    print(f"  - X_train feature matrix shape: {X_train_vec.shape}")
    print(f"  - X_test feature matrix shape : {X_test_vec.shape}")

    # 9. Train and Evaluate Models
    print("\n[Step 7/8] Training and evaluating candidate classification models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear Support Vector Machine": LinearSVC(random_state=42, max_iter=2000, dual="auto")
    }

    results = []
    trained_models = {}
    predictions = {}

    for name, clf in models.items():
        print(f"\n  [-] Training {name}...")
        t0_train = time.time()
        clf.fit(X_train_vec, y_train)
        t1_train = time.time()

        y_pred = clf.predict(X_test_vec)
        eval_metrics = calculate_metrics(y_test, y_pred, model_name=name)

        results.append({
            "Model": name,
            "Accuracy": eval_metrics["Accuracy"],
            "Precision": eval_metrics["Precision"],
            "Recall": eval_metrics["Recall"],
            "F1-score": eval_metrics["F1-score"],
            "Training Time (s)": round(t1_train - t0_train, 2)
        })

        trained_models[name] = clf
        predictions[name] = (y_pred, eval_metrics["Confusion_Matrix"])
        print_detailed_report(y_test, y_pred, model_name=name)

    # 10. Model Comparison Table sorted by F1-score
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(by="F1-score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE (Sorted by F1-score)")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # Save comparison table
    comparison_csv_path = os.path.join(model_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nSaved model comparison table to: {comparison_csv_path}")

    # 11. Select and persist the best model
    best_row = comparison_df.iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    best_f1 = best_row["F1-score"]
    best_acc = best_row["Accuracy"]
    best_cm = predictions[best_model_name][1]

    print("\n" + "=" * 60)
    print(f"[BEST MODEL SELECTED]: {best_model_name}")
    print(f"   - F1-Score: {best_f1:.4f} ({best_f1*100:.2f}%)")
    print(f"   - Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("=" * 60)

    # Save artifacts using joblib
    print("\n[Step 8/8] Saving trained artifacts...")
    best_model_path = os.path.join(model_dir, "best_model.joblib")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    cm_plot_path = os.path.join(model_dir, "confusion_matrix.png")

    joblib.dump(best_model, best_model_path)
    print(f"  - Saved model to: {best_model_path}")

    joblib.dump(vectorizer, vectorizer_path)
    print(f"  - Saved TF-IDF vectorizer to: {vectorizer_path}")

    # Save confusion matrix plot
    save_confusion_matrix_plot(best_cm, cm_plot_path, model_name=best_model_name)

    total_pipeline_time = time.time() - start_total_time
    print(f"\n[SUCCESS] Pipeline successfully finished in {total_pipeline_time:.2f} seconds.")

    return {
        "final_records": dedup_count,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features_dim": X_train_vec.shape[1],
        "comparison_df": comparison_df,
        "best_model_name": best_model_name,
        "best_row": best_row
    }


if __name__ == "__main__":
    run_pipeline()
