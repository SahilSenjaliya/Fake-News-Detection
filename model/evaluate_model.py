"""
Model Evaluation Module for Fake News Detection.

Provides reusable metrics calculation, comparison tables generation,
and confusion matrix visualization utilities for scikit-learn models.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes accuracy, precision, recall, f1-score, and confusion matrix.

    Args:
        y_true: Ground truth binary labels (0 = Fake, 1 = Real).
        y_pred: Predicted binary labels.
        model_name: Name identifier for the model.

    Returns:
        Dictionary containing all evaluation metrics and the confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Confusion_Matrix": cm
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    output_path: str,
    model_name: str = "Selected Model",
    labels: Tuple[str, str] = ("Fake (0)", "Real (1)")
) -> None:
    """
    Generates and saves a styled Confusion Matrix heatmap.

    Args:
        cm: 2x2 confusion matrix array.
        output_path: Path where the PNG image will be saved.
        model_name: Model name to show in the plot title.
        labels: Tuple of class display names.
    """
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=15, weight='bold')
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved successfully to: {output_path}")


def print_detailed_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> None:
    """
    Prints a detailed text classification report and confusion matrix to console.
    """
    print(f"\n{'=' * 50}")
    print(f"Detailed Classification Report: {model_name}")
    print(f"{'=' * 50}")
    print(classification_report(y_true, y_pred, target_names=["Fake (0)", "Real (1)"], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"{'=' * 50}\n")
