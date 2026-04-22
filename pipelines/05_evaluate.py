#!/usr/bin/env python3
"""
Script 05: Model Evaluation & Visualizations
- Evaluates LR, RF, and GBT on test set
- Computes: Accuracy, AUC-ROC, Precision, Recall, F1
- Generates confusion matrix and ROC curve plots
- Saves evaluation metrics JSON and prediction samples
"""
import os, sys, json, time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve

from pyspark.sql import SparkSession
from src.io_utils import get_base_path, configure_spark_for_s3
from src.logger import get_logger
from pyspark.sql import functions as F
from pyspark.ml.classification import (
    LogisticRegressionModel,
    RandomForestClassificationModel,
    GBTClassificationModel,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

PROC_DIR = get_base_path("processed")
OUTPUT_DIR = get_base_path("output")
MODELS_DIR = os.path.join(get_base_path("output"), "models")

log = get_logger("05_evaluate")

def evaluate_model(name, model, test_df):
    log(f"\n--- Evaluating: {name} ---")
    preds = model.transform(test_df)

    binary_eval = BinaryClassificationEvaluator(labelCol="label",
                                                rawPredictionCol="rawPrediction",
                                                metricName="areaUnderROC")
    multi_eval  = MulticlassClassificationEvaluator(labelCol="label",
                                                    predictionCol="prediction")

    auc  = binary_eval.evaluate(preds)
    acc  = multi_eval.setMetricName("accuracy").evaluate(preds)
    f1   = multi_eval.setMetricName("f1").evaluate(preds)
    prec = multi_eval.setMetricName("weightedPrecision").evaluate(preds)
    rec  = multi_eval.setMetricName("weightedRecall").evaluate(preds)

    log(f"  AUC-ROC  : {auc:.4f}")
    log(f"  Accuracy : {acc:.4f}")
    log(f"  F1       : {f1:.4f}")
    log(f"  Precision: {prec:.4f}")
    log(f"  Recall   : {rec:.4f}")

    probs = preds.select("probability", "label").toPandas()
    y_true = probs["label"].astype(int)
    y_scores = probs["probability"].apply(lambda x: float(x[1]))
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[optimal_idx]
    
    log(f"  Optimum Threshold (Youdens J): {best_thresh:.4f}")

    # Confusion matrix
    cm_data = preds.groupBy("label", "prediction").count().collect()
    tn = tp = fp = fn = 0
    for row in cm_data:
        l, p, c = int(row["label"]), int(row["prediction"]), row["count"]
        if l == 0 and p == 0: tn = c
        elif l == 1 and p == 1: tp = c
        elif l == 0 and p == 1: fp = c
        elif l == 1 and p == 0: fn = c

    log(f"  Confusion Matrix:")
    log(f"    TN={tn:,}  FP={fp:,}")
    log(f"    FN={fn:,}  TP={tp:,}")

    metrics = {
        "auc_roc": round(auc, 4),
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }
    return metrics, preds

def plot_roc_comparison(all_metrics, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, metrics in all_metrics.items():
        if "fpr" in metrics and "tpr" in metrics:
            ax.plot(metrics["fpr"], metrics["tpr"], label=f"{name.replace('_', ' ').title()} (AUC = {metrics['auc_roc']:.4f})")
    ax.plot([0, 1], [0, 1], 'r--', label='Random Baseline')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved ROC comparison: {save_path}")
    
    # Prune fpr/tpr so JSON isn't massive
    for k in all_metrics:
        all_metrics[k].pop("fpr", None)
        all_metrics[k].pop("tpr", None)

def plot_cm(cm, title, save_path):
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["On-time", "Delayed"]); ax.set_yticklabels(["On-time", "Delayed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center",
                    color="white" if matrix[i,j] > matrix.max()/2 else "black", fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved confusion matrix: {save_path}")

def plot_comparison(all_metrics, save_path):
    """Bar chart comparing all models on key metrics."""
    models = list(all_metrics.keys())
    metrics_to_plot = ["auc_roc", "accuracy", "f1_score", "precision", "recall"]
    x = np.arange(len(metrics_to_plot))
    width = 0.22
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_name, metrics) in enumerate(all_metrics.items()):
        vals = [metrics[m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, vals, width, label=model_name.replace("_", " ").title(),
                      color=colors[i], alpha=0.87, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Flight Delay Prediction", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved comparison chart: {save_path}")

def main():
    log("=" * 70)
    log("STEP 05 — Model Evaluation")
    log("=" * 70)

    builder = SparkSession.builder \
        .appName("FlightDelay-Evaluate") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.enabled", "false") \
        
    spark = configure_spark_for_s3(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    test_path = os.path.join(PROC_DIR, "test_set.parquet")
    log(f"Loading test set from {test_path}...")
    test_df = spark.read.parquet(test_path).cache()
    log(f"Test rows: {test_df.count():,}")

    # ── Load models ───────────────────────────────────────────────────────
    lr_model  = LogisticRegressionModel.load(os.path.join(MODELS_DIR, "logistic_regression"))
    rf_model  = RandomForestClassificationModel.load(os.path.join(MODELS_DIR, "random_forest"))
    gbt_model = GBTClassificationModel.load(os.path.join(MODELS_DIR, "gbt"))

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_metrics = {}

    lr_metrics, lr_preds   = evaluate_model("Logistic Regression", lr_model, test_df)
    all_metrics["logistic_regression"] = lr_metrics
    plot_cm(lr_metrics["confusion_matrix"], "LR — Confusion Matrix",
            os.path.join(get_base_path("output"), "cm_logistic_regression.png"))

    rf_metrics, rf_preds   = evaluate_model("Random Forest", rf_model, test_df)
    all_metrics["random_forest"] = rf_metrics
    plot_cm(rf_metrics["confusion_matrix"], "Random Forest — Confusion Matrix",
            os.path.join(get_base_path("output"), "cm_random_forest.png"))

    gbt_metrics, gbt_preds = evaluate_model("GBT", gbt_model, test_df)
    all_metrics["gbt"] = gbt_metrics
    plot_cm(gbt_metrics["confusion_matrix"], "GBT — Confusion Matrix",
            os.path.join(get_base_path("output"), "cm_gbt.png"))

    # ── Comparison chart ──────────────────────────────────────────────────
    plot_comparison(all_metrics, os.path.join(get_base_path("output"), "model_comparison.png"))
    plot_roc_comparison(all_metrics, os.path.join(get_base_path("output"), "roc_comparison.png"))

    # ── Feature Importance (RF) ───────────────────────────────────────────
    fi = rf_model.featureImportances
    fi_list = [(i, float(fi[i])) for i in range(fi.size) if float(fi[i]) > 0]
    fi_list.sort(key=lambda x: -x[1])
    top_fi = fi_list[:15]
    log("\n--- Top 15 Feature Importances (Random Forest) ---")
    try:
        feat_info = json.load(open(os.path.join(get_base_path("output"), "03_feature_names.json")))
        feat_names = feat_info["all_features"]
    except:
        feat_names = [f"feature_{i}" for i in range(fi.size)]
    for idx, imp in top_fi:
        name = feat_names[idx] if idx < len(feat_names) else f"feature_{idx}"
        log(f"  {name:<35}: {imp:.4f}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    names  = [feat_names[i] if i < len(feat_names) else f"f{i}" for i, _ in top_fi]
    values = [v for _, v in top_fi]
    colors_fi = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(names[::-1], values[::-1], color=colors_fi[::-1], edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Top 15 Feature Importances — Random Forest", fontsize=13, fontweight="bold")
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(os.path.join(get_base_path("output"), "feature_importance_rf.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("  Feature importance chart saved.")

    # ── Sample Predictions ────────────────────────────────────────────────
    sample_preds = gbt_preds.select("label", "prediction", "probability") \
                            .limit(20) \
                            .toPandas()
    sample_preds["probability_delayed"] = sample_preds["probability"].apply(lambda x: round(float(x[1]), 4))
    sample_preds.drop(columns=["probability"], inplace=True)
    sample_preds["label"] = sample_preds["label"].astype(int)
    sample_preds["prediction"] = sample_preds["prediction"].astype(int)
    sample_preds.to_csv(os.path.join(get_base_path("output"), "sample_predictions.csv"), index=False)
    log(f"\nSample predictions saved.")

    # ── Save all metrics ──────────────────────────────────────────────────
    with open(os.path.join(get_base_path("output"), "05_all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    log(f"All metrics saved to output/05_all_metrics.json")

    test_df.unpersist()
    spark.stop()
    log("Step 05 complete.\n")

if __name__ == "__main__":
    main()
