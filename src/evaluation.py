import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve, average_precision_score

)
import numpy as np

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nROC-AUC Score:")
    print(roc_auc_score(y_test, y_proba))

def plot_roc(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()



def plot_precision_recall(model, X_test, y_test):
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP = {avg_precision:.4f})")
    plt.show()

    
# Automatic threshold selection based on F1 score

def find_best_threshold(model, X_test, y_test):
    
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    
    print("Best Threshold:", round(best_threshold, 4))
    print("Best F1 Score:", round(f1_scores[best_index], 4))
    
    return best_threshold

# Business Cost Evaluation
def business_impact(model, X_test, y_test, threshold):
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    cost_fn = 1000   # cost of missing fraud
    cost_fp = 50     # cost of false alert
    
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    
    print("False Negatives (Missed Fraud):", fn)
    print("False Positives (False Alerts):", fp)
    print("Estimated Total Cost:", total_cost)