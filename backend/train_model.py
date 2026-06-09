"""
XGBoost Model Training Pipeline
Trains a binary classifier to predict chasing team win probability.
Saves the model as a .pkl file and prints top feature importances.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

from generate_dataset import generate_dataset
from feature_engineering import engineer_features, FEATURE_COLUMNS, TARGET_COLUMN


# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "model", "xgb_t20_model.pkl")


def get_top_feature_importances(model: XGBClassifier, feature_names: list, top_n: int = 3) -> list:
    """
    Extract the top-N most important features from the trained model.

    Returns:
        List of dicts: [{"feature": str, "importance": float}, ...]
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    return [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in indices
    ]


def train_model():
    """Full training pipeline: generate data → engineer features → train → save."""

    # ── Step 1: Generate or load dataset ──
    csv_path = os.path.join(DATA_DIR, "t20_matches.csv")

    if os.path.exists(csv_path):
        print("[*] Loading existing dataset...")
        df = pd.read_csv(csv_path)
    else:
        print("[*] Generating fresh dataset...")
        os.makedirs(DATA_DIR, exist_ok=True)
        df = generate_dataset(n_matches=2000)
        df.to_csv(csv_path, index=False)

    print(f"   -> {len(df)} matches loaded")

    # ── Step 2: Feature engineering ──
    print("\n[*] Engineering features...")
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print(f"   -> Feature matrix shape: {X.shape}")
    print(f"   -> Target distribution: {y.value_counts().to_dict()}")

    # ── Step 3: Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Step 4: Train XGBoost ──
    print("\n[*] Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Step 5: Evaluate ──
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n[+] Model Performance:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   ROC-AUC:  {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Batting First Won', 'Chasing Team Won'])}")

    # ── Step 6: Feature importances ──
    top_features = get_top_feature_importances(model, FEATURE_COLUMNS, top_n=3)
    print("[+] Top 3 Feature Importances:")
    for i, feat in enumerate(top_features, 1):
        print(f"   {i}. {feat['feature']}: {feat['importance']:.4f}")

    # ── Step 7: Save model ──
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[+] Model saved -> {MODEL_PATH}")

    return model, top_features


if __name__ == "__main__":
    train_model()
