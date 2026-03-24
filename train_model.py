"""
train_model.py — Train and save the Fake Confidence Detection model.

Run this script once (or whenever dataset.csv is updated) to
rebuild the model file at models/confidence_model.pkl.

Usage:
    python train_model.py
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ──────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv("dataset.csv")

# Strip whitespace from all string columns
df["text"]  = df["text"].str.strip()
df["label"] = df["label"].str.strip()

# Drop empty rows
df.dropna(subset=["text", "label"], inplace=True)
df = df[df["text"] != ""]

print(f"  Total samples : {len(df)}")
print(f"  Label counts  :\n{df['label'].value_counts().to_string()}\n")

X = df["text"]
y = df["label"]

# ──────────────────────────────────────────────
# 2. Train / test split
# ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ──────────────────────────────────────────────
# 3. Build pipeline
# ──────────────────────────────────────────────

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 3),          # unigrams, bigrams, trigrams
            max_features=5000,
            sublinear_tf=True,           # dampen very common terms
            min_df=1,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            C=1.0,                       # regularisation strength
            solver="lbfgs",
            class_weight="balanced",     # handles slight class imbalance
        ),
    ),
])

# ──────────────────────────────────────────────
# 4. Train
# ──────────────────────────────────────────────

print("Training model...")
model.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 5. Evaluate
# ──────────────────────────────────────────────

y_pred = model.predict(X_test)

print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation for a more robust estimate
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\n5-Fold CV Accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# ──────────────────────────────────────────────
# 6. Save model
# ──────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/confidence_model.pkl")
print("\nModel saved to models/confidence_model.pkl ✔")
