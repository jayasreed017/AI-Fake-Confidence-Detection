"""
backend.py — Feature extraction, prediction, and explanation logic
for AI Fake Confidence Detection.
"""

import re
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

# ──────────────────────────────────────────────
# Word banks (expanded)
# ──────────────────────────────────────────────

HESITATION_WORDS = [
    "um", "uh", "hmm", "er", "ah",
    "maybe", "perhaps", "possibly", "probably",
    "i think", "i guess", "i suppose", "i believe",
    "not sure", "not certain", "not totally sure",
    "sort of", "kind of", "somewhat", "fairly",
    "i hope", "i feel like", "could be", "might be",
    "hard to say", "not fully", "half remember",
]

EXAGGERATION_WORDS = [
    "definitely", "absolutely", "certainly", "always",
    "never", "every time", "totally", "completely",
    "perfectly", "without a doubt", "100 percent",
    "unquestionably", "literally", "guaranteed",
    "best ever", "best solution", "no way", "impossible",
    "100%", "ninety nine percent", "always right",
    "never wrong", "never fail", "no one knows",
]

GENUINE_MARKERS = [
    "i practiced", "i studied", "i prepared",
    "i verified", "i analyzed", "i reviewed",
    "i worked through", "i tested", "i confirmed",
    "i derived", "from experience", "step by step",
    "i recall", "i have used", "based on",
    "i have solved", "i have experience",
    "i can demonstrate", "let me explain",
    "i understand", "i know because",
]

# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────

def load_model():
    return joblib.load("models/confidence_model.pkl")

MODEL = load_model()

# ──────────────────────────────────────────────
# Custom feature extraction
# ──────────────────────────────────────────────

def compute_custom_features(text: str) -> list:
    """
    Returns a list of hand-crafted numeric features.
    """
    t = text.lower()
    words = t.split()
    sentences = re.split(r'[.!?]', t)
    sentences = [s.strip() for s in sentences if s.strip()]

    hesitation_count  = sum(1 for w in HESITATION_WORDS  if w in t)
    exaggeration_count = sum(1 for w in EXAGGERATION_WORDS if w in t)
    genuine_count      = sum(1 for w in GENUINE_MARKERS   if w in t)

    word_count      = len(words)
    avg_word_len    = np.mean([len(w) for w in words]) if words else 0
    sentence_count  = max(len(sentences), 1)
    avg_sent_len    = word_count / sentence_count

    # Uncertainty phrases ratio
    uncertainty_ratio = hesitation_count / max(word_count, 1)
    # Exaggeration density
    exag_ratio = exaggeration_count / max(word_count, 1)
    # Genuine signal ratio
    genuine_ratio = genuine_count / max(word_count, 1)

    return [
        hesitation_count,
        exaggeration_count,
        genuine_count,
        word_count,
        avg_word_len,
        avg_sent_len,
        uncertainty_ratio,
        exag_ratio,
        genuine_ratio,
    ]

# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def extract_features(text: str):
    """
    Predict fake / genuine confidence.
    Returns (prediction_label, confidence_score_percent).
    """
    prediction = MODEL.predict([text])[0]

    try:
        prob = MODEL.predict_proba([text])[0]
        confidence_score = round(max(prob) * 100, 1)
    except Exception:
        confidence_score = 75.0

    # ── Heuristic override ──
    # If custom features show strong signals, we can nudge the score.
    feats = compute_custom_features(text)
    hesitation_count, exaggeration_count, genuine_count = feats[0], feats[1], feats[2]

    if prediction == "genuine" and (hesitation_count >= 3 or exaggeration_count >= 3):
        prediction = "fake"
        confidence_score = min(confidence_score + 10, 99.0)

    if prediction == "fake" and genuine_count >= 3 and hesitation_count == 0 and exaggeration_count == 0:
        prediction = "genuine"
        confidence_score = min(confidence_score + 10, 99.0)

    return prediction, confidence_score

# ──────────────────────────────────────────────
# Explanation
# ──────────────────────────────────────────────

def explain_text(text: str) -> dict:
    """
    Returns a rich explanation dictionary with:
      - hesitation_words   : list of found hesitation words
      - exaggeration_words : list of found exaggeration words
      - genuine_markers    : list of found genuine markers
      - reasoning          : human-readable string
      - signal_scores      : dict with numeric scores
    """
    t = text.lower()

    found_hesitation    = [w for w in HESITATION_WORDS    if w in t]
    found_exaggeration  = [w for w in EXAGGERATION_WORDS  if w in t]
    found_genuine       = [w for w in GENUINE_MARKERS     if w in t]

    feats = compute_custom_features(text)
    word_count = feats[3]

    # Build reasoning narrative
    reasons = []

    if found_hesitation:
        reasons.append(
            f"Uncertainty language detected ({len(found_hesitation)} instance(s)): "
            f"*{', '.join(found_hesitation[:5])}*."
        )

    if found_exaggeration:
        reasons.append(
            f"Over-confident or exaggerated phrasing found ({len(found_exaggeration)} instance(s)): "
            f"*{', '.join(found_exaggeration[:5])}*."
        )

    if found_genuine:
        reasons.append(
            f"Genuine preparation signals present ({len(found_genuine)} instance(s)): "
            f"*{', '.join(found_genuine[:5])}*."
        )

    if word_count < 5:
        reasons.append("Very short input — confidence in prediction is lower.")

    if not reasons:
        reasons.append("No strong linguistic markers detected. Prediction is based on overall sentence structure and vocabulary patterns.")

    reasoning = " ".join(reasons)

    signal_scores = {
        "Hesitation Score":   min(len(found_hesitation)  * 20, 100),
        "Exaggeration Score": min(len(found_exaggeration) * 20, 100),
        "Genuine Signal":     min(len(found_genuine)      * 20, 100),
    }

    return {
        "hesitation_words":   found_hesitation,
        "exaggeration_words": found_exaggeration,
        "genuine_markers":    found_genuine,
        "reasoning":          reasoning,
        "signal_scores":      signal_scores,
    }

# ──────────────────────────────────────────────
# Sample inputs (used by app.py)
# ──────────────────────────────────────────────

SAMPLE_INPUTS = {
    "✅ Genuine (Prepared)":      "I practiced this concept multiple times and I clearly understand the solution. I can explain each step in detail.",
    "❌ Fake (Hesitant)":          "Um I think maybe this is correct, uh not really sure but I will try and hope it works.",
    "❌ Fake (Over-confident)":    "I am absolutely 100 percent sure this is always the best solution and I never make mistakes here.",
    "🔍 Ambiguous":                "I sort of know the answer and I believe it might be correct, probably.",
}
