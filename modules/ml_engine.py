"""
Module 1: ML Prediction Engine
================================
Implements Decision Tree and Naive Bayes classifiers for student performance prediction.

Syllabus Concepts Covered (Unit III):
  - Decision Tree Classification: Illustrative Example
  - Naive Bayes Models
  - Bayes' Rule and its Use
  - Quantifying Uncertainty: Acting under Uncertainty
  - Basic Probability Notations
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==============================================================
# DECISION TREE CLASSIFIER
# Syllabus: Decision Tree Classification: Illustrative Example
# Uses entropy (information gain) as the split criterion
# ==============================================================

def train_decision_tree(X: np.ndarray, y: np.ndarray, max_depth: int = 4) -> DecisionTreeClassifier:
    """
    Train a Decision Tree with controlled depth to avoid overfitting.
    
    Uses ENTROPY as the criterion — this directly implements the
    Information Gain concept from the syllabus.
    
    Algorithm:
        1. Compute entropy of full dataset: H(S) = -Σ p_i * log2(p_i)
        2. For each feature, compute information gain:
           IG(S, feature) = H(S) - Σ (|Sv|/|S|) * H(Sv)
        3. Split on feature with highest IG
        4. Recurse on each branch until max_depth or pure leaf
    
    Syllabus: Decision Tree Classification (Unit III)
    
    Args:
        X: Feature matrix (n_samples x n_features)
        y: Target labels (encoded as integers)
        max_depth: Maximum tree depth (controls overfitting)
    
    Returns:
        Trained DecisionTreeClassifier model
    """
    model = DecisionTreeClassifier(
        criterion='entropy',      # Information Gain — matches syllabus
        max_depth=max_depth,
        random_state=42,
        min_samples_split=3,
        min_samples_leaf=2
    )
    model.fit(X, y)
    return model


# ==============================================================
# NAIVE BAYES CLASSIFIER
# Syllabus: Naive Bayes Models, Bayes' Rule and its Use
# P(class | features) ∝ P(class) × Π P(feature_i | class)
# ==============================================================

def train_naive_bayes(X: np.ndarray, y: np.ndarray) -> GaussianNB:
    """
    Train a Gaussian Naive Bayes classifier.
    
    Algorithm (Bayes' Rule):
        For each class c:
          P(c | x1, x2, ..., xn) ∝ P(c) × Π P(xi | c)
        
        Where:
          P(c)     = prior probability = count(class c) / total samples
          P(xi | c) = likelihood = Gaussian(xi; mean_i_c, var_i_c)
        
        Prediction = argmax_c P(c | features)
    
    The 'naive' assumption: all features are conditionally independent
    given the class. This simplifies computation enormously.
    
    Syllabus: Naive Bayes Models, Bayes' Rule and its Use (Unit III)
    
    Args:
        X: Feature matrix
        y: Target labels
    
    Returns:
        Trained GaussianNB model
    """
    model = GaussianNB()
    model.fit(X, y)
    return model


def predict_with_probabilities(model, features: list, class_names: list) -> dict:
    """
    Predict class and return probability for each class.
    
    This demonstrates QUANTIFYING UNCERTAINTY — instead of just giving
    a single answer, we show how confident the model is.
    
    Syllabus: Quantifying Uncertainty, Acting under Uncertainty (Unit III)
    
    Args:
        model: Trained classifier (DT or NB)
        features: List of feature values for one student
        class_names: List of class names ['Fail', 'High', 'Pass']
    
    Returns:
        dict with 'prediction', 'probabilities', 'confidence'
    """
    features_array = np.array(features).reshape(1, -1)
    prediction_idx = model.predict(features_array)[0]
    prediction = class_names[prediction_idx]
    
    # Get probability distribution over all classes
    proba = model.predict_proba(features_array)[0]
    probabilities = {class_names[i]: float(proba[i]) for i in range(len(class_names))}
    
    # Confidence = probability of the predicted class
    confidence = float(proba[prediction_idx])
    
    return {
        'prediction': prediction,
        'probabilities': probabilities,
        'confidence': confidence
    }


def get_decision_path(tree_model: DecisionTreeClassifier, feature_names: list, 
                       features: list, class_names: list) -> list:
    """
    Trace which nodes/features the Decision Tree used to reach its decision.
    
    This is crucial for EXPLAINABILITY — showing the AI's reasoning process.
    
    Syllabus: Decision Tree Classification — Illustrative Example (Unit III)
    
    Args:
        tree_model: Trained Decision Tree
        feature_names: List of feature names
        features: Student's feature values
        class_names: Class name labels
    
    Returns:
        List of step dicts: [{'feature': 'attendance', 'threshold': 75.0, 
                              'value': 65, 'direction': 'YES/NO'}, ...]
    """
    features_array = np.array(features).reshape(1, -1)
    
    # Get the decision path (which nodes were visited)
    node_indicator = tree_model.decision_path(features_array)
    node_indices = node_indicator.indices
    
    tree = tree_model.tree_
    path_steps = []
    
    for node_id in node_indices:
        # If it's a leaf node
        if tree.feature[node_id] < 0:
            predicted_class_idx = np.argmax(tree.value[node_id])
            predicted_class = class_names[predicted_class_idx]
            samples_in_leaf = tree.value[node_id][0]
            path_steps.append({
                'type': 'leaf',
                'prediction': predicted_class,
                'samples': {class_names[i]: int(samples_in_leaf[i]) for i in range(len(class_names))},
                'description': f"→ Prediction: **{predicted_class}** ✅"
            })
        else:
            # It's a decision node
            feat_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feat_name = feature_names[feat_idx]
            feat_value = features[feat_idx]
            goes_left = feat_value <= threshold
            direction = "YES" if goes_left else "NO"
            
            path_steps.append({
                'type': 'decision',
                'feature': feat_name,
                'threshold': round(threshold, 2),
                'value': feat_value,
                'direction': direction,
                'description': f"{feat_name} ≤ {threshold:.1f}? → **{direction}** (actual: {feat_value})"
            })
    
    return path_steps


def get_tree_text(tree_model: DecisionTreeClassifier, feature_names: list) -> str:
    """
    Get a text representation of the full Decision Tree structure.
    
    Syllabus: Decision Tree Classification (Unit III)
    
    Args:
        tree_model: Trained Decision Tree
        feature_names: Feature names
    
    Returns:
        String representation of the tree
    """
    return export_text(tree_model, feature_names=feature_names, show_weights=True)


def get_feature_importances(tree_model: DecisionTreeClassifier, feature_names: list) -> dict:
    """
    Get feature importance scores from the Decision Tree.
    Importance = total reduction in entropy contributed by each feature.
    
    Syllabus: Decision Tree — Information Gain (Unit III)
    
    Args:
        tree_model: Trained Decision Tree
        feature_names: Feature names
    
    Returns:
        dict of feature_name -> importance score
    """
    importances = tree_model.feature_importances_
    return {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}


def compare_models(dt_result: dict, nb_result: dict) -> dict:
    """
    Compare predictions from Decision Tree and Naive Bayes.
    If they disagree, explain WHY based on algorithmic differences.
    
    This demonstrates deep understanding of how different ML algorithms
    can reach different conclusions on the same data.
    
    Args:
        dt_result: Decision Tree prediction result
        nb_result: Naive Bayes prediction result
    
    Returns:
        dict with 'agree', 'explanation', 'dt_prediction', 'nb_prediction'
    """
    dt_pred = dt_result['prediction']
    nb_pred = nb_result['prediction']
    agree = dt_pred == nb_pred
    
    if agree:
        explanation = (
            f"✅ **Both models agree: {dt_pred}**\n\n"
            f"The Decision Tree reached this by splitting on specific feature thresholds, "
            f"while Naive Bayes computed P({dt_pred}|features) = {nb_result['confidence']:.1%} "
            f"using Bayes' Rule. Agreement between fundamentally different algorithms "
            f"increases our confidence in the prediction."
        )
    else:
        dt_conf = dt_result['confidence']
        nb_conf = nb_result['confidence']
        explanation = (
            f"⚠️ **Models disagree!**\n\n"
            f"- **Decision Tree** predicts **{dt_pred}** (confidence: {dt_conf:.1%})\n"
            f"- **Naive Bayes** predicts **{nb_pred}** (confidence: {nb_conf:.1%})\n\n"
            f"**Why the disagreement?**\n"
            f"- The Decision Tree makes hard splits on individual features. "
            f"If a single key feature (like attendance) crosses a threshold, "
            f"the prediction can flip entirely.\n"
            f"- Naive Bayes considers ALL features simultaneously using probability. "
            f"It can be swayed by the cumulative effect of several slightly unfavorable features.\n\n"
            f"**Recommendation:** Trust the model with higher confidence, "
            f"or consider the student as a 'borderline' case needing attention."
        )
    
    return {
        'agree': agree,
        'dt_prediction': dt_pred,
        'nb_prediction': nb_pred,
        'dt_confidence': dt_result['confidence'],
        'nb_confidence': nb_result['confidence'],
        'explanation': explanation
    }


def evaluate_model(model, X: np.ndarray, y: np.ndarray, model_name: str) -> dict:
    """
    Evaluate a model using cross-validation.
    
    Syllabus: Machine Learning Basics (Unit III)
    
    Args:
        model: Trained sklearn model
        X: Feature matrix
        y: Target labels
        model_name: Name for display
    
    Returns:
        dict with 'accuracy', 'cv_scores', 'mean_cv'
    """
    # Overall accuracy
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    # Cross-validation (3-fold for small dataset)
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    
    return {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'cv_scores': cv_scores.tolist(),
        'mean_cv': float(cv_scores.mean()),
        'std_cv': float(cv_scores.std())
    }
