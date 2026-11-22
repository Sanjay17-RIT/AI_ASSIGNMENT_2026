"""
Feature Engineering Engine
==========================
Synthesizes domain-engineered academic features and multimodal fusion metrics.
Course: Artificial Intelligence and Computational Systems (25 A D 35)
Candidate: Sai Sanjay A (1MS25AD053)

Engineered Features:
1. Academic Consistency Index (ACI)
2. Effort-to-Outcome Yield (EOY)
3. Assignment Diligence Rate (ADR)
4. Theory-to-Practical Alignment Ratio (TPAR)
5. Early Warning Composite Score (EWCS)
6. Multimodal Sentiment Polarity Feature (NLP Lexicon Fusion)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_academic_consistency_index(internal_marks: float, previous_gpa: float) -> float:
    """
    Evaluates whether current semester performance aligns with historical academic GPA.
    Formula: ACI = (internal_marks / 50.0) / max(0.1, previous_gpa / 10.0)
    """
    norm_internals = internal_marks / 50.0
    norm_gpa = max(0.1, previous_gpa / 10.0)
    return round(float(norm_internals / norm_gpa), 3)


def calculate_effort_outcome_yield(internal_marks: float, study_hours: float) -> float:
    """
    Measures study efficiency by comparing weekly time investment with internal marks achieved.
    Formula: EOY = internal_marks / max(1.0, study_hours)
    """
    safe_hours = max(1.0, study_hours)
    return round(float(internal_marks / safe_hours), 3)


def calculate_assignment_diligence_rate(assignments_completed: float, total_assignments: float = 10.0) -> float:
    """
    Formula: ADR = (assignments_completed / total_assignments) * 100
    """
    safe_total = max(1.0, total_assignments)
    return round(float((assignments_completed / safe_total) * 100.0), 2)


def calculate_theory_practical_alignment(internal_marks: float, lab_score: float) -> float:
    """
    Identifies disparities between theoretical test marks and hands-on laboratory assessments.
    Formula: TPAR = (internal_marks / 50.0) / max(0.1, lab_score / 30.0)
    """
    norm_theory = internal_marks / 50.0
    norm_lab = max(0.1, lab_score / 30.0)
    return round(float(norm_theory / norm_lab), 3)


def calculate_early_warning_composite_score(
    attendance: float,
    internal_marks: float,
    assignments_completed: float,
    previous_gpa: float
) -> float:
    """
    Normalized multi-factorial risk index from 0 to 100:
    High EWCS (>= 60) indicates severe intervention needed.
    """
    adr = (assignments_completed / 10.0) * 100.0
    norm_internals = (internal_marks / 50.0) * 100.0
    norm_gpa = (previous_gpa / 10.0) * 100.0
    
    academic_health = (
        0.35 * attendance +
        0.30 * norm_internals +
        0.15 * adr +
        0.20 * norm_gpa
    )
    risk_score = 100.0 - academic_health
    return round(float(np.clip(risk_score, 0.0, 100.0)), 2)


def extract_sentiment_polarity(feedback_text: str) -> Dict[str, Any]:
    """
    Lightweight rule-based lexical polarity score [-1.0, +1.0] for feature fusion.
    """
    if pd.isna(feedback_text) or not isinstance(feedback_text, str):
        return {'polarity': 0.0, 'label': 'Neutral'}
        
    text_lower = feedback_text.lower()
    
    pos_words = {
        'excellent', 'consistent', 'deep', 'understanding', 'outstanding',
        'exceptional', 'creativity', 'dedicated', 'thorough', 'good', 'solid',
        'potential', 'decent', 'hard', 'satisfactory', 'improvement', 'brilliant',
        'leadership', 'active', 'receptive', 'diligent', 'organized'
    }
    neg_words = {
        'struggles', 'poor', 'affecting', 'immediate', 'intervention', 'weak',
        'rarely', 'failing', 'minimal', 'counseling', 'below', 'lapses',
        'extremely', 'borderline', 'inconsistent', 'careless', 'irregular',
        'disruptive', 'procrastination', 'anxiety', 'unprepared'
    }
    
    words = text_lower.split()
    pos_count = sum(1 for w in words if w.strip('.,!?;:"()') in pos_words)
    neg_count = sum(1 for w in words if w.strip('.,!?;:"()') in neg_words)
    total = pos_count + neg_count
    
    if total == 0:
        polarity = 0.0
        label = 'Neutral'
    else:
        polarity = (pos_count - neg_count) / total
        if polarity > 0.15:
            label = 'Positive'
        elif polarity < -0.15:
            label = 'Negative'
        else:
            label = 'Constructive'
            
    return {'polarity': round(polarity, 3), 'label': label}


def engineer_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches student dataframe with all engineered academic features.
    Does not drop original columns; ensures 100% backward compatibility.
    """
    df_eng = df.copy()
    
    df_eng['academic_consistency_index'] = df_eng.apply(
        lambda r: calculate_academic_consistency_index(r['internal_marks'], r['previous_gpa']),
        axis=1
    )
    
    df_eng['effort_outcome_yield'] = df_eng.apply(
        lambda r: calculate_effort_outcome_yield(r['internal_marks'], r['study_hours_per_week']),
        axis=1
    )
    
    df_eng['assignment_diligence_rate'] = df_eng.apply(
        lambda r: calculate_assignment_diligence_rate(r['assignments_completed']),
        axis=1
    )
    
    df_eng['theory_practical_alignment'] = df_eng.apply(
        lambda r: calculate_theory_practical_alignment(r['internal_marks'], r['lab_score']),
        axis=1
    )
    
    df_eng['early_warning_composite_score'] = df_eng.apply(
        lambda r: calculate_early_warning_composite_score(
            r['attendance'], r['internal_marks'], r['assignments_completed'], r['previous_gpa']
        ),
        axis=1
    )
    
    # NLP Multimodal Fusion Feature
    if 'teacher_feedback' in df_eng.columns:
        sentiment_res = df_eng['teacher_feedback'].apply(extract_sentiment_polarity)
        df_eng['sentiment_polarity'] = sentiment_res.apply(lambda x: x['polarity'])
        df_eng['sentiment_label'] = sentiment_res.apply(lambda x: x['label'])
    else:
        df_eng['sentiment_polarity'] = 0.0
        df_eng['sentiment_label'] = 'Neutral'
        
    return df_eng
