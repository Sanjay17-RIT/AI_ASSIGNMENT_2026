"""
Data Cleaning & Sanitization Engine
====================================
Production-grade data cleaning, validation, and outlier management.
Course: Artificial Intelligence and Computational Systems (25 A D 35)
Candidate: Sai Sanjay A (1MS25AD053)

Features:
- Academic boundary validation (Attendance, Internals, GPA, Lab)
- Statistical outlier detection & Winsorization via Interquartile Range (IQR)
- Missing value imputation using cohort medians
- Teacher qualitative feedback text sanitization
"""

import re
import unicodedata
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np


# Official VTU / Academic Range Constraints
ACADEMIC_CONSTRAINTS = {
    'attendance': (0.0, 100.0),
    'internal_marks': (0.0, 50.0),
    'assignments_completed': (0, 10),
    'study_hours_per_week': (0.0, 60.0),
    'previous_gpa': (0.0, 10.0),
    'lab_score': (0.0, 30.0),
}


def sanitize_text(text: str) -> str:
    """
    Sanitize unstructured teacher feedback text for NLP pipelines.
    - Normalizes unicode characters (NFKD)
    - Strips HTML tags or malformed escape sequences
    - Replaces multiple whitespace with a single space
    """
    if pd.isna(text) or not isinstance(text, str):
        return "Student exhibits regular attendance and course participation."
    
    # Unicode normalize
    text = unicodedata.normalize('NFKD', text)
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Strip excessive non-alphanumeric punctuation while keeping basic sentence stops
    text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_and_cap_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str] = None,
    factor: float = 1.5
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Detect and cap continuous numeric outliers using Interquartile Range (IQR).
    Applies Winsorization (capping at lower and upper whiskers).
    
    Returns:
        Tuple of (clean_df, audit_log)
    """
    cleaned_df = df.copy()
    audit_log: List[Dict[str, Any]] = []
    
    if columns is None:
        columns = ['study_hours_per_week', 'previous_gpa', 'attendance', 'internal_marks']
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        
        # Determine natural bounds from constraints if available
        min_allowed, max_allowed = ACADEMIC_CONSTRAINTS.get(col, (-np.inf, np.inf))
        
        lower_bound = max(min_allowed, q1 - factor * iqr)
        upper_bound = min(max_allowed, q3 + factor * iqr)
        
        # Detect outliers
        outliers_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
        outlier_indices = cleaned_df[outliers_mask].index.tolist()
        
        for idx in outlier_indices:
            original_val = float(cleaned_df.loc[idx, col])
            capped_val = float(np.clip(original_val, lower_bound, upper_bound))
            audit_log.append({
                'row_index': idx,
                'student_id': int(cleaned_df.loc[idx, 'student_id']) if 'student_id' in cleaned_df.columns else idx,
                'student_name': cleaned_df.loc[idx, 'name'] if 'name' in cleaned_df.columns else f"Student #{idx}",
                'column': col,
                'original_value': original_val,
                'capped_value': capped_val,
                'lower_whisker': round(lower_bound, 2),
                'upper_whisker': round(upper_bound, 2),
                'reason': 'IQR Outlier Capping (Winsorization)'
            })
            cleaned_df.loc[idx, col] = capped_val
            
    return cleaned_df, audit_log


def validate_academic_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Enforces strict physical/academic domain limits on all records.
    """
    validated_df = df.copy()
    violations: List[Dict[str, Any]] = []
    
    for col, (min_val, max_val) in ACADEMIC_CONSTRAINTS.items():
        if col not in validated_df.columns:
            continue
            
        out_of_bounds = (validated_df[col] < min_val) | (validated_df[col] > max_val)
        indices = validated_df[out_of_bounds].index.tolist()
        
        for idx in indices:
            raw_val = float(validated_df.loc[idx, col])
            fixed_val = float(np.clip(raw_val, min_val, max_val))
            violations.append({
                'row_index': idx,
                'column': col,
                'invalid_value': raw_val,
                'clamped_value': fixed_val,
                'bounds': (min_val, max_val)
            })
            validated_df.loc[idx, col] = fixed_val
            
    return validated_df, violations


def clean_student_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete end-to-end data cleaning pipeline:
    1. Duplicate removal
    2. Missing value imputation via median/mode
    3. Academic domain range enforcement
    4. Statistical IQR outlier detection & Winsorization
    5. Qualitative text sanitization
    """
    clean_df = df.copy()
    report = {
        'initial_rows': len(clean_df),
        'duplicates_removed': 0,
        'missing_values_imputed': {},
        'range_violations_fixed': 0,
        'iqr_outliers_capped': 0,
        'audit_log': []
    }
    
    # 1. Deduplication
    if 'student_id' in clean_df.columns:
        dups = clean_df.duplicated(subset=['student_id']).sum()
        if dups > 0:
            clean_df = clean_df.drop_duplicates(subset=['student_id'], keep='first').reset_index(drop=True)
            report['duplicates_removed'] = int(dups)
            
    # 2. Missing Value Imputation
    numeric_cols = [c for c in ACADEMIC_CONSTRAINTS.keys() if c in clean_df.columns]
    for col in numeric_cols:
        null_count = clean_df[col].isnull().sum()
        if null_count > 0:
            median_val = clean_df[col].median()
            clean_df[col] = clean_df[col].fillna(median_val)
            report['missing_values_imputed'][col] = int(null_count)
            
    if 'teacher_feedback' in clean_df.columns:
        null_feedback = clean_df['teacher_feedback'].isnull().sum()
        if null_feedback > 0:
            clean_df['teacher_feedback'] = clean_df['teacher_feedback'].fillna(
                "Student exhibits regular attendance and coursework participation."
            )
            report['missing_values_imputed']['teacher_feedback'] = int(null_feedback)
            
    # 3. Academic Range Validation
    clean_df, range_violations = validate_academic_ranges(clean_df)
    report['range_violations_fixed'] = len(range_violations)
    report['audit_log'].extend(range_violations)
    
    # 4. Outlier Capping (IQR)
    clean_df, outlier_audits = detect_and_cap_outliers_iqr(clean_df)
    report['iqr_outliers_capped'] = len(outlier_audits)
    report['audit_log'].extend(outlier_audits)
    
    # 5. Text Sanitization
    if 'teacher_feedback' in clean_df.columns:
        clean_df['teacher_feedback'] = clean_df['teacher_feedback'].apply(sanitize_text)
        
    report['final_rows'] = len(clean_df)
    report['cleaned_dataframe'] = clean_df
    return report
