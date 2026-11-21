"""
Data Loader Utility
====================
Loads and preprocesses the student dataset for ML models.
Syllabus: Data preprocessing for Machine Learning (Unit III)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


def load_student_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load the student dataset from CSV.
    
    Args:
        csv_path: Path to CSV file. If None, uses default path.
    
    Returns:
        pandas DataFrame with all student data.
    """
    if csv_path is None:
        # Default path relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "students.csv")
    
    df = pd.read_csv(csv_path)
    return df


def prepare_features_and_labels(df: pd.DataFrame) -> dict:
    """
    Separate features (X) and labels (y) from the dataset.
    Encode the target variable ('High'/'Pass'/'Fail') to numeric.
    
    Syllabus Concept: Data preparation for Decision Tree & Naive Bayes (Unit III)
    
    Args:
        df: Student DataFrame
    
    Returns:
        dict with keys: 'X', 'y', 'feature_names', 'label_encoder', 'class_names'
    """
    feature_columns = [
        'attendance', 'internal_marks', 'assignments_completed',
        'study_hours_per_week', 'previous_gpa', 'lab_score'
    ]
    
    X = df[feature_columns].values
    
    # Encode target: Fail=0, Pass=1, High=2
    le = LabelEncoder()
    y = le.fit_transform(df['result'])
    
    return {
        'X': X,
        'y': y,
        'feature_names': feature_columns,
        'label_encoder': le,
        'class_names': list(le.classes_)
    }


def categorize_value(value: float, feature_name: str) -> str:
    """
    Convert numeric feature values to categorical labels for the Knowledge Base.
    Used by Module 4 (FOL assertions).
    
    Syllabus Concept: FOL Assertions, Symbols and Interpretations (Unit II)
    
    Args:
        value: The numeric value
        feature_name: Name of the feature
    
    Returns:
        Category string: 'low', 'medium', or 'high'
    """
    thresholds = {
        'attendance': {'low': 60, 'high': 80},
        'internal_marks': {'low': 20, 'high': 35},
        'assignments_completed': {'low': 4, 'high': 7},
        'study_hours_per_week': {'low': 8, 'high': 15},
        'previous_gpa': {'low': 5.0, 'high': 7.0},
        'lab_score': {'low': 12, 'high': 22}
    }
    
    if feature_name in thresholds:
        t = thresholds[feature_name]
        if value <= t['low']:
            return 'low'
        elif value >= t['high']:
            return 'high'
        else:
            return 'medium'
    return 'unknown'


def get_student_by_name(df: pd.DataFrame, name: str) -> dict:
    """
    Look up a student by name and return their data as a dictionary.
    
    Args:
        df: Student DataFrame
        name: Student name to search for
    
    Returns:
        dict with student data, or None if not found
    """
    matches = df[df['name'].str.lower().str.contains(name.lower())]
    if len(matches) > 0:
        return matches.iloc[0].to_dict()
    return None
