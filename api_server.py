"""
FastAPI REST API Gateway
========================
High-performance asynchronous API bridge connecting the TypeScript Glassmorphic
Frontend with the Python AI modules.

Course: Artificial Intelligence and Computational Systems (25 A D 35)
Candidate: Sai Sanjay A (1MS25AD053)
Guide: Shylaja V, Assistant Professor, Dept. of AI & DS, MSRIT
"""

import os
import sys
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Ensure project root is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from utils.data_cleaner import clean_student_dataset
from utils.feature_engineering import (
    engineer_features_pipeline, calculate_academic_consistency_index,
    calculate_effort_outcome_yield, calculate_assignment_diligence_rate,
    calculate_theory_practical_alignment, calculate_early_warning_composite_score,
    extract_sentiment_polarity
)
from modules.ml_engine import (
    train_decision_tree, train_naive_bayes, predict_with_probabilities,
    get_decision_path, get_feature_importances, compare_models
)
from modules.nlp_analyzer import (
    full_nlp_analysis, preprocess_text, build_bag_of_words,
    classify_sentiment
)
from modules.knowledge_base import (
    create_knowledge_base, assert_facts, backward_chain, forward_chain
)
from modules.ga_optimizer import run_ga, get_ga_summary

# Initialize FastAPI application
app = FastAPI(
    title="AI Student Performance Predictor & Advisory Gateway",
    version="2.0.0",
    description="Production REST API serving ML, NLP, FOL Expert Advisory, and Genetic Algorithm Engines."
)

# Enable CORS for local Vite/Next.js TypeScript frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# GLOBAL IN-MEMORY MODEL CACHE
# -----------------------------------------------------------------------------
DATA_CACHE: Dict[str, Any] = {}


def initialize_ai_pipeline():
    """Load, clean, engineer features, and train baseline models."""
    csv_path = os.path.join(CURRENT_DIR, "data", "students.csv")
    raw_df = pd.read_csv(csv_path)
    
    # 1. Clean Data
    clean_res = clean_student_dataset(raw_df)
    clean_df = clean_res['cleaned_dataframe']
    
    # 2. Engineer Features
    eng_df = engineer_features_pipeline(clean_df)
    
    # 3. Train Models on Cleaned Data
    feature_cols = [
        'attendance', 'internal_marks', 'assignments_completed',
        'study_hours_per_week', 'previous_gpa', 'lab_score'
    ]
    X = eng_df[feature_cols].values
    
    # Map labels: Fail=0, Pass=1, High=2
    label_map = {'Fail': 0, 'Pass': 1, 'High': 2}
    reverse_label_map = {0: 'Fail', 1: 'Pass', 2: 'High'}
    y = eng_df['result'].map(label_map).values
    
    dt_model = train_decision_tree(X, y, max_depth=4)
    nb_model = train_naive_bayes(X, y)
    
    DATA_CACHE['raw_df'] = raw_df
    DATA_CACHE['clean_df'] = clean_df
    DATA_CACHE['eng_df'] = eng_df
    DATA_CACHE['clean_report'] = clean_res
    DATA_CACHE['feature_cols'] = feature_cols
    DATA_CACHE['dt_model'] = dt_model
    DATA_CACHE['nb_model'] = nb_model
    DATA_CACHE['label_map'] = label_map
    DATA_CACHE['reverse_label_map'] = reverse_label_map
    DATA_CACHE['class_names'] = ['Fail', 'Pass', 'High']


# Initialize on startup
initialize_ai_pipeline()


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------
class StudentInput(BaseModel):
    attendance: float = Field(..., ge=0.0, le=100.0)
    internal_marks: float = Field(..., ge=0.0, le=50.0)
    assignments_completed: float = Field(..., ge=0.0, le=10.0)
    study_hours_per_week: float = Field(..., ge=0.0, le=60.0)
    previous_gpa: float = Field(..., ge=0.0, le=10.0)
    lab_score: float = Field(..., ge=0.0, le=30.0)
    teacher_feedback: Optional[str] = "Good student showing regular participation."


class NLPInput(BaseModel):
    text: str


class GoalQueryInput(BaseModel):
    student_id: int
    goal: str = "UrgentIntervention"


class GAOptimizeInput(BaseModel):
    subject_weights: Dict[str, float] = {
        "AI": 0.8,
        "Math": 0.9,
        "Algorithms": 0.7,
        "Databases": 0.5
    }
    total_hours: int = 30
    generations: int = 30


# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "AI Student Predictor API Gateway",
        "version": "2.0.0-PROD",
        "author": "Sai Sanjay A (1MS25AD053)"
    }


@app.get("/api/cohort")
def get_cohort_data():
    """Returns the complete cohort of students with cleaned & engineered features."""
    eng_df = DATA_CACHE['eng_df']
    students = eng_df.to_dict(orient="records")
    return {
        "total_students": len(students),
        "students": students,
        "cleaning_summary": {
            "duplicates_removed": DATA_CACHE['clean_report']['duplicates_removed'],
            "iqr_outliers_capped": DATA_CACHE['clean_report']['iqr_outliers_capped'],
            "range_violations_fixed": DATA_CACHE['clean_report']['range_violations_fixed'],
            "audit_log": DATA_CACHE['clean_report']['audit_log']
        }
    }


@app.post("/api/predict/all")
def predict_student_comprehensive(input_data: StudentInput):
    """
    Runs Decision Tree, Naive Bayes, Feature Engineering, and NLP on input.
    """
    feature_cols = DATA_CACHE['feature_cols']
    features_vec = [
        input_data.attendance,
        input_data.internal_marks,
        input_data.assignments_completed,
        input_data.study_hours_per_week,
        input_data.previous_gpa,
        input_data.lab_score
    ]
    
    # 1. Feature Engineering
    aci = calculate_academic_consistency_index(input_data.internal_marks, input_data.previous_gpa)
    eoy = calculate_effort_outcome_yield(input_data.internal_marks, input_data.study_hours_per_week)
    adr = calculate_assignment_diligence_rate(input_data.assignments_completed)
    tpar = calculate_theory_practical_alignment(input_data.internal_marks, input_data.lab_score)
    ewcs = calculate_early_warning_composite_score(
        input_data.attendance, input_data.internal_marks,
        input_data.assignments_completed, input_data.previous_gpa
    )
    sentiment = extract_sentiment_polarity(input_data.teacher_feedback or "")
    
    # 2. Decision Tree Prediction & Path
    dt = DATA_CACHE['dt_model']
    nb = DATA_CACHE['nb_model']
    class_names = DATA_CACHE['class_names']
    
    dt_pred_res = predict_with_probabilities(dt, features_vec, class_names)
    nb_pred_res = predict_with_probabilities(nb, features_vec, class_names)
    decision_path_info = get_decision_path(dt, features_vec, feature_cols, class_names)
    feature_importances = get_feature_importances(dt, feature_cols)
    
    # 3. NLP Analysis of feedback
    nlp_res = full_nlp_analysis(input_data.teacher_feedback or "")
    
    return {
        "predictions": {
            "decision_tree": {
                "predicted_class": dt_pred_res['predicted_class'],
                "confidence": dt_pred_res['confidence'],
                "probabilities": dt_pred_res['probabilities'],
                "decision_path": decision_path_info['steps']
            },
            "naive_bayes": {
                "predicted_class": nb_pred_res['predicted_class'],
                "confidence": nb_pred_res['confidence'],
                "probabilities": nb_pred_res['probabilities']
            }
        },
        "engineered_features": {
            "academic_consistency_index": aci,
            "effort_outcome_yield": eoy,
            "assignment_diligence_rate": adr,
            "theory_practical_alignment": tpar,
            "early_warning_composite_score": ewcs,
            "sentiment_polarity": sentiment['polarity'],
            "sentiment_label": sentiment['label']
        },
        "nlp_analysis": {
            "sentiment": nlp_res['sentiment'],
            "bow_top_words": nlp_res['bow'][:8] if 'bow' in nlp_res else [],
            "tokens_count": len(nlp_res.get('tokens', []))
        },
        "feature_importances": feature_importances
    }


@app.post("/api/kb/query")
def query_knowledge_base(query: GoalQueryInput):
    """
    Executes First-Order Logic Backward Chaining for pedagogical advising.
    """
    eng_df = DATA_CACHE['eng_df']
    student_row = eng_df[eng_df['student_id'] == query.student_id]
    if student_row.empty:
        raise HTTPException(status_code=404, detail="Student ID not found")
        
    student = student_row.iloc[0].to_dict()
    kb = create_knowledge_base()
    assert_facts(kb, student)
    
    # Forward chain for derived facts
    derived_facts = forward_chain(kb)
    
    # Backward chain on specified goal
    proof_tree = backward_chain(kb, query.goal, student.get('name', 'Student'))
    
    return {
        "student_id": query.student_id,
        "student_name": student.get('name'),
        "goal": query.goal,
        "proof_tree": proof_tree,
        "derived_facts": [str(f) for f in derived_facts]
    }


@app.post("/api/ga/schedule")
def optimize_schedule(ga_input: GAOptimizeInput):
    """
    Executes Genetic Algorithm study scheduler for cognitive fatigue optimization.
    """
    subjects = list(ga_input.subject_weights.keys())
    weights = list(ga_input.subject_weights.values())
    
    # Run GA optimization
    ga_result = run_ga(
        subjects=subjects,
        weakness_weights=weights,
        total_hours=ga_input.total_hours,
        generations=ga_input.generations
    )
    
    summary = get_ga_summary(ga_result)
    return {
        "schedule": summary,
        "fitness_history": ga_result.get('fitness_history', []),
        "generations": ga_input.generations
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
