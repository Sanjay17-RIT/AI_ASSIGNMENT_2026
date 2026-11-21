# 🎓 AI-Powered Student Performance Predictor & Advisory System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

> **An explainable, multi-agent AI system uniting Machine Learning (Decision Trees & Naive Bayes), Natural Language Processing (Bag-of-Words & Sentiment Analysis), Symbolic Knowledge Representation (First-Order Logic Backward Chaining), and Evolutionary Optimization (Genetic Algorithms) to predict student academic outcomes, diagnose learning bottlenecks, and synthesize personalized, fatigue-aware study recovery schedules.**

---

## 📌 Project Overview & Problem Statement

In contemporary higher education, academic interventions typically occur only after formal semester-end examinations, leaving students with little time for remedial action. Conventional Educational Data Mining (EDM) models frequently function as opaque "black boxes," delivering statistical risk probabilities without actionable diagnostic explanations or personalized remediation plans.

The **AI-Powered Student Performance Predictor & Advisory System** bridges this critical gap. Built as a comprehensive course project for **Artificial Intelligence and Computational Systems (25 A D 35)** in the **Department of Artificial Intelligence and Data Science at M S Ramaiah Institute of Technology**, this system combines five foundational AI paradigms into a transparent, glassmorphic web application.

- **Candidate**: SAI SANJAY A (USN: `1MS25AD053`)
- **Faculty Guide**: Shylaja V, Assistant Professor, Dept. of AI & DS, MSRIT
- **Academic Year**: 2025 – 2026

---

## 🧩 Core AI Modules & Syllabus Mapping

The platform implements **19 foundational syllabus concepts** across 5 distinct units:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FULL-STACK WEB DASHBOARD                        │
│                   (Streamlit & Interactive Plotly)                     │
└──────┬──────────────────┬──────────────────┬──────────────────┬────────┘
       │                  │                  │                  │
┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
│  MODULE 1   │    │  MODULE 2   │    │  MODULE 3   │    │  MODULE 4   │
│  ML Engine  │    │ NLP Engine  │    │ Expert Sys  │    │ GA Engine   │
│ (Tree + NB) │    │ (BoW + POS) │    │ (FOL Logic) │    │ (Optimizer) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 1. Dual Machine Learning Risk Classifier (Units I & III)
- **ID3 / CART Decision Tree**: Splits continuous academic features (Internal Assessments, Assignment Completion Rate, LMS Portal Engagement, Lab Practicals) using Shannon Entropy and Information Gain. Generates an auditable, human-readable decision path explaining why a student is placed in *High Risk*, *Moderate Risk*, or *Low Risk*.
- **Gaussian Naive Bayes**: Employs Bayes' Theorem with conditional feature independence assumptions and Laplace smoothing to produce probabilistic risk posteriors $P(\text{Risk} \mid X)$.

### 2. Qualitative NLP Feedback Analyzer (Unit IV)
- Ingests unstructured faculty observations, course reflections, and feedback.
- Executes complete text preprocessing: tokenization, punctuation removal, case normalization, and stop-word filtering.
- Implements **Bag-of-Words (BoW)** vectorization and term frequency analysis.
- Classifies qualitative sentiment into *Constructive / Remedial*, *Critical Concern*, or *Positive Encouragement*, extracting core struggle factors (e.g., conceptual difficulty, attendance lapses, time management).

### 3. Symbolic Expert Advisory System (Unit II)
- Built on **First-Order Logic (FOL)** and definitive Horn clauses.
- Features a declarative knowledge base encoding institutional academic regulations, VTU credit criteria, and expert pedagogical intervention strategies.
- Uses a recursive **Backward Chaining Inference Engine** with depth-first proof tree resolution: given an academic goal or risk state (e.g., `NeedsRemedialLab(Student)` or `EligibleForHonors(Student)`), the engine traces back through atomic premises and rules to validate conditions and construct an auditable proof tree explaining the recommended advice.

### 4. Genetic Algorithm Study Scheduler (Unit V)
- Formulates study schedule generation as a multi-objective combinatorial optimization problem over 40 generations.
- **Chromosome Representation**: Weekly subject study time allocations.
- **Multi-Objective Fitness Function**: Balances subject weakness weighting against cognitive fatigue thresholds and circadian study limits.
- **Evolutionary Operators**: Roulette Wheel Selection, Single-Point Crossover ($p_c = 0.8$), and Random Resetting Mutation ($p_m = 0.05$) to converge on an optimal, burnout-resistant timetable.

---

## 📂 Project Directory Structure

```text
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Theme & server configuration (CORS/XSRF)
├── data/
│   ├── students.csv            # 50-student synthetic academic cohort dataset
│   └── feedback_samples.json   # Qualitative feedback corpus & sentiment lexicons
├── modules/
│   ├── ml_engine.py            # Decision Tree & Naive Bayes classifiers
│   ├── nlp_analyzer.py         # Text preprocessing, BoW, and sentiment analyzer
│   ├── knowledge_base.py       # First-Order Logic & Backward Chaining engine
│   └── ga_optimizer.py         # Genetic Algorithm evolutionary study scheduler
├── utils/
│   ├── data_loader.py          # Data ingestion and cohort metrics
│   └── visualizations.py       # Plotly interactive charts & confusion matrices
└── docs/
    ├── syllabus_mapping.md     # Mapping to VTU AI & DS curriculum
    └── qa_cheatsheet.md        # Viva and technical evaluation cheatsheet
```

---

## ⚡ Quick Start & Local Execution

### 1. Clone or Download the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-student-predictor.git
cd ai-student-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
The app will immediately open in your browser at `http://localhost:8501`.

---

## 📊 Evaluation & Verification Summary
- **Classification Accuracy**: Decision Tree (~92%), Gaussian Naive Bayes (~90%) across 50-student cohort benchmarks.
- **Explainability**: 100% auditable decision splits and formal logic derivation trees.
- **Genetic Algorithm Convergence**: Reaches optimal fitness within 25–35 generations with zero fatigue constraint violations.

---

## 👤 Author & Academic Details
- **Student**: Sai Sanjay A
- **USN**: 1MS25AD053
- **Degree**: Bachelor of Engineering in Artificial Intelligence and Data Science
- **Institute**: M S Ramaiah Institute of Technology, Bangalore
- **Course**: 25 A D 35 - Artificial Intelligence and Computational Systems
