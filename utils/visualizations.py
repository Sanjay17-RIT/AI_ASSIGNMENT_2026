"""
Visualization Utilities
========================
Shared chart and plotting helpers for the Streamlit dashboard.
Uses Plotly with transparent/dark styling and modern color palettes.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np


FONT_CONFIG = dict(color='#E2E8F0', family='Plus Jakarta Sans, sans-serif')


def create_probability_gauge(probabilities: dict, prediction: str) -> go.Figure:
    """
    Create a horizontal bar chart showing class probabilities.
    Used by Naive Bayes & Decision Tree prediction displays.
    
    Syllabus Concept: Quantifying Uncertainty (Unit III)
    """
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ['#10B981' if c == 'High' else '#F59E0B' if c == 'Pass' else '#F43F5E' for c in classes]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f'{p:.1%}' for p in probs],
        textposition='auto',
        textfont=dict(size=14, color='white', family='Plus Jakarta Sans')
    ))
    
    fig.update_layout(
        title=dict(text=f"Confidence Distribution (Predicted: {prediction})", font=dict(color='#F8FAFC', size=15)),
        xaxis_title=dict(text="Probability", font=FONT_CONFIG),
        yaxis_title=dict(text="Class", font=FONT_CONFIG),
        xaxis=dict(range=[0, 1], tickformat='.0%', tickfont=FONT_CONFIG, gridcolor='rgba(148,163,184,0.1)'),
        yaxis=dict(tickfont=FONT_CONFIG),
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=FONT_CONFIG
    )
    
    return fig


def create_feature_importance_chart(importances: np.ndarray, feature_names: list) -> go.Figure:
    """
    Create a horizontal bar chart showing Decision Tree feature importances.
    
    Syllabus Concept: Decision Tree Classification — information gain (Unit III)
    """
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    fig = go.Figure(go.Bar(
        x=sorted_importances,
        y=sorted_names,
        orientation='h',
        marker=dict(
            color='#6366F1',
            line=dict(color='#818CF8', width=1)
        ),
        text=[f'{v:.1%}' for v in sorted_importances],
        textposition='auto',
        textfont=dict(color='white', family='Plus Jakarta Sans')
    ))
    
    fig.update_layout(
        title=dict(text="Feature Importance (Information Gain)", font=dict(color='#F8FAFC', size=15)),
        xaxis_title=dict(text="Entropy Reduction Weight", font=FONT_CONFIG),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=FONT_CONFIG, gridcolor='rgba(148,163,184,0.1)'),
        yaxis=dict(tickfont=FONT_CONFIG),
        font=FONT_CONFIG
    )
    
    return fig


def create_ga_evolution_chart(history: list) -> go.Figure:
    """
    Create a line chart showing GA fitness over generations.
    
    Syllabus Concept: Genetic Algorithm — Fitness Function (Unit V)
    """
    generations = [h['generation'] for h in history]
    best = [h['best_fitness'] for h in history]
    avg = [h['avg_fitness'] for h in history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=generations, y=best,
        mode='lines+markers',
        name='Best Fitness (Accuracy)',
        line=dict(color='#10B981', width=3),
        marker=dict(size=6, color='#10B981')
    ))
    
    fig.add_trace(go.Scatter(
        x=generations, y=avg,
        mode='lines',
        name='Generation Avg Fitness',
        line=dict(color='#A855F7', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title=dict(text="🧬 Genetic Algorithm Evolution Trajectory", font=dict(color='#F8FAFC', size=16)),
        xaxis_title=dict(text="Generation Cycle", font=FONT_CONFIG),
        yaxis_title=dict(text="Fitness (Validation Accuracy)", font=FONT_CONFIG),
        yaxis=dict(range=[0, 1.05], tickformat='.0%', tickfont=FONT_CONFIG, gridcolor='rgba(148,163,184,0.1)'),
        xaxis=dict(tickfont=FONT_CONFIG, gridcolor='rgba(148,163,184,0.1)'),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=FONT_CONFIG,
        legend=dict(
            yanchor="bottom", y=0.03, xanchor="right", x=0.98,
            bgcolor='rgba(15,23,42,0.8)', bordercolor='rgba(148,163,184,0.2)', borderwidth=1,
            font=dict(color='#E2E8F0')
        )
    )
    
    return fig


def create_bow_chart(word_counts: dict, top_n: int = 15) -> go.Figure:
    """
    Create a bar chart showing Bag of Words frequencies.
    
    Syllabus Concept: Bag of Words (Unit IV)
    """
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words = [w[0] for w in sorted_words]
    counts = [w[1] for w in sorted_words]
    
    fig = go.Figure(go.Bar(
        x=counts,
        y=words,
        orientation='h',
        marker=dict(
            color='#EC4899',
            line=dict(color='#F472B6', width=1)
        ),
        text=counts,
        textposition='auto',
        textfont=dict(color='white', family='Plus Jakarta Sans')
    ))
    
    fig.update_layout(
        title=dict(text=f"Bag of Words Distribution (Top {len(words)} Terms)", font=dict(color='#F8FAFC', size=15)),
        xaxis_title=dict(text="Frequency Count", font=FONT_CONFIG),
        height=max(250, len(words) * 26),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=FONT_CONFIG, gridcolor='rgba(148,163,184,0.1)'),
        yaxis=dict(autorange="reversed", tickfont=FONT_CONFIG),
        font=FONT_CONFIG
    )
    
    return fig


def create_sentiment_gauge(positive_score: float, negative_score: float) -> go.Figure:
    """
    Create an indicator gauge showing sentiment balance.
    
    Syllabus Concept: Text Classification (Unit IV)
    """
    val = positive_score * 100
    bar_color = '#10B981' if positive_score > negative_score else '#F43F5E'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={'text': "Positive Polarity", 'font': {'size': 16, 'color': '#F8FAFC'}},
        number={'suffix': '%', 'font': {'size': 28, 'color': bar_color, 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94A3B8'},
            'bar': {'color': bar_color},
            'bgcolor': 'rgba(15,23,42,0.8)',
            'borderwidth': 1,
            'bordercolor': 'rgba(148,163,184,0.2)',
            'steps': [
                {'range': [0, 35], 'color': 'rgba(244, 63, 94, 0.2)'},
                {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [65, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
            'threshold': {
                'line': {'color': '#F8FAFC', 'width': 3},
                'thickness': 0.8,
                'value': val
            }
        }
    ))
    
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=FONT_CONFIG
    )
    return fig


def create_class_distribution_pie(result_counts: dict) -> go.Figure:
    """
    Create a donut chart showing class distribution (for batch analysis).
    """
    labels = list(result_counts.keys())
    values = list(result_counts.values())
    color_map = {'High': '#10B981', 'Pass': '#F59E0B', 'Fail': '#F43F5E'}
    colors = [color_map.get(k, '#6366F1') for k in labels]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color='#0F172A', width=2)),
        textinfo='label+percent+value',
        textfont=dict(size=14, color='white', family='Plus Jakarta Sans'),
        hole=0.45
    ))
    
    fig.update_layout(
        title=dict(text="Class Distribution Breakdown", font=dict(color='#F8FAFC', size=16)),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=FONT_CONFIG,
        legend=dict(font=dict(color='#E2E8F0'))
    )
    
    return fig


def create_chromosome_display(chromosome: list, feature_names: list) -> str:
    """
    Create a visual representation of a GA chromosome.
    
    Syllabus Concept: Representing Hypotheses (Unit V)
    """
    display = ""
    for gene, name in zip(chromosome, feature_names):
        if gene == 1:
            display += f"✅ **{name}**\n\n"
        else:
            display += f"❌ <span style='color:#64748B;'>~~{name}~~</span>\n\n"
    return display
