"""
Module 2: NLP Feedback Analyzer
=================================
Analyzes teacher feedback text using NLP techniques.

Syllabus Concepts Covered (Unit IV):
  - Language Models (tokenization, preprocessing)
  - Bag of Words (word frequency vectors)
  - Parts-of-Speech Tagging
  - Text Classification with Naive Bayes
  - Grammar, Parsing
"""

import re
import json
import os
import math
from collections import Counter


# ==============================================================
# TEXT PREPROCESSING
# Syllabus: Language Models — Tokenization (Unit IV)
# ==============================================================

# Common English stop words
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}


def preprocess_text(text: str) -> list:
    """
    Tokenize text, convert to lowercase, remove stopwords and punctuation.
    
    Pipeline:
        Raw Text → Lowercase → Tokenize → Remove Punctuation → Remove Stopwords
    
    Syllabus: Language Models — tokenization and preprocessing (Unit IV)
    
    Args:
        text: Raw feedback text string
    
    Returns:
        List of cleaned tokens
    """
    # Lowercase
    text = text.lower()
    
    # Tokenize: split on non-alphanumeric characters
    tokens = re.findall(r'[a-z]+', text)
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    
    return tokens


# ==============================================================
# BAG OF WORDS
# Syllabus: Bag of Words (Unit IV)
# Represents text as a frequency vector of words
# ==============================================================

def build_bag_of_words(tokens: list, vocabulary: list = None) -> dict:
    """
    Create a word frequency vector from tokens.
    
    The Bag of Words model represents text as a multiset (bag) of its words,
    disregarding grammar and word order but keeping multiplicity.
    
    Example:
        Input: ['effort', 'good', 'effort', 'student', 'improve']
        Output: {'effort': 2, 'good': 1, 'student': 1, 'improve': 1}
    
    Syllabus: Bag of Words (Unit IV)
    
    Args:
        tokens: List of preprocessed tokens
        vocabulary: Optional fixed vocabulary to use (limits output to these words)
    
    Returns:
        Dictionary mapping words to their frequencies
    """
    if vocabulary:
        return {word: tokens.count(word) for word in vocabulary}
    else:
        return dict(Counter(tokens))


# ==============================================================
# PARTS-OF-SPEECH TAGGING
# Syllabus: Parts-of-Speech (Unit IV)
# Tags each word as Noun, Verb, Adjective, etc.
# ==============================================================

# Simple rule-based POS tagger (no external dependency required)
# In production, NLTK's pos_tag would be used, but this demonstrates the concept

# Common word -> POS mappings for educational domain
ADJECTIVES = {
    'excellent', 'good', 'great', 'strong', 'weak', 'poor', 'outstanding',
    'brilliant', 'exceptional', 'consistent', 'inconsistent', 'active',
    'creative', 'thorough', 'reliable', 'average', 'satisfactory', 'moderate',
    'significant', 'immediate', 'serious', 'dedicated', 'capable', 'diligent',
    'motivated', 'talented', 'promising', 'remarkable', 'superb', 'chronic',
    'remedial', 'mandatory', 'extra', 'best', 'better', 'worse', 'careless',
    'deep', 'genuine', 'steady', 'regular', 'additional', 'fundamental'
}

VERBS = {
    'shows', 'demonstrates', 'needs', 'improve', 'struggles', 'performs',
    'participates', 'helps', 'recommended', 'maintain', 'attend', 'study',
    'focus', 'develop', 'submit', 'engage', 'encourage', 'assign', 'consider',
    'lacks', 'failing', 'attending', 'submitting', 'improving', 'struggling',
    'showing', 'performing', 'working', 'understanding', 'learning', 'growing'
}

NOUNS = {
    'student', 'performance', 'effort', 'attendance', 'assignments', 'exam',
    'exams', 'class', 'concepts', 'understanding', 'skills', 'work',
    'improvement', 'potential', 'intervention', 'counseling', 'support',
    'mentor', 'discussions', 'participation', 'lab', 'theory', 'practical',
    'semester', 'grades', 'score', 'marks', 'behavior', 'motivation',
    'discipline', 'management', 'preparation', 'strategy', 'ability',
    'knowledge', 'progress', 'achievement', 'opportunity', 'course'
}

ADVERBS = {
    'very', 'consistently', 'rarely', 'always', 'sometimes', 'actively',
    'significantly', 'highly', 'poorly', 'well', 'seriously', 'regularly',
    'occasionally', 'frequently', 'never', 'usually', 'immediately'
}


def pos_tag_text(text: str) -> list:
    """
    Tag each word with its Part of Speech category.
    
    Uses a rule-based approach:
        1. Check against known word lists (Adjective, Verb, Noun, Adverb)
        2. Apply suffix rules for unknown words:
           - Words ending in '-ly' → Adverb
           - Words ending in '-tion', '-ment', '-ness' → Noun
           - Words ending in '-ing', '-ed' → Verb
           - Words ending in '-ful', '-ous', '-ive' → Adjective
        3. Default to Noun (most common POS in English)
    
    Syllabus: Parts-of-Speech Tagging (Unit IV)
    
    Args:
        text: Raw text string
    
    Returns:
        List of (word, POS_tag) tuples
        Tags: 'NN' (Noun), 'VB' (Verb), 'JJ' (Adjective), 'RB' (Adverb), 'DT' (Determiner), 'Other'
    """
    words = re.findall(r'[a-zA-Z]+', text.lower())
    tagged = []
    
    determiners = {'a', 'an', 'the', 'this', 'that', 'these', 'those'}
    prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 'of', 'about'}
    conjunctions = {'and', 'but', 'or', 'nor', 'yet', 'so'}
    pronouns = {'he', 'she', 'it', 'they', 'we', 'i', 'you', 'his', 'her', 'their'}
    
    for word in words:
        if word in determiners:
            tagged.append((word, 'DT'))
        elif word in prepositions:
            tagged.append((word, 'IN'))
        elif word in conjunctions:
            tagged.append((word, 'CC'))
        elif word in pronouns:
            tagged.append((word, 'PRP'))
        elif word in ADJECTIVES:
            tagged.append((word, 'JJ'))
        elif word in VERBS:
            tagged.append((word, 'VB'))
        elif word in NOUNS:
            tagged.append((word, 'NN'))
        elif word in ADVERBS:
            tagged.append((word, 'RB'))
        # Suffix-based rules
        elif word.endswith('ly'):
            tagged.append((word, 'RB'))
        elif word.endswith(('tion', 'ment', 'ness', 'ity', 'ance', 'ence')):
            tagged.append((word, 'NN'))
        elif word.endswith(('ing', 'ed', 'es')):
            tagged.append((word, 'VB'))
        elif word.endswith(('ful', 'ous', 'ive', 'able', 'ible', 'al', 'ent')):
            tagged.append((word, 'JJ'))
        else:
            tagged.append((word, 'NN'))  # Default: Noun
    
    return tagged


def get_pos_summary(pos_tags: list) -> dict:
    """
    Summarize POS tag distribution.
    
    Args:
        pos_tags: List of (word, tag) tuples
    
    Returns:
        dict with counts: {'Nouns': 12, 'Verbs': 8, 'Adjectives': 5, ...}
    """
    tag_map = {
        'NN': 'Nouns', 'VB': 'Verbs', 'JJ': 'Adjectives', 'RB': 'Adverbs',
        'DT': 'Determiners', 'IN': 'Prepositions', 'CC': 'Conjunctions',
        'PRP': 'Pronouns'
    }
    
    counts = Counter(tag for _, tag in pos_tags)
    summary = {}
    for tag, count in counts.most_common():
        label = tag_map.get(tag, 'Other')
        summary[label] = summary.get(label, 0) + count
    
    return summary


# ==============================================================
# SENTIMENT CLASSIFICATION
# Syllabus: Text Classification with Naive Bayes (Unit IV)
# ==============================================================

def load_sentiment_lexicon(json_path: str = None) -> dict:
    """
    Load positive and negative word lexicons from JSON file.
    
    Args:
        json_path: Path to feedback_samples.json
    
    Returns:
        dict with 'positive_words' and 'negative_words' sets
    """
    if json_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(base_dir, "data", "feedback_samples.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return {
        'positive_words': set(data['positive_words']),
        'negative_words': set(data['negative_words']),
        'sample_feedbacks': data.get('sample_feedbacks', [])
    }


def classify_sentiment(text: str, lexicon: dict = None) -> dict:
    """
    Classify feedback sentiment using a hybrid approach:
      1. Lexicon-based scoring: Count positive/negative words
      2. Naive Bayes-inspired probability: P(positive|words) vs P(negative|words)
    
    Algorithm:
        positive_score = Σ (1/total_words) for each positive word found
        negative_score = Σ (1/total_words) for each negative word found
        
        If positive_score > negative_score * 1.5 → Positive
        If negative_score > positive_score * 1.5 → Negative
        Else → Mixed
    
    Syllabus: Text Classification with Naive Bayes (Unit IV)
    
    Args:
        text: Feedback text
        lexicon: Optional pre-loaded lexicon dict
    
    Returns:
        dict with label, scores, and contributing words
    """
    if lexicon is None:
        lexicon = load_sentiment_lexicon()
    
    tokens = preprocess_text(text)
    all_words = re.findall(r'[a-z]+', text.lower())
    total_words = max(len(all_words), 1)
    
    # Find positive and negative words
    positive_found = [t for t in tokens if t in lexicon['positive_words']]
    negative_found = [t for t in tokens if t in lexicon['negative_words']]
    
    # Calculate scores (normalized by total words)
    positive_score = len(positive_found) / total_words
    negative_score = len(negative_found) / total_words
    
    # Normalize to 0-1 range
    total_sentiment = positive_score + negative_score
    if total_sentiment > 0:
        pos_normalized = positive_score / total_sentiment
        neg_normalized = negative_score / total_sentiment
    else:
        pos_normalized = 0.5
        neg_normalized = 0.5
    
    # Classify
    if pos_normalized > 0.65:
        label = "Positive 😊"
    elif neg_normalized > 0.65:
        label = "Negative 😟"
    else:
        label = "Mixed 😐"
    
    return {
        'label': label,
        'positive_score': round(pos_normalized, 3),
        'negative_score': round(neg_normalized, 3),
        'positive_words': positive_found,
        'negative_words': negative_found,
        'total_words_analyzed': len(tokens)
    }


# ==============================================================
# WORD-LEVEL SENTIMENT CONTRIBUTION
# Shows how each word shifts the overall sentiment
# ==============================================================

def get_word_contributions(text: str, lexicon: dict = None) -> list:
    """
    For each word in the text, calculate its sentiment contribution.
    Used for the highlighted-text visualization in the UI.
    
    Positive words get green highlighting, negative words get red.
    
    Syllabus: Language Models, Bag of Words (Unit IV)
    
    Args:
        text: Raw feedback text
        lexicon: Optional pre-loaded lexicon
    
    Returns:
        List of dicts: [{'word': 'excellent', 'score': +0.8, 'sentiment': 'positive'}, ...]
    """
    if lexicon is None:
        lexicon = load_sentiment_lexicon()
    
    words = re.findall(r'[a-zA-Z]+', text)
    contributions = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in lexicon['positive_words']:
            contributions.append({
                'word': word,
                'original': word,
                'score': 0.8,
                'sentiment': 'positive',
                'color': '#2ecc71'
            })
        elif word_lower in lexicon['negative_words']:
            contributions.append({
                'word': word,
                'original': word,
                'score': -0.8,
                'sentiment': 'negative',
                'color': '#e74c3c'
            })
        else:
            contributions.append({
                'word': word,
                'original': word,
                'score': 0.0,
                'sentiment': 'neutral',
                'color': '#95a5a6'
            })
    
    return contributions


# ==============================================================
# THEME EXTRACTION
# Syllabus: Grammar, Parsing (Unit IV)
# Uses POS tags to identify key themes
# ==============================================================

def extract_themes(text: str) -> dict:
    """
    Extract key themes from feedback using POS tag analysis.
    
    Strategy:
        - Strengths = positive adjectives + associated nouns
        - Weaknesses = negative adjectives/verbs + associated nouns
        - Action items = imperative verbs (needs, improve, focus, etc.)
    
    Syllabus: Grammar, Parsing (Unit IV)
    
    Args:
        text: Feedback text
    
    Returns:
        dict with 'strengths', 'weaknesses', 'action_items'
    """
    pos_tags = pos_tag_text(text)
    lexicon = load_sentiment_lexicon()
    
    strengths = []
    weaknesses = []
    action_items = []
    
    action_verbs = {'improve', 'focus', 'attend', 'develop', 'maintain', 'study',
                    'participate', 'engage', 'submit', 'practice', 'review', 'prepare'}
    
    for i, (word, tag) in enumerate(pos_tags):
        if word in lexicon['positive_words'] and tag in ('JJ', 'RB'):
            # Look for associated noun
            context = ""
            for j in range(i+1, min(i+3, len(pos_tags))):
                if pos_tags[j][1] == 'NN':
                    context = pos_tags[j][0]
                    break
            entry = f"{word} {context}".strip() if context else word
            if entry not in strengths:
                strengths.append(entry)
        
        elif word in lexicon['negative_words'] and tag in ('JJ', 'VB', 'RB'):
            context = ""
            for j in range(i+1, min(i+3, len(pos_tags))):
                if pos_tags[j][1] == 'NN':
                    context = pos_tags[j][0]
                    break
            entry = f"{word} {context}".strip() if context else word
            if entry not in weaknesses:
                weaknesses.append(entry)
        
        elif word in action_verbs:
            context_words = []
            for j in range(i+1, min(i+4, len(pos_tags))):
                if pos_tags[j][1] in ('NN', 'JJ'):
                    context_words.append(pos_tags[j][0])
            action = f"{word} {' '.join(context_words)}".strip()
            if action not in action_items:
                action_items.append(action)
    
    return {
        'strengths': strengths if strengths else ['No specific strengths identified'],
        'weaknesses': weaknesses if weaknesses else ['No specific weaknesses identified'],
        'action_items': action_items if action_items else ['No specific action items identified']
    }


def full_nlp_analysis(text: str) -> dict:
    """
    Run the complete NLP analysis pipeline on a feedback text.
    Combines all NLP functions into a single comprehensive result.
    
    Pipeline:
        Raw Text → Preprocess → BoW → POS Tag → Sentiment → Themes → Word Contributions
    
    Args:
        text: Raw feedback text
    
    Returns:
        Comprehensive analysis dict
    """
    lexicon = load_sentiment_lexicon()
    tokens = preprocess_text(text)
    bow = build_bag_of_words(tokens)
    pos_tags = pos_tag_text(text)
    pos_summary = get_pos_summary(pos_tags)
    sentiment = classify_sentiment(text, lexicon)
    themes = extract_themes(text)
    word_contribs = get_word_contributions(text, lexicon)
    
    return {
        'tokens': tokens,
        'bag_of_words': bow,
        'pos_tags': pos_tags,
        'pos_summary': pos_summary,
        'sentiment': sentiment,
        'themes': themes,
        'word_contributions': word_contribs,
        'original_text': text
    }
