"""
Module 4: Knowledge-Based Advisory System
============================================
Implements a rule-based expert system using First-Order Logic (FOL)
with Forward Chaining and Backward Chaining inference engines.

THIS IS A REAL RULE ENGINE — not hardcoded if-else statements.
Facts and rules are separate data structures, and chaining algorithms
iterate dynamically to derive new conclusions.

Syllabus Concepts Covered (Unit II):
  - Knowledge-Based Agents
  - First-Order Logic: Syntax and Semantics
  - FOL Assertions, Symbols and Interpretations
  - Atomic Sentences, Complex Sentences
  - Forward Chaining
  - Backward Chaining
  - The Knowledge Engineering Process
"""

from dataclasses import dataclass, field
from typing import Optional


# ==============================================================
# DATA STRUCTURES — FOL Representation
# Syllabus: FOL Syntax and Semantics, Atomic Sentences (Unit II)
# ==============================================================

@dataclass
class Fact:
    """
    A ground predicate in First-Order Logic.
    
    In FOL, an atomic sentence is a predicate applied to terms.
    Example: Attendance(Alice, low) means "Alice's attendance is low"
    
    Syllabus: FOL Assertions, Atomic Sentences (Unit II)
    
    Attributes:
        predicate: The predicate name (e.g., "Attendance", "Grade")
        subject: The subject entity (e.g., "Alice", "Student1")
        value: The value/property (e.g., "low", "Pass", "positive")
    """
    predicate: str      # e.g., "Attendance", "Grade", "Sentiment"
    subject: str        # e.g., "Alice"
    value: str          # e.g., "low", "Pass", "positive"
    
    def __str__(self):
        return f"{self.predicate}({self.subject}, {self.value})"
    
    def __eq__(self, other):
        if not isinstance(other, Fact):
            return False
        return (self.predicate == other.predicate and 
                self.subject == other.subject and 
                self.value == other.value)
    
    def __hash__(self):
        return hash((self.predicate, self.subject, self.value))


@dataclass
class Rule:
    """
    A production rule with conditions (antecedents) and conclusion (consequent).
    
    In FOL: IF condition1(x) ∧ condition2(x) ∧ ... THEN conclusion(x)
    
    Example:
        Rule: "At-Risk Detection"
        IF Grade(x) = "Fail" AND Attendance(x) = "low"
        THEN AtRisk(x) = "true"
    
    Syllabus: Complex Sentences, Knowledge Engineering Process (Unit II)
    
    Attributes:
        name: Human-readable rule name
        conditions: List of (predicate, value) tuples that must all be satisfied
        conclusion: (predicate, value) tuple to assert if conditions are met
        explanation: Why this rule exists
        recommendation: Action to take when rule fires
        priority: Higher priority rules are displayed first
    """
    name: str
    conditions: list            # [(predicate, value), ...]
    conclusion: tuple           # (predicate, value)
    explanation: str
    recommendation: str
    priority: int = 1           # Higher = more important


# ==============================================================
# KNOWLEDGE BASE CREATION
# Syllabus: The Knowledge Engineering Process (Unit II)
# This is where domain knowledge is encoded into logical rules
# ==============================================================

def create_knowledge_base() -> list:
    """
    Define the complete Knowledge Base with all rules.
    
    This implements the Knowledge Engineering Process:
        1. Identify the domain (student performance advisory)
        2. Define predicates (Grade, Attendance, Sentiment, etc.)
        3. Encode expert knowledge as IF-THEN rules
        4. Assign priorities and recommendations
    
    Syllabus: The Knowledge Engineering Process (Unit II)
    
    Returns:
        List of Rule objects
    """
    rules = [
        # ----- Rule 1: At-Risk Detection -----
        # IF Grade(x) = "Fail" AND Attendance(x) = "low"
        # THEN AtRisk(x) = "true"
        Rule(
            name="At-Risk Detection",
            conditions=[("Grade", "Fail"), ("Attendance", "low")],
            conclusion=("AtRisk", "true"),
            explanation="Student has failing grade AND low attendance — both critical indicators",
            recommendation="🚨 Mandatory counseling session + remedial classes. Contact parents immediately.",
            priority=5
        ),
        
        # ----- Rule 2: Attendance Warning -----
        # IF Attendance(x) = "low"
        # THEN AttendanceWarning(x) = "true"
        Rule(
            name="Attendance Warning",
            conditions=[("Attendance", "low")],
            conclusion=("AttendanceWarning", "true"),
            explanation="Attendance is below the acceptable threshold (60%)",
            recommendation="⚠️ Send attendance warning notice to student and parents.",
            priority=4
        ),
        
        # ----- Rule 3: Scholarship Candidate -----
        # IF Grade(x) = "High" AND Attendance(x) = "high" AND Sentiment(x) = "positive"
        # THEN ScholarshipCandidate(x) = "true"
        Rule(
            name="Scholarship Candidate",
            conditions=[("Grade", "High"), ("Attendance", "high"), ("Sentiment", "positive")],
            conclusion=("ScholarshipCandidate", "true"),
            explanation="Excellent grades + high attendance + positive teacher feedback",
            recommendation="🏆 Nominate for merit scholarship. Consider for academic excellence award.",
            priority=3
        ),
        
        # ----- Rule 4: Needs Mentoring -----
        # IF Grade(x) = "Pass" AND Sentiment(x) = "negative"
        # THEN NeedsMentoring(x) = "true"
        Rule(
            name="Needs Mentoring",
            conditions=[("Grade", "Pass"), ("Sentiment", "negative")],
            conclusion=("NeedsMentoring", "true"),
            explanation="Passing grades but negative teacher feedback indicates hidden issues",
            recommendation="👨‍🏫 Assign faculty mentor for one-on-one academic guidance.",
            priority=3
        ),
        
        # ----- Rule 5: High Potential -----
        # IF Grade(x) = "Pass" AND StudyHours(x) = "high" AND Sentiment(x) = "positive"
        # THEN HighPotential(x) = "true"
        Rule(
            name="High Potential",
            conditions=[("Grade", "Pass"), ("StudyHours", "high"), ("Sentiment", "positive")],
            conclusion=("HighPotential", "true"),
            explanation="Passing grades + high study effort + positive feedback = untapped potential",
            recommendation="🌟 Encourage to aim higher. Consider advanced coursework or research projects.",
            priority=2
        ),
        
        # ----- Rule 6: Urgent Intervention (CHAINED RULE) -----
        # IF AtRisk(x) = "true" AND Sentiment(x) = "negative"
        # THEN UrgentIntervention(x) = "true"
        # NOTE: This is a CHAINED rule — depends on Rule 1 firing first!
        Rule(
            name="Urgent Intervention",
            conditions=[("AtRisk", "true"), ("Sentiment", "negative")],
            conclusion=("UrgentIntervention", "true"),
            explanation="Student is at-risk AND teacher feedback is negative — highest priority",
            recommendation="🆘 IMMEDIATE meeting with HOD + parents. Consider academic probation.",
            priority=5
        ),
        
        # ----- Rule 7: Lab Weakness -----
        # IF LabScore(x) = "low" AND Grade(x) != "High"
        # → Simplified as: IF LabScore(x) = "low" AND Grade(x) = "Fail"
        # OR: IF LabScore(x) = "low" AND Grade(x) = "Pass"
        Rule(
            name="Lab Weakness (Failing)",
            conditions=[("LabScore", "low"), ("Grade", "Fail")],
            conclusion=("LabWeakness", "true"),
            explanation="Poor lab performance combined with failing grade",
            recommendation="🔧 Extra lab sessions + peer programming. Assign lab partner.",
            priority=3
        ),
        
        Rule(
            name="Lab Weakness (Passing)",
            conditions=[("LabScore", "low"), ("Grade", "Pass")],
            conclusion=("LabWeakness", "true"),
            explanation="Poor lab performance despite passing theory — practical skills gap",
            recommendation="🔧 Additional lab practice sessions. Focus on hands-on skills.",
            priority=2
        ),
        
        # ----- Rule 8: Overall Strong -----
        # IF Grade(x) = "High" AND Sentiment(x) = "positive" AND LabScore(x) = "high"
        # THEN OverallStrong(x) = "true"
        Rule(
            name="Overall Strong",
            conditions=[("Grade", "High"), ("Sentiment", "positive"), ("LabScore", "high")],
            conclusion=("OverallStrong", "true"),
            explanation="Excellent in all areas: grades, feedback, and practical skills",
            recommendation="⭐ Candidate for Teaching Assistant role. Recommend for internships.",
            priority=2
        ),
        
        # ----- Rule 9: Improving Student -----
        # IF Grade(x) = "Pass" AND Sentiment(x) = "positive" AND Attendance(x) = "high"
        # THEN Improving(x) = "true"
        Rule(
            name="Improving Student",
            conditions=[("Grade", "Pass"), ("Sentiment", "positive"), ("Attendance", "high")],
            conclusion=("Improving", "true"),
            explanation="Passing with positive feedback and good attendance — on upward trajectory",
            recommendation="📈 Encourage continued effort. Set specific goals for next semester.",
            priority=2
        ),
        
        # ----- Rule 10: Disengaged Student -----
        # IF Attendance(x) = "low" AND Sentiment(x) = "negative" AND StudyHours(x) = "low"
        # THEN Disengaged(x) = "true"
        Rule(
            name="Disengaged Student",
            conditions=[("Attendance", "low"), ("Sentiment", "negative"), ("StudyHours", "low")],
            conclusion=("Disengaged", "true"),
            explanation="Low attendance + negative feedback + low study hours = complete disengagement",
            recommendation="🚨 Personal meeting required. Investigate root cause (personal issues, motivation, etc.).",
            priority=4
        ),
    ]
    
    return rules


# ==============================================================
# FACT ASSERTION
# Syllabus: FOL Assertions, Symbols and Interpretations (Unit II)
# Convert raw data into ground predicates
# ==============================================================

def assert_facts(student_name: str, student_data: dict, 
                 ml_prediction: str, sentiment_label: str) -> list:
    """
    Convert student data + ML prediction + NLP sentiment into FOL facts.
    
    This is the process of GROUNDING — taking real-world data and
    representing it as logical facts that the inference engine can use.
    
    Example output:
        Grade(Alice, Pass)
        Attendance(Alice, low)
        Sentiment(Alice, negative)
        StudyHours(Alice, medium)
        LabScore(Alice, low)
    
    Syllabus: FOL Assertions, Symbols and Interpretations (Unit II)
    
    Args:
        student_name: Name of the student
        student_data: dict with keys: attendance, study_hours_per_week, lab_score
        ml_prediction: ML model prediction ('High', 'Pass', or 'Fail')
        sentiment_label: NLP sentiment label string
    
    Returns:
        List of Fact objects
    """
    facts = []
    
    # Fact 1: Grade (from ML prediction)
    facts.append(Fact("Grade", student_name, ml_prediction))
    
    # Fact 2: Attendance (categorized)
    attendance = student_data.get('attendance', 75)
    if attendance <= 60:
        att_level = "low"
    elif attendance >= 80:
        att_level = "high"
    else:
        att_level = "medium"
    facts.append(Fact("Attendance", student_name, att_level))
    
    # Fact 3: Sentiment (from NLP analysis)
    # Clean the sentiment label (remove emoji)
    if "positive" in sentiment_label.lower():
        sent = "positive"
    elif "negative" in sentiment_label.lower():
        sent = "negative"
    else:
        sent = "mixed"
    facts.append(Fact("Sentiment", student_name, sent))
    
    # Fact 4: Study Hours (categorized)
    study_hours = student_data.get('study_hours_per_week', 10)
    if study_hours <= 8:
        hours_level = "low"
    elif study_hours >= 15:
        hours_level = "high"
    else:
        hours_level = "medium"
    facts.append(Fact("StudyHours", student_name, hours_level))
    
    # Fact 5: Lab Score (categorized)
    lab_score = student_data.get('lab_score', 15)
    if lab_score <= 12:
        lab_level = "low"
    elif lab_score >= 22:
        lab_level = "high"
    else:
        lab_level = "medium"
    facts.append(Fact("LabScore", student_name, lab_level))
    
    return facts


# ==============================================================
# FORWARD CHAINING
# Syllabus: Forward Chaining / Data-Driven Inference (Unit II)
# Start from facts → apply rules → derive new facts → repeat
# ==============================================================

def forward_chain(facts: list, rules: list) -> dict:
    """
    Forward Chaining Algorithm — Data-Driven Inference.
    
    Algorithm:
        1. Start with initial facts
        2. REPEAT:
           a. For each rule in the knowledge base:
              - Check if ALL conditions are satisfied by current facts
              - If YES and conclusion not already known:
                  - Add conclusion as a new fact
                  - Record which rule fired and why
                  - Set flag: new_fact_added = True
           b. If no new facts were added → STOP (fixed point reached)
        3. Return all derived facts, fired rules, and inference trace
    
    Key property: Forward chaining is SOUND — it only derives facts
    that logically follow from the rules and known facts.
    
    Syllabus: Forward Chaining (Unit II)
    
    Args:
        facts: List of initial Fact objects
        rules: List of Rule objects (knowledge base)
    
    Returns:
        dict with:
            'all_facts': All facts (initial + derived)
            'derived_facts': Only newly derived facts
            'fired_rules': List of rules that fired with their trigger info
            'inference_trace': Step-by-step trace of the reasoning
            'recommendations': List of recommendations from fired rules
    """
    # Working memory: set of all known facts
    known_facts = set(facts)
    derived_facts = []
    fired_rules = []
    inference_trace = []
    recommendations = []
    
    iteration = 0
    max_iterations = 50  # Safety limit to prevent infinite loops
    
    inference_trace.append({
        'step': 0,
        'action': 'INITIALIZE',
        'detail': f"Starting with {len(facts)} initial facts",
        'facts_added': [str(f) for f in facts]
    })
    
    new_fact_added = True
    while new_fact_added and iteration < max_iterations:
        new_fact_added = False
        iteration += 1
        
        for rule in rules:
            # Check if ALL conditions are satisfied
            subject = None
            all_conditions_met = True
            matched_facts = []
            
            for predicate, value in rule.conditions:
                condition_met = False
                for fact in known_facts:
                    if fact.predicate == predicate and fact.value == value:
                        condition_met = True
                        subject = fact.subject
                        matched_facts.append(str(fact))
                        break
                
                if not condition_met:
                    all_conditions_met = False
                    break
            
            if all_conditions_met and subject is not None:
                # Create the conclusion fact
                new_fact = Fact(rule.conclusion[0], subject, rule.conclusion[1])
                
                # Only add if not already known
                if new_fact not in known_facts:
                    known_facts.add(new_fact)
                    derived_facts.append(new_fact)
                    new_fact_added = True
                    
                    fired_rules.append({
                        'rule_name': rule.name,
                        'conditions_matched': matched_facts,
                        'conclusion': str(new_fact),
                        'explanation': rule.explanation,
                        'recommendation': rule.recommendation,
                        'priority': rule.priority
                    })
                    
                    recommendations.append({
                        'rule': rule.name,
                        'recommendation': rule.recommendation,
                        'priority': rule.priority
                    })
                    
                    inference_trace.append({
                        'step': iteration,
                        'action': 'RULE_FIRED',
                        'rule': rule.name,
                        'conditions': matched_facts,
                        'derived': str(new_fact),
                        'detail': f"Rule '{rule.name}' fired: {' ∧ '.join(matched_facts)} → {new_fact}"
                    })
    
    # Sort recommendations by priority (highest first)
    recommendations.sort(key=lambda x: x['priority'], reverse=True)
    fired_rules.sort(key=lambda x: x['priority'], reverse=True)
    
    inference_trace.append({
        'step': iteration + 1,
        'action': 'COMPLETE',
        'detail': f"Forward chaining complete. {len(derived_facts)} new facts derived, {len(fired_rules)} rules fired."
    })
    
    return {
        'all_facts': list(known_facts),
        'derived_facts': derived_facts,
        'fired_rules': fired_rules,
        'inference_trace': inference_trace,
        'recommendations': recommendations,
        'iterations': iteration
    }


# ==============================================================
# BACKWARD CHAINING
# Syllabus: Backward Chaining / Goal-Directed Inference (Unit II)
# Start from a goal → check if provable from facts + rules
# ==============================================================

def backward_chain(facts: list, rules: list, goal_predicate: str, 
                   goal_value: str, subject: str) -> dict:
    """
    Backward Chaining Algorithm — Goal-Directed Inference.
    
    Algorithm:
        prove(goal):
            1. If goal is in known facts → return True (base case)
            2. For each rule whose conclusion matches the goal:
               a. Try to prove ALL conditions of that rule (recursive)
               b. If all conditions are proven → goal is proven
            3. If no rule can prove the goal → return False
    
    Key property: Backward chaining is GOAL-DIRECTED — it only
    explores rules relevant to the query, making it more efficient
    than forward chaining for specific queries.
    
    Syllabus: Backward Chaining (Unit II)
    
    Args:
        facts: List of Fact objects
        rules: List of Rule objects
        goal_predicate: What we're trying to prove (e.g., "ScholarshipCandidate")
        goal_value: Expected value (e.g., "true")
        subject: The subject entity (e.g., "Alice")
    
    Returns:
        dict with:
            'goal': String representation of the goal
            'proven': True/False
            'proof_trace': Step-by-step trace of the proof attempt
    """
    goal = Fact(goal_predicate, subject, goal_value)
    proof_trace = []
    proven_cache = {}  # Memoization to avoid redundant checks
    
    def prove(target: Fact, depth: int = 0) -> bool:
        """Recursive proof function."""
        indent = "  " * depth
        
        # Check cache
        target_key = str(target)
        if target_key in proven_cache:
            proof_trace.append({
                'depth': depth,
                'action': 'CACHE_HIT',
                'detail': f"{indent}{'✅' if proven_cache[target_key] else '❌'} {target} (cached)"
            })
            return proven_cache[target_key]
        
        # Base case: check if target is in known facts
        for fact in facts:
            if fact == target:
                proof_trace.append({
                    'depth': depth,
                    'action': 'FACT_FOUND',
                    'detail': f"{indent}✅ {target} — found in known facts"
                })
                proven_cache[target_key] = True
                return True
        
        proof_trace.append({
            'depth': depth,
            'action': 'SEARCHING',
            'detail': f"{indent}🔍 Trying to prove: {target}"
        })
        
        # Try to prove using rules
        for rule in rules:
            # Check if rule's conclusion matches our target
            if rule.conclusion[0] == target.predicate and rule.conclusion[1] == target.value:
                proof_trace.append({
                    'depth': depth,
                    'action': 'TRYING_RULE',
                    'detail': f"{indent}📋 Trying rule: '{rule.name}'"
                })
                
                # Try to prove all conditions
                all_proven = True
                for cond_pred, cond_val in rule.conditions:
                    cond_fact = Fact(cond_pred, subject, cond_val)
                    
                    if not prove(cond_fact, depth + 1):
                        all_proven = False
                        proof_trace.append({
                            'depth': depth + 1,
                            'action': 'CONDITION_FAILED',
                            'detail': f"{indent}  ❌ Condition {cond_fact} — FAILED"
                        })
                        break
                
                if all_proven:
                    proof_trace.append({
                        'depth': depth,
                        'action': 'RULE_SUCCEEDED',
                        'detail': f"{indent}✅ Rule '{rule.name}' succeeded → {target} PROVEN"
                    })
                    proven_cache[target_key] = True
                    return True
        
        # No rule could prove it
        proof_trace.append({
            'depth': depth,
            'action': 'FAILED',
            'detail': f"{indent}❌ {target} — could NOT be proven"
        })
        proven_cache[target_key] = False
        return False
    
    # Run the proof
    result = prove(goal)
    
    return {
        'goal': str(goal),
        'proven': result,
        'proof_trace': proof_trace
    }


# ==============================================================
# AVAILABLE GOALS FOR BACKWARD CHAINING UI
# ==============================================================

def get_available_goals() -> list:
    """
    Return list of goals that can be queried via backward chaining.
    
    Returns:
        List of (predicate, value, description) tuples
    """
    return [
        ("AtRisk", "true", "Is the student at risk of failing?"),
        ("AttendanceWarning", "true", "Should the student receive an attendance warning?"),
        ("ScholarshipCandidate", "true", "Is the student eligible for a scholarship?"),
        ("NeedsMentoring", "true", "Does the student need a faculty mentor?"),
        ("HighPotential", "true", "Does the student have high untapped potential?"),
        ("UrgentIntervention", "true", "Does the student need urgent intervention?"),
        ("LabWeakness", "true", "Does the student have weakness in lab performance?"),
        ("OverallStrong", "true", "Is the student strong in all areas?"),
        ("Improving", "true", "Is the student on an improving trajectory?"),
        ("Disengaged", "true", "Is the student disengaged from academics?"),
    ]


def get_inference_explanation(fired_rules: list) -> str:
    """
    Format the full inference chain in a human-readable format for the UI.
    
    Syllabus: Knowledge-Based Agents (Unit II)
    
    Args:
        fired_rules: List of fired rule dicts from forward_chain()
    
    Returns:
        Formatted markdown string
    """
    if not fired_rules:
        return "ℹ️ No rules were triggered for this student's profile."
    
    explanation = "### 🔗 Inference Chain\n\n"
    
    for i, rule in enumerate(fired_rules, 1):
        explanation += f"**Step {i}: Rule '{rule['rule_name']}' Fired**\n"
        explanation += f"- **Conditions matched:** {' ∧ '.join(rule['conditions_matched'])}\n"
        explanation += f"- **Derived:** {rule['conclusion']}\n"
        explanation += f"- **Why:** {rule['explanation']}\n"
        explanation += f"- **Action:** {rule['recommendation']}\n\n"
    
    return explanation
