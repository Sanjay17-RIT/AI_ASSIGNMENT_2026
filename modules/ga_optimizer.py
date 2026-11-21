"""
Module 3: Genetic Algorithm Feature Optimizer
===============================================
Custom-built GA that evolves the best feature subset for student prediction.

THIS IS BUILT FROM SCRATCH — not using any GA library.
Every operator (selection, crossover, mutation) is implemented manually.

Syllabus Concepts Covered (Unit V):
  - Genetic Algorithms — Representing Hypotheses
  - Genetic Operators (Crossover, Mutation)
  - Fitness Function and Selection
  - An Illustrative Example
"""

import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


# ==============================================================
# POPULATION INITIALIZATION
# Syllabus: Representing Hypotheses (Unit V)
# Each chromosome is a binary string representing a feature subset
# ==============================================================

def initialize_population(pop_size: int, num_features: int) -> list:
    """
    Create an initial population of random binary chromosomes.
    
    Each chromosome is a list of 0s and 1s:
      - 1 means "use this feature"
      - 0 means "skip this feature"
    
    Example with 6 features:
      [1, 0, 1, 1, 0, 1] → Use features 0, 2, 3, 5 (skip 1, 4)
    
    Constraint: At least 1 feature must be selected per chromosome.
    
    Syllabus: Genetic Algorithms — Representing Hypotheses (Unit V)
    
    Args:
        pop_size: Number of chromosomes in population
        num_features: Number of features (length of each chromosome)
    
    Returns:
        List of binary chromosomes (list of lists)
    """
    population = []
    for _ in range(pop_size):
        # Generate random binary chromosome
        chromosome = [random.randint(0, 1) for _ in range(num_features)]
        
        # Ensure at least one feature is selected
        if sum(chromosome) == 0:
            chromosome[random.randint(0, num_features - 1)] = 1
        
        population.append(chromosome)
    
    return population


# ==============================================================
# FITNESS FUNCTION
# Syllabus: Fitness Function and Selection (Unit V)
# Fitness = Classification accuracy using selected features
# ==============================================================

def fitness_function(chromosome: list, X: np.ndarray, y: np.ndarray, 
                     model_type: str = 'decision_tree') -> float:
    """
    Evaluate the fitness of a chromosome.
    
    Fitness = Cross-validation accuracy of a classifier trained using
    ONLY the features selected by this chromosome.
    
    Algorithm:
        1. Extract columns from X where chromosome[i] == 1
        2. Train a classifier on these selected features
        3. Compute 3-fold cross-validation accuracy
        4. Return mean accuracy as fitness score
    
    Higher fitness = better feature subset.
    
    Syllabus: Fitness Function and Selection (Unit V)
    
    Args:
        chromosome: Binary list [1,0,1,1,0,1]
        X: Full feature matrix
        y: Target labels
        model_type: 'decision_tree' or 'naive_bayes'
    
    Returns:
        Fitness score (0.0 to 1.0)
    """
    # Get indices of selected features
    selected = [i for i, gene in enumerate(chromosome) if gene == 1]
    
    if len(selected) == 0:
        return 0.0  # No features selected = worst fitness
    
    # Extract only selected features
    X_selected = X[:, selected]
    
    # Choose model
    if model_type == 'naive_bayes':
        model = GaussianNB()
    else:
        model = DecisionTreeClassifier(
            criterion='entropy', max_depth=4, random_state=42
        )
    
    # Cross-validation (3-fold for small dataset)
    try:
        scores = cross_val_score(model, X_selected, y, cv=3, scoring='accuracy')
        return float(scores.mean())
    except Exception:
        return 0.0


# ==============================================================
# SELECTION
# Syllabus: Selection (Unit V)
# Tournament selection: pick k random, keep the best
# ==============================================================

def selection(population: list, fitness_scores: list, 
              method: str = 'tournament', tournament_size: int = 3) -> list:
    """
    Select parents for the next generation.
    
    Tournament Selection Algorithm:
        1. Randomly pick 'tournament_size' individuals from population
        2. Select the one with the highest fitness
        3. Repeat until we have enough parents
    
    Why tournament selection?
        - Easy to implement and understand
        - Maintains selection pressure (better individuals more likely to be chosen)
        - But doesn't completely exclude weaker individuals (maintains diversity)
    
    Syllabus: Fitness Function and Selection (Unit V)
    
    Args:
        population: List of chromosomes
        fitness_scores: Corresponding fitness scores
        method: Selection method ('tournament' or 'roulette')
        tournament_size: Number of individuals per tournament
    
    Returns:
        List of selected parent chromosomes
    """
    selected = []
    pop_size = len(population)
    
    if method == 'roulette':
        # ---- Roulette Wheel Selection ----
        # Probability of selection ∝ fitness score
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            probabilities = [1.0 / pop_size] * pop_size
        else:
            probabilities = [f / total_fitness for f in fitness_scores]
        
        for _ in range(pop_size):
            r = random.random()
            cumulative = 0
            for i, p in enumerate(probabilities):
                cumulative += p
                if r <= cumulative:
                    selected.append(population[i][:])  # Copy
                    break
    else:
        # ---- Tournament Selection ----
        for _ in range(pop_size):
            # Pick random contestants
            contestants = random.sample(range(pop_size), min(tournament_size, pop_size))
            # Find the best among them
            best_idx = max(contestants, key=lambda i: fitness_scores[i])
            selected.append(population[best_idx][:])  # Copy
    
    return selected


# ==============================================================
# CROSSOVER
# Syllabus: Genetic Operators — Crossover (Unit V)
# Single-point crossover: swap tails of two parents
# ==============================================================

def crossover(parent1: list, parent2: list, crossover_rate: float = 0.8) -> tuple:
    """
    Single-point crossover between two parent chromosomes.
    
    Algorithm:
        1. Generate random number r
        2. If r < crossover_rate:
           a. Choose random crossover point
           b. Child1 = parent1[:point] + parent2[point:]
           c. Child2 = parent2[:point] + parent1[point:]
        3. Else: children are copies of parents
    
    Example:
        Parent 1: [1, 1, 0, | 0, 1, 0]
        Parent 2: [0, 1, 1, | 0, 1, 1]
                             ↕ crossover point = 3
        Child 1:  [1, 1, 0, | 0, 1, 1]
        Child 2:  [0, 1, 1, | 0, 1, 0]
    
    Syllabus: Genetic Operators — Crossover (Unit V)
    
    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        crossover_rate: Probability of crossover occurring
    
    Returns:
        Tuple of (child1, child2)
    """
    if random.random() < crossover_rate and len(parent1) > 1:
        # Choose crossover point (not at the very start or end)
        point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    else:
        child1 = parent1[:]
        child2 = parent2[:]
    
    return child1, child2


# ==============================================================
# MUTATION
# Syllabus: Genetic Operators — Mutation (Unit V)
# Bit-flip mutation: randomly flip 0↔1
# ==============================================================

def mutation(chromosome: list, mutation_rate: float = 0.1) -> list:
    """
    Bit-flip mutation on a chromosome.
    
    Algorithm:
        For each gene (bit) in the chromosome:
            Generate random number r
            If r < mutation_rate:
                Flip the bit (0→1 or 1→0)
        
        Ensure at least one feature remains selected after mutation.
    
    Example:
        Before: [1, 1, 0, 0, 1, 1]
        After:  [1, 1, 0, 1, 1, 1]  ← bit 3 flipped (0→1)
    
    Syllabus: Genetic Operators — Mutation (Unit V)
    
    Args:
        chromosome: Binary chromosome to mutate
        mutation_rate: Probability of each bit flipping
    
    Returns:
        Mutated chromosome
    """
    mutated = chromosome[:]
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]  # Flip: 0→1 or 1→0
    
    # Ensure at least one feature is selected
    if sum(mutated) == 0:
        mutated[random.randint(0, len(mutated) - 1)] = 1
    
    return mutated


# ==============================================================
# MAIN GA LOOP
# Syllabus: Genetic Algorithms — An Illustrative Example (Unit V)
# ==============================================================

def run_ga(X: np.ndarray, y: np.ndarray, feature_names: list,
           generations: int = 50, pop_size: int = 20,
           crossover_rate: float = 0.8, mutation_rate: float = 0.1,
           model_type: str = 'decision_tree',
           progress_callback=None) -> dict:
    """
    Run the complete Genetic Algorithm for feature optimization.
    
    Full Algorithm:
        1. INITIALIZE random population of binary chromosomes
        2. For each GENERATION:
           a. EVALUATE fitness of every chromosome
           b. Record best fitness and best chromosome
           c. SELECT parents using tournament selection
           d. Apply CROSSOVER to create children
           e. Apply MUTATION to introduce variation
           f. REPLACE old population with new generation
        3. Return the best solution found across all generations
    
    This is a COMPLETE implementation — not a wrapper around a library.
    Every step (initialize, evaluate, select, crossover, mutate) uses
    our custom functions defined above.
    
    Syllabus: Genetic Algorithms — An Illustrative Example (Unit V)
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        generations: Number of generations to evolve
        pop_size: Population size
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation per bit
        model_type: 'decision_tree' or 'naive_bayes'
        progress_callback: Optional callback(generation, best_fitness) for UI progress
    
    Returns:
        dict with:
            'best_chromosome': Best binary chromosome found
            'best_features': Names of selected features
            'best_accuracy': Best fitness (accuracy) achieved
            'history': List of per-generation stats
            'initial_accuracy': Accuracy using all features (baseline)
            'improvement': Accuracy improvement over baseline
    """
    num_features = X.shape[1]
    
    # --- Step 0: Baseline (using ALL features) ---
    baseline_fitness = fitness_function([1] * num_features, X, y, model_type)
    
    # --- Step 1: INITIALIZE population ---
    population = initialize_population(pop_size, num_features)
    
    # Track evolution history
    history = []
    overall_best_chromosome = None
    overall_best_fitness = 0.0
    
    # --- Step 2: EVOLVE over generations ---
    for gen in range(generations):
        
        # --- Step 2a: EVALUATE fitness ---
        fitness_scores = [
            fitness_function(chrom, X, y, model_type) 
            for chrom in population
        ]
        
        # --- Step 2b: Record best ---
        gen_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_chromosome = population[gen_best_idx][:]
        gen_avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        # Update overall best
        if gen_best_fitness > overall_best_fitness:
            overall_best_fitness = gen_best_fitness
            overall_best_chromosome = gen_best_chromosome[:]
        
        # Record history
        history.append({
            'generation': gen + 1,
            'best_fitness': round(gen_best_fitness, 4),
            'avg_fitness': round(gen_avg_fitness, 4),
            'best_chromosome': gen_best_chromosome,
            'diversity': len(set(str(c) for c in population))  # Unique chromosomes
        })
        
        # Progress callback for UI
        if progress_callback:
            progress_callback(gen + 1, gen_best_fitness)
        
        # --- Step 2c: SELECT parents ---
        parents = selection(population, fitness_scores, method='tournament')
        
        # --- Step 2d: CROSSOVER ---
        children = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = crossover(parents[i], parents[i+1], crossover_rate)
            children.append(child1)
            children.append(child2)
        
        # Handle odd population size
        if len(children) < pop_size:
            children.append(parents[-1][:])
        
        # --- Step 2e: MUTATION ---
        population = [mutation(child, mutation_rate) for child in children[:pop_size]]
        
        # Elitism: always keep the best from previous generation
        population[0] = overall_best_chromosome[:]
    
    # --- Step 3: Return results ---
    best_features = [
        feature_names[i] for i, gene in enumerate(overall_best_chromosome) 
        if gene == 1
    ]
    
    improvement = overall_best_fitness - baseline_fitness
    
    return {
        'best_chromosome': overall_best_chromosome,
        'best_features': best_features,
        'best_accuracy': round(overall_best_fitness, 4),
        'baseline_accuracy': round(baseline_fitness, 4),
        'improvement': round(improvement, 4),
        'history': history,
        'generations_run': generations,
        'population_size': pop_size,
        'total_evaluations': generations * pop_size
    }


def get_ga_summary(result: dict) -> str:
    """
    Generate a human-readable summary of the GA results.
    
    Args:
        result: Output from run_ga()
    
    Returns:
        Formatted summary string
    """
    summary = f"""
### 🧬 Genetic Algorithm Results

**Best Feature Subset Found:**
{', '.join(result['best_features'])}

**Performance:**
| Metric | Value |
|--------|-------|
| Baseline Accuracy (all features) | {result['baseline_accuracy']:.1%} |
| Optimized Accuracy (GA selected) | {result['best_accuracy']:.1%} |
| Improvement | {'+' if result['improvement'] >= 0 else ''}{result['improvement']:.1%} |
| Generations Run | {result['generations_run']} |
| Population Size | {result['population_size']} |
| Total Evaluations | {result['total_evaluations']} |

**Best Chromosome:** `{result['best_chromosome']}`

**Interpretation:** The GA found that using only **{len(result['best_features'])}** features 
({', '.join(result['best_features'])}) achieves {'better' if result['improvement'] > 0 else 'comparable'} 
accuracy than using all features.
"""
    return summary
