from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.push.type_library import PushTypeLibrary

from pyshgp.gp.population import Population
from pyshgp.gp.evaluation import Evaluator
from pyshgp.push.interpreter import PushInterpreter

import numpy as np
from nn import SimpleNeuralNetwork, visualize_network

# Define custom fitness function
def custom_fitness_function(genome):
    # SimpleNeuralNetwork(genome) # needs to retuen a fitness score between 0 and 1
    return np.random.random()

# Initialize PyshGP components
instruction_set = InstructionSet().register_core()
type_library = PushTypeLibrary(register_core=True)
spawner = GeneSpawner(
    n_inputs=1,
    instruction_set=instruction_set,
    literals=[],
    erc_generators=[]
)

# Create the PushEstimator
estimator = PushEstimator(
    spawner=spawner,
    population_size=100,
    max_generations=50,
    initial_genome_size=(10, 50)
)

# Initialize the population
estimator.fit(X=[[]], y=[[]])  # Dummy data for initialization

# Main evolution loop
for generation in range(50):
    # Access genomes and compute fitness
    fitness_scores = []
    for individual in estimator.search.population.individuals:
        genome = individual.genome
        fitness = custom_fitness_function(genome)
        fitness_scores.append(fitness)
    
    # Print fitness scores for this generation
    print(f"Generation {generation + 1}: {fitness_scores}")
    
    # Update the population with fitness scores
    for individual, fitness in zip(estimator.search.population.individuals, fitness_scores):
        individual.error_vector = np.array([1 - fitness])  # Convert fitness to error (lower is better)
    
    # Evolve to the next generation
    estimator.search.step()

print("Evolution complete.")


'''
Additional working push methods

# Access the evaluated population
population = Population(individuals=estimator.population_size)

# Print details of the population
print(f"Population size: {len(population)}")
print(f"Median error: {population.median_error()}")
print(f"Genome diversity: {population.genome_diversity()}")

# Evaluate all unevaluated individuals in the population
evaluator = Evaluator(fitness)
population.evaluate(evaluator)

# Print the best individual after evaluation
best_after_evaluation = population.best()
print("Best individual after evaluation:", best_after_evaluation)

# Interpret the best individual's program to access the stack
interpreter = PushInterpreter()
interpreter.execute(best_after_evaluation.program, inputs=[X[0]])

# Accessing the stacks
int_stack = interpreter.state.int_stack
float_stack = interpreter.state.float_stack
bool_stack = interpreter.state.bool_stack

print("Integer Stack:", int_stack)
print("Float Stack:", float_stack)
print("Boolean Stack:", bool_stack)
'''