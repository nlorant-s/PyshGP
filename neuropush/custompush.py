# TO DO:
# fix false attributes and methods (get running)
# determine input data & fitness function (start with XOR)
# connect to nn.py and ensure individual fitness evals

from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.push.type_library import PushTypeLibrary
from pyshgp.gp.individual import Individual
from pyshgp.gp.population import Population
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.gp.search import SearchAlgorithm, SearchConfiguration
from pyshgp.gp.selection import Lexicase
from pyshgp.gp.variation import VariationOperator
from pyshgp.push.config import PushConfig
import numpy as np

class CustomSearch(SearchAlgorithm):
    def __init__(self, config, variation_op):
        super().__init__(config)
        self.variation_op = variation_op
        self.selector = Lexicase(epsilon=True)

    def step(self):
        parents = self.selector.select(self.population, n=len(self.population))
        children = [self.variation_op.produce(parents, self.config.spawner) for _ in range(len(self.population))]
        self.population = Population(children)
        return self.population

def configure_as_nn_and_test_fitness(individual, X, y):
    # This is a placeholder function. Replace with your actual NN configuration and testing logic.
    # For now, we'll just return a random error vector.
    return np.random.rand(len(y))

# Initialize PyshGP components
instruction_set = InstructionSet().register_core()
type_library = PushTypeLibrary(register_core=True)
push_config = PushConfig(type_library=type_library)
spawner = GeneSpawner(
    n_inputs=1,
    instruction_set=instruction_set,
    literals=[],
    erc_generators=[]
)

# Generate some dummy data for this example
X = np.random.rand(100, 1)
y = X * 2 + 1 + np.random.normal(0, 0.1, (100, 1))

# Create search configuration
search_config = SearchConfiguration(
    signature=None,  # We'll set this later
    evaluator=None,  # We're not using the built-in evaluator
    spawner=spawner,
    population_size=100, # to adjust
    max_generations=50, # to adjust
    initial_genome_size=(10, 50),
    simplification_steps=500,
    error_threshold=0.0,
    selection="lexicase",
    variation="umad", # does this work?
    push_config=push_config # does this work?
)

# Create variation operator
variation_op = VariationOperator(search_config.push_config, search_config.spawner)

# Create custom search
custom_search = CustomSearch(search_config, variation_op)

# Initialize the population
initial_population = Population()
for _ in range(search_config.population_size):
    genome = spawner.spawn_genome(search_config.initial_genome_size)
    individual = Individual(genome, search_config.signature)
    initial_population.add(individual)

custom_search.population = initial_population

# Evolution loop
for generation in range(50):
    # Evaluate each individual using the NN configuration
    for individual in custom_search.population:
        error_vector = configure_as_nn_and_test_fitness(individual, X, y)
        individual.error_vector = error_vector
    
    # Print generation statistics
    print(f"Generation {generation + 1}:")
    print(f"  Median error: {custom_search.population.median_error()}")
    print(f"  Best error: {min(np.sum(individual.error_vector) for individual in custom_search.population)}")
    print(f"  Genome diversity: {custom_search.population.genome_diversity()}")
    
    # Here you can do additional processing or store data as needed
    
    # Evolve to the next generation
    custom_search.step()

# Get the best individual from the final population
best_individual = custom_search.population.best()
print("\nBest Individual:")
print(f"  Error: {np.sum(best_individual.error_vector)}")

# Interpret the best program
interpreter = PushInterpreter(instruction_set)
program_output = interpreter.run(best_individual.program, inputs=X[0].tolist())
print(f"\nProgram output for first input: {program_output}")