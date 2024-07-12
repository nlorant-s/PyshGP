from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.push.type_library import PushTypeLibrary
from pyshgp.push.atoms import Literal
from pyshgp.push.types import PushFloat, PushInt
from pyshgp.gp.individual import Individual
from pyshgp.gp.population import Population
from pyshgp.gp.search import SearchAlgorithm, SearchConfiguration
from pyshgp.gp.selection import Lexicase

from pyshgp.gp.variation import VariationOperator, AdditionMutation
from pyshgp.push.config import PushConfig
from pyshgp.gp.estimators import PushEstimator
from pyshgp.push.interpreter import PushInterpreter

from neural_network import NeuralNetwork, visualize_network, input_size, output_size
import numpy as np
import random
from multiprocessing import freeze_support

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

MIN_LAYER_SIZE = 1
MAX_LAYER_SIZE = 64
MIN_FLOAT_VALUE = -1.0
MAX_FLOAT_VALUE = 1.0

# Define ERC (Ephemeral Random Constant) generators
def weights_generator():
    return Literal(value=random.uniform(MIN_FLOAT_VALUE, MAX_FLOAT_VALUE), push_type=PushFloat)

def layer_size_generator():
    return Literal(value=random.randint(MIN_LAYER_SIZE, MAX_LAYER_SIZE), push_type=PushInt)

# Custom genome spawning function to ensure we get the desired number of each type
def custom_spawn_genome(num_layers, num_floats):
    genome = []
    for _ in range(num_layers):
        genome.append(layer_size_generator())
    for _ in range(num_floats):
        genome.append(weights_generator())
    return genome
    # Function to interpret a genome as a neural network architecture
def genome_extractor(genome):
    architecture = []
    weights = []
    for gene in genome:
        if isinstance(gene, Literal):
            if gene.push_type == PushInt:
                architecture.append(gene.value)
            elif gene.push_type == PushFloat:
                weights.append(gene.value)
    return architecture, weights

def display_genome(genome):
    int_values = []
    float_values = []
    for gene in genome:
        if isinstance(gene, Literal):
            if gene.push_type == PushInt:
                int_values.append(str(gene.value))
            elif gene.push_type == PushFloat:
                float_values.append(f"{gene.value:.4f}")
    
    return f"\nInts: [{', '.join(int_values)}]\nFloats: [{', '.join(float_values)}]"

def main():
    # Generate XOR data
    X = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                  [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                  [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                  [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    y = np.array([[0], [1], [1], [0], [1], [0], [0], [1],
                  [1], [0], [0], [1], [0], [1], [1], [0]])

    def fitness_eval(architecture, weights):
        all_layers = [input_size] + architecture + [output_size]
        network = NeuralNetwork(all_layers, weights)
        visualization = visualize_network(network, 'show')
        return visualization

    # Initialize PyshGP components
    instruction_set = InstructionSet().register_core()

    type_library = PushTypeLibrary(register_core=False)
    type_library.register(PushInt)
    instruction_set = InstructionSet(type_library=type_library)

    # push_config = PushConfig()

    spawner = GeneSpawner(
        n_inputs=0,
        instruction_set=instruction_set,
        literals=[],
        erc_generators=[layer_size_generator, weights_generator],
    )

    NUM_LAYERS = np.random.randint(1, 5)
    # NUM_WEIGHTS = MAX_LAYER_SIZE**2 * 3 + MAX_LAYER_SIZE # need to solve this
    NUM_WEIGHTS = np.random.randint(1, 100)

    # Create search configuration
    search_config = SearchConfiguration(
        signature=None,  # We'll set this later
        evaluator=None,  # We're not using the built-in evaluator
        spawner=spawner,
        population_size=3, # to adjust
        max_generations=50, # to adjust
        initial_genome_size=(10, 100),
        simplification_steps=0,
        error_threshold=0.0,
        selection="lexicase",
    )

    # Create variation operator
    # variation_op = VariationOperator(rate=0.1, spawner=search_config.spawner, instruction_set=search_config.spawner.instruction_set)

    # Create custom search
    custom_search = CustomSearch(search_config, variation_op="umad")

    # Initialize the population
    initial_population = Population()
    for _ in range(search_config.population_size):
        genome = custom_spawn_genome(NUM_LAYERS, NUM_WEIGHTS)
        print("\nGENOME:", display_genome(genome))
        individual = Individual(genome, search_config.signature)
        initial_population.add(individual)

    custom_search.population = initial_population

    # Evolution loop
    for generation in range(50):
        # Evaluate each individual
        for individual in custom_search.population:
            architecture, weights = genome_extractor(individual.genome)
            error = fitness_eval(architecture, weights)
            individual.error_vector = np.array([error])
        
        # Print generation statistics
        print(f"Generation {generation + 1}:")
        print(f"  Median error: {custom_search.population.median_error()}")
        print(f"  Best error: {min(individual.total_error for individual in custom_search.population)}")
        best_arch, best_params = genome_extractor(custom_search.population.best().genome)
        print(f"  Best architecture: {best_arch}")
        print(f"  Best float params: {best_params[:5]}...")  # Print first 5 float params
        
        # Evolve to the next generation
        custom_search.step()

    # Get the best individual from the final population
    best_individual = custom_search.population.best()
    best_architecture, best_params = genome_extractor(best_individual.genome)
    print("\nBest Neural Network Configuration:")
    print(f"  Architecture: {best_architecture}")
    print(f"  Float params: {best_params[:5]}...")  # Print first 5 float params
    print(f"  Error: {best_individual.total_error}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()