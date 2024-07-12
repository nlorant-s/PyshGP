from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.push.type_library import PushTypeLibrary
from pyshgp.push.atoms import Literal
from pyshgp.push.types import PushFloat, PushInt
from pyshgp.gp.individual import Individual
from pyshgp.gp.population import Population
from pyshgp.gp.search import SearchAlgorithm, SearchConfiguration
from pyshgp.gp.selection import Lexicase
from pyshgp.gp.variation import LiteralMutation, VariationStrategy, AdditionMutation, DeletionMutation, Alternation, Genesis

from pyshgp.gp.variation import VariationOperator, AdditionMutation
from pyshgp.push.config import PushConfig
from pyshgp.gp.estimators import PushEstimator
from pyshgp.push.interpreter import PushInterpreter

from neural_network import NeuralNetwork, visualize_network, input_size, output_size
import numpy as np
import random
from multiprocessing import freeze_support

class CustomSearch(SearchAlgorithm):
    def __init__(self, config, variation_strategy):
        super().__init__(config)
        self.variation_strategy = variation_strategy
        self.selector = Lexicase(epsilon=True)

    def step(self):
        # Ensure all individuals have valid error vectors
        for individual in self.population:
            if individual.error_vector is None or len(individual.error_vector) == 0:
                individual.error_vector = np.array([float('inf')])
            elif np.isscalar(individual.error_vector):
                individual.error_vector = np.array([individual.error_vector])
        
        # Filter out individuals with invalid error vectors
        valid_individuals = [ind for ind in self.population if np.isfinite(ind.error_vector).all() and len(ind.error_vector) > 0]
        
        if not valid_individuals:
            print("Warning: No valid individuals found. Generating new random population.")
            return self.init_population()
        
        parents = self.selector.select(valid_individuals, n=len(self.population))
        children = []
        for _ in range(len(self.population)):
            op = self.variation_strategy.choose()
            child_genome = op.produce([p.genome for p in parents], self.config.spawner)
            child = Individual(child_genome, self.config.signature)
            children.append(child)
        self.population = Population(children)
        return self.population

    def init_population(self):
        individuals = []
        for _ in range(self.config.population_size):
            genome = self.config.spawner.spawn_genome(self.config.initial_genome_size)
            individual = Individual(genome, self.config.signature)
            individual.error_vector = np.array([float('inf')])  # Initialize with worst fitness
            individuals.append(individual)
        return Population(individuals)

MIN_LAYER_SIZE = 1
MAX_LAYER_SIZE = 16
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
    int_values = list(str( input_size))
    float_values = []
    for gene in genome:
        if isinstance(gene, Literal):
            if gene.push_type == PushInt:
                int_values.append(str(gene.value))
            elif gene.push_type == PushFloat:
                float_values.append(f"{gene.value:.4f}")
    int_values.append(str(output_size))
    
    return f"\nLayers: [{', '.join(int_values)}]\nnum weights: {len(', '.join(float_values))}"

def main():
    # Generate XOR data
    X = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                  [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                  [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                  [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    y = np.array([[0], [1], [1], [0], [1], [0], [0], [1],
                  [1], [0], [0], [1], [0], [1], [1], [0]])

    def fitness_eval(architecture, weights, X, y):
        try:
            all_layers = [input_size] + architecture + [output_size]
            network = NeuralNetwork(all_layers, weights)
            predictions = network.predict(X)
            mse = np.mean((predictions - y) ** 2)
            return np.array([mse])  # Return as a 1D numpy array with one element
        except Exception as e:
            print(f"Error in fitness evaluation: {str(e)}")
            return np.array([float('inf')])  # Return worst possible fitness as a 1D array

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

    MAX_HIDDEN_LAYERS = 3
    NUM_WEIGHTS = MAX_LAYER_SIZE**2 * MAX_HIDDEN_LAYERS + MAX_LAYER_SIZE
    TOTAL_GENES = MAX_HIDDEN_LAYERS + NUM_WEIGHTS

    # Create search configuration
    search_config = SearchConfiguration(
        signature=None,  # We'll set this later
        evaluator=None,  # We're not using the built-in evaluator
        spawner=spawner,
        population_size=3, # to adjust
        max_generations=50, # to adjust
        initial_genome_size=(TOTAL_GENES, TOTAL_GENES + 1),
        simplification_steps=0,
        error_threshold=0.0,
        selection="lexicase",
    )

    def create_variation_strategy(spawner):
        variation_strategy = VariationStrategy()
        
        # Add various mutation operators
        variation_strategy.add(AdditionMutation(addition_rate=0.09), 0.3)
        variation_strategy.add(DeletionMutation(deletion_rate=0.09), 0.3)
        
        # Add crossover operator
        variation_strategy.add(Alternation(alternation_rate=0.1, alignment_deviation=10), 0.3)
        
        # Add a small chance of creating entirely new genomes
        variation_strategy.add(Genesis(size=(TOTAL_GENES, TOTAL_GENES + 1)), 0.1)
        
        return variation_strategy

    # Create variation operator
    variation_strategy = create_variation_strategy(spawner)

    # Create custom search
    custom_search = CustomSearch(search_config, variation_strategy)

    # Initialize the population
    initial_population = Population()
    for _ in range(search_config.population_size):
        HIDDEN_LAYERS = np.random.randint(1, MAX_HIDDEN_LAYERS + 1)
        genome = custom_spawn_genome(HIDDEN_LAYERS, NUM_WEIGHTS)
        print("\nGENOME:", display_genome(genome))
        individual = Individual(genome, search_config.signature)
        initial_population.add(individual)

    custom_search.population = initial_population

    # Evolution loop
    for generation in range(50):
        # Evaluate each individual
        for individual in custom_search.population:
            architecture, weights = genome_extractor(individual.genome)
            error = fitness_eval(architecture, weights, X, y)
            individual.error_vector = np.array([error])
        
        # Print generation statistics
        print(f"Generation {generation + 1}:")
        if custom_search.population.evaluated:
            print(f"  Median error: {custom_search.population.median_error()}")
            print(f"  Best error: {min(individual.total_error for individual in custom_search.population)}")
            best_arch, best_params = genome_extractor(custom_search.population.best().genome)
            print(f"  Best architecture: {best_arch}")
            print(f"  Best float params: {best_params[:5]}...")  # Print first 5 float params
        else:
            print("  No evaluated individuals in population!")

        # Evolve to the next generation
        custom_search.step()

    # Get the best individual from the final population
    if custom_search.population.evaluated:
        best_individual = custom_search.population.best()
        best_architecture, best_params = genome_extractor(best_individual.genome)
        print("\nBest Neural Network Configuration:")
        print(f"  Architecture: {best_architecture}")
        print(f"  Float params: {best_params[:5]}...")  # Print first 5 float params
        print(f"  Error: {best_individual.total_error}")
    else:
        print("No evaluated individuals in final population!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()