from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.push.type_library import PushTypeLibrary
from pyshgp.push.atoms import Literal
from pyshgp.push.types import PushFloat, PushInt
from pyshgp.gp.individual import Individual
from pyshgp.gp.population import Population
from pyshgp.gp.search import SearchAlgorithm, SearchConfiguration
from pyshgp.gp.selection import Lexicase, Tournament
from pyshgp.gp.variation import LiteralMutation, VariationStrategy, AdditionMutation, DeletionMutation, Alternation, Genesis, Cloning
from neural_network import NeuralNetwork, visualize_network, input_size, output_size
import numpy as np
import random
from multiprocessing import freeze_support
from datetime import datetime

# Constants
population_size = 100
max_generations = 8

MIN_LAYER_SIZE = 1
MAX_LAYER_SIZE = 16
MIN_FLOAT_VALUE = -1.0
MAX_FLOAT_VALUE = 1.0

MAX_HIDDEN_LAYERS = 3
NUM_WEIGHTS = MAX_LAYER_SIZE**2 * MAX_HIDDEN_LAYERS + MAX_LAYER_SIZE
TOTAL_GENES = MAX_HIDDEN_LAYERS + NUM_WEIGHTS

print_genomes = False
show_network = False

# XOR data
X = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
y = np.array([[0], [1], [1], [0], [1], [0], [0], [1],
                [1], [0], [0], [1], [0], [1], [1], [0]])

class CustomSearch(SearchAlgorithm):
    def __init__(self, config, variation_strategy):
        """
        50% of the time, select parents using lexicase selection.
        50% of the time, select parents using tournament selection.
        """
        super().__init__(config)
        self.variation_strategy = variation_strategy
        self.lexicase_selector = Lexicase(epsilon=True)
        self.tournament_selector = Tournament(tournament_size=7) # decrease tournament size for greater pressure

    def step(self):
        """
        Evolve the population by selecting parents and producing children.
        """
        # Ensure we're working with a Population object
        if not isinstance(self.population, Population):
            self.population = Population(self.population)

        # Ensure all individuals have valid error vectors
        for individual in self.population:
            if individual.error_vector is None or len(individual.error_vector) == 0:
                individual.error_vector = np.array([float('inf')])
            elif np.isscalar(individual.error_vector):
                individual.error_vector = np.array([individual.error_vector])
        
        # Filter out individuals with invalid error vectors
        valid_individuals = Population([ind for ind in self.population if np.isfinite(ind.error_vector).all() and len(ind.error_vector) > 0])
        
        if len(valid_individuals) == 0:
            print("Warning: No valid individuals found. Generating new random population.")
            return self.init_population()
        
        # Elitism: Keep the best individual
        best_individual = min(valid_individuals, key=lambda ind: ind.total_error)

        # Adjust selection sizes based on the number of valid individuals
        lexicase_size = min(len(valid_individuals), len(self.population)//2)
        tournament_size = min(len(valid_individuals), len(self.population)//2)

        # Adjust tournament size if necessary
        self.tournament_selector.tournament_size = min(self.tournament_selector.tournament_size, len(valid_individuals))

        parents_lexicase = self.lexicase_selector.select(valid_individuals, n=lexicase_size)
        parents_tournament = self.tournament_selector.select(valid_individuals, n=tournament_size)
        parents = parents_lexicase + parents_tournament

        # Debugging to ensure selection with respect to error vectors
        # print("Selected parents error vectors:")
        # for parent in parents_lexicase[:3] + parents_tournament[:2]:  # Print a few from each selection method
        #     print(f"  {parent.error_vector:.4f}")
        
        children = [] # [best_individual]

        for _ in range(len(self.population)):
            op = np.random.choice(self.variation_strategy.elements)
            child_genome = op.produce([p.genome for p in parents], self.config.spawner)
            child = Individual(child_genome, self.config.signature)
            # print(f"Child: {display_genome(child.genome)}") # FOR DEBUG
            children.append(child)
        self.population = Population(children)
        return self.population

    def init_population(self):
        """
        Initialize the population with random individuals.
        """
        individuals = []
        for _ in range(self.config.population_size):
            genome = self.config.spawner.spawn_genome(self.config.initial_genome_size)
            individual = Individual(genome, self.config.signature)
            individual.error_vector = np.array([float('inf')])  # Initialize with worst fitness
            individuals.append(individual)
        return Population(individuals)

# Define ERC (Ephemeral Random Constant) generators
def weights_generator():
    return Literal(value=random.uniform(MIN_FLOAT_VALUE, MAX_FLOAT_VALUE), push_type=PushFloat)

def layer_size_generator():
    return Literal(value=random.randint(MIN_LAYER_SIZE, MAX_LAYER_SIZE), push_type=PushInt)

# Custom genome spawning function to ensure we get the desired number of each type
def custom_spawn_genome(num_layers, num_weights):
    genome = []
    for _ in range(num_layers):
        genome.append(layer_size_generator())
    for _ in range(num_weights):
        genome.append(weights_generator())
    
    return genome

# interprets a genome as a neural network architecture
def genome_extractor(genome):
    architecture = []
    weights = []
    for gene in genome:
        if isinstance(gene, Literal):
            if gene.push_type == PushInt and len(architecture) < MAX_HIDDEN_LAYERS:
                architecture.append(max(MIN_LAYER_SIZE, min(MAX_LAYER_SIZE, gene.value)))
            elif gene.push_type == PushFloat:
                weights.append(gene.value)
    
    # Ensure we always have MAX_HIDDEN_LAYERS
    while len(architecture) < MAX_HIDDEN_LAYERS:
        architecture.append(random.randint(MIN_LAYER_SIZE, MAX_LAYER_SIZE))
    
    return architecture[:MAX_HIDDEN_LAYERS], weights[:NUM_WEIGHTS]

# Display the genomes architecture and number of weights in the terminal
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

def logger(logged):
    # Get current date and time
    current_time = datetime.now().strftime("%m%d%Y-%H%M")
    
    # Create a filename with the current date and time
    filename = f"{current_time}.txt"
    print(f"Logging to {filename}")
    
    with open(filename, "a") as file:
        file.write(logged)
        file.save(f'/logs/{filename}')
    
    return None

def main():
    def fitness_eval(architecture, weights, X, y):
        try:
            all_layers = [input_size] + architecture + [output_size]
            network = NeuralNetwork(all_layers, weights)
            if show_network:
                visualize_network(network, 'show')
            predictions = network.predict(X)

            # Calculate mean squared error
            mse = np.mean((predictions - y) ** 2)
            # Calculate binary cross-entropy loss
            bce = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

            return np.array([bce])  # Return as a 1D numpy array with one element
        except Exception as e:
            # print(f"Error in fitness evaluation: {str(e)}")
            return None

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

    # Create search configuration
    search_config = SearchConfiguration(
        signature=None,  # We'll set this later
        evaluator=None,  # We're not using the built-in evaluator
        spawner=spawner,
        population_size=population_size,
        max_generations=max_generations,
        initial_genome_size=(TOTAL_GENES, TOTAL_GENES + 1),
        simplification_steps=0,
        error_threshold=0.0,
        selection="lexicase",
    )

    def create_variation_strategy(spawner):
        variation_strategy = VariationStrategy()
        
        # Fine-grained mutation operators
        variation_strategy.add(AdditionMutation(addition_rate=0.005), 0.2)
        variation_strategy.add(DeletionMutation(deletion_rate=0.005), 0.2)
        
        # Crossover operator
        variation_strategy.add(Alternation(alternation_rate=0.1, alignment_deviation=5), 0.3)
        
        # Literal mutation (for fine-tuning weights)
        # variation_strategy.add(LiteralMutation(push_type=PushFloat, rate=0.1), 0.2)
        
        # Occasional genome reset (for maintaining diversity)
        variation_strategy.add(Genesis(size=(TOTAL_GENES, TOTAL_GENES + 1)), 0.05)
        
        # Cloning (to preserve good solutions)
        variation_strategy.add(Cloning(), 0.05)
        
        return variation_strategy

    # Create variation operator
    variation_strategy = create_variation_strategy(spawner)    

    # Create custom search
    custom_search = CustomSearch(search_config, variation_strategy)

    # Initialize the population
    population = []
    
    def spawn_eval():
        nonlocal population

        if not population: # If it's the first generation, initialize the population
            population = []
            for _ in range(search_config.population_size):
                HIDDEN_LAYERS = np.random.randint(1, MAX_HIDDEN_LAYERS + 1)
                genome = custom_spawn_genome(HIDDEN_LAYERS, NUM_WEIGHTS)
                if print_genomes:
                    print(display_genome(genome))
                individual = Individual(genome, search_config.signature)
                # print(f"Initial individual: {display_genome(individual.genome)}") # FOR DEBUG
                population.append(individual)
        else:  # For subsequent generations, use the evolved population from CustomSearch.step()
            population = custom_search.step()

        custom_search.population = Population(population)

        # Evaluate each individual
        for individual in custom_search.population:
            architecture, weights = genome_extractor(individual.genome)
            error = fitness_eval(architecture, weights, X, y)
            if error is not None:
                individual.error_vector = np.array([error])
            else:
                individual.error_vector = None
            # print(f"Individual's Error: {individual.error_vector}") # to log error vector

    # Evolution loop
    for generation in range(search_config.max_generations):
        spawn_eval()
        # Print generation statistics
        print(f"\nGeneration {generation + 1}:")
        if len(custom_search.population) > 0:
            evaluated_individuals = [ind for ind in custom_search.population if ind.error_vector is not None]
            if evaluated_individuals:
                print(f"  {len(evaluated_individuals)}/{len(custom_search.population)} evaluated")
                print(f"  Median error: {np.median([ind.total_error for ind in evaluated_individuals]):.4f}")
                best_individual = min(evaluated_individuals, key=lambda ind: ind.total_error)
                best_arch, best_params = genome_extractor(best_individual.genome)
                print(f"  Best architecture: {input_size, best_arch, output_size}")
                print(f"    Error: {min(ind.total_error for ind in evaluated_individuals):.4f}\n")
                # print(f"  Best weights: {best_params[:5]}...")  # Print first 5 float params
            else:
                print("  No evaluated individuals in population!")
        else:
            print("  No individuals in population!")

    # Get the best individual from the final population
    if len(custom_search.population) > 0:
        evaluated_individuals = [ind for ind in custom_search.population if ind.error_vector is not None]
        if evaluated_individuals:
            best_individual = min(evaluated_individuals, key=lambda ind: ind.total_error)
            best_architecture, best_params = genome_extractor(best_individual.genome)
            print("\nBest architecture:")
            print(f"  Hidden layers: {best_architecture}")
            print(f"  Weights: {len(best_params)}")  # Print first 3 float params
            print(f"  Error: {best_individual.total_error:.4f}")
        else:
            print("\nNo evaluated individuals in final population!")
    else:
        print("\nNo individuals in final population!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()