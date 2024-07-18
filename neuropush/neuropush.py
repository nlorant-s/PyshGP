from pyshgp.gp.variation import VariationStrategy, AdditionMutation, DeletionMutation, Alternation, Cloning
from pyshgp.gp.search import SearchAlgorithm, SearchConfiguration
from neural_network import NeuralNetwork, visualize_network
from neuromutations import NullMutation, FloatReplacement, IntReplacement
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.gp.selection import Lexicase, Tournament
from pyshgp.push.type_library import PushTypeLibrary
from pyshgp.push.types import PushFloat, PushInt
from pyshgp.gp.individual import Individual
from pyshgp.gp.population import Population
from multiprocessing import freeze_support
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.atoms import Literal
from datetime import datetime
import numpy as np
import random

# INITIALIZATION CONSTANTS
population_size = 600
max_generations = 30
MIN_LAYER_SIZE = 1
MAX_LAYER_SIZE = 16
MIN_WEIGHT_VALUE = -1.0
MAX_WEIGHT_VALUE = 1.0
MAX_HIDDEN_LAYERS = 3
MIN_HIDDEN_LAYERS = 0
MAX_WEIGHTS = MAX_LAYER_SIZE**2 * MAX_HIDDEN_LAYERS + MAX_LAYER_SIZE
TOTAL_GENES = MAX_HIDDEN_LAYERS + MAX_WEIGHTS
print_genomes = False
show_network = False
input_size = 4
output_size = 1

bold = '\033[1m'
endbold = '\033[0m'

# XOR dataset
simple_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
simple_y = np.array([[0], [1], [1], [0]])

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
        if show_network:
            visualize_network(network, 'show')

        predictions = network.predict(X)

        # Calculate individual errors for each data point
        individual_errors = []
        for i in range(len(X)):
            # Calculate binary cross-entropy loss for each data point
            bce = -y[i] * np.log(predictions[i]) - (1 - y[i]) * np.log(1 - predictions[i])
            individual_errors.append(bce)

        # Return as a numpy array
        return np.array(individual_errors)

    except Exception as e:
        print(f"Error in fitness evaluation: {str(e)}")
        return None

def variation_strategy(spawner):
    variation_strategy = VariationStrategy()
    # variation_strategy.add(NullMutation(), 0.9)
    variation_strategy.add(IntReplacement(rate=0.5), 1)
    variation_strategy.add(FloatReplacement(rate=0.5), 1)
    
    return variation_strategy

class CustomSearch(SearchAlgorithm):
    def __init__(self, config, variation_strategy):
        """
        40% lexicase selection
        40% tournament selection
        20% random generations
        """
        super().__init__(config)
        self.variation_strategy = variation_strategy
        self.lexicase_selector = Lexicase(epsilon=True)
        self.tournament_selector = Tournament(tournament_size=7) # decrease tournament size for greater pressure
        self.num_gen = 2

    def step(self):
        """
        Evolve the population by selecting parents and producing children.
        """
        # Filter out individuals with invalid error vectors
        valid_individuals = Population([ind for ind in self.population if np.isfinite(ind.error_vector).all() and len(ind.error_vector) > 0])

        # Elitism: Keep the best 10%
        best_individuals = sorted(valid_individuals, key=lambda ind: ind.total_error)[:int(len(valid_individuals) * 0.1)]

        # Adjust selection sizes based on the number of valid individuals
        total_size = len(self.population) - len(best_individuals)
        random_size = int(total_size * 0.2)  # 20% random
        remaining_size = total_size # - random_size
        lexicase_size = remaining_size # // 2  # Split remaining evenly between lexicase and tournament
        tournament_size = remaining_size # - lexicase_size

        # downsize tournament size if necessary
        self.tournament_selector.tournament_size = min(self.tournament_selector.tournament_size, len(valid_individuals))

        parents_lexicase = self.lexicase_selector.select(valid_individuals, n=lexicase_size)
        parents_tournament = self.tournament_selector.select(valid_individuals, n=tournament_size)
        randomly_generated = [custom_spawn_genome(np.random.randint(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS + 1), MAX_WEIGHTS) for _ in range(random_size)]
        parents = parents_lexicase
        
        # print(f"  {len(parents)} parents selected, {len(set(id(parent) for parent in parents))} unique")

        # Debugging to ensure selection with respect to error vectors
        # print("Lexicase parents error vectors:")
        # for parent in parents:  # Print a few from each selection method
        #     print(f"  {print_genome(parent.genome)}")

        children = best_individuals

        if print_genomes:
            print(f"\n{bold}GENERATION {self.num_gen}{endbold}")
            self.num_gen += 1
            print("---------------------------------")

            for ind in best_individuals:
                print("*last generation best individual")
                print(print_genome(ind.genome))

        for _ in range(len(self.population) - len(best_individuals)):
            op = np.random.choice(self.variation_strategy.elements)
            num_parents = op.num_parents
            selected_parents = random.sample(parents, num_parents) # is this problematic? is it needed?
            # print(f"  {len(selected_parents)} parents selected, {len(set(id(selected_parent) for selected_parent in parents))} unique")
            child_genome = op.produce([p.genome for p in parents], self.config.spawner) # not using selected_parents
            if print_genomes:
                print(print_genome(child_genome))
            child = Individual(child_genome, self.config.signature)
            children.append(child)

        self.population = Population(children)

        for child in self.population:
            architecture, weights = genome_extractor(child.genome)
            error = fitness_eval(architecture, weights, X, y)
            if error is not None:
                child.error_vector = error
            else:
                child.error_vector = None

        return self.population
    
# Define ERC (Ephemeral Random Constant) generators
def weights_generator():
    return Literal(value=random.uniform(MIN_WEIGHT_VALUE, MAX_WEIGHT_VALUE), push_type=PushFloat)

def layer_size_generator():
    return Literal(value=random.randint(MIN_LAYER_SIZE, MAX_LAYER_SIZE), push_type=PushInt)

# generates genome with a specified number of ints for layers and floats for weights
def custom_spawn_genome(num_layers, num_weights):
    genome = []
    for _ in range(num_layers):
        genome.append(layer_size_generator())
    for _ in range(num_weights):
        genome.append(weights_generator())

    return genome

# transforms genome into a list of layer sizes and a list of weights
def genome_extractor(genome):
    layers = []
    weights = []
    for gene in genome:
        if isinstance(gene, Literal):
            if gene.push_type == PushInt:
                layers.append(gene.value)
            elif gene.push_type == PushFloat:
                weights.append(gene.value)
        else:
            print(f"Error: Unexpected gene type {type(gene)}")

    return layers, weights

# Display the genomes architecture and number of weights in the terminal
def print_genome(genome):
    int_values = list(str(input_size))
    float_values = []
    for gene in genome:
        if isinstance(gene, Literal):
            if gene.push_type == PushInt:
                int_values.append(str(gene.value))
            elif gene.push_type == PushFloat:
                float_values.append(f"{gene.value:.4f}")
    int_values.append(str(output_size))
    layers = '-'.join(int_values)
    weights = [float(value) for value in float_values]
    num_weights = len(weights)
    
    return f"{layers}\n{num_weights}\n"

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

def genome_to_hashable(genome):
    return tuple((gene.push_type, gene.value) for gene in genome if isinstance(gene, Literal))

def main():
    print(f"{bold}Beginning evolution{endbold}")
    print(f"Population size: {population_size}, generations: {max_generations}\n")
    # Initialize PyshGP components
    instruction_set = InstructionSet().register_core()
    type_library = PushTypeLibrary(register_core=False)
    type_library.register(PushInt)
    instruction_set = InstructionSet(type_library=type_library)

    spawner = GeneSpawner(
        n_inputs=0,
        instruction_set=instruction_set,
        literals=[],
        erc_generators=[layer_size_generator, weights_generator],
    )

    try:
        # Create variation operator
        global variation_strategy
        variation_strategy = variation_strategy(spawner)    

        # Create search configuration
        search_config = SearchConfiguration(
            signature=None,  # not used atm
            evaluator=None,  # not used atm
            spawner=spawner,
            population_size=population_size,
            max_generations=max_generations,
            initial_genome_size=(TOTAL_GENES, TOTAL_GENES + 1),
            simplification_steps=0,
            error_threshold=0.0,
            selection="lexicase",
        )

        # Create custom search
        custom_search = CustomSearch(search_config, variation_strategy)
    except Exception as e:
        print(f"Error in CustomSearch initialization: {str(e)}")

    # Initialize the population
    population = []

    def evolve():
        nonlocal population
        if not population: # If it's the first generation, initialize the population
            population = []
            if print_genomes:
                print(f"{bold}GENERATION 1{endbold}")
                print("---------------------------------")
            for _ in range(search_config.population_size):
                HIDDEN_LAYERS = np.random.randint(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS + 1)
                # HIDDEN_LAYERS = 0
                genome = custom_spawn_genome(HIDDEN_LAYERS, MAX_WEIGHTS)
                if print_genomes:
                    print(print_genome(genome))
                individual = Individual(genome, search_config.signature)
                # print(f"Initial individual: {display_genome(individual.genome)}") # FOR DEBUG
                population.append(individual)
        else:  # For subsequent generations, use the evolved population from CustomSearch.step()
            population = custom_search.step()

        custom_search.population = Population(population)
        # Evaluate each individual
        for individual in custom_search.population:
            architecture, weights = genome_extractor(individual.genome)
            # print(f"Architecture: {architecture}")
            error = fitness_eval(architecture, weights, X, y)
            if error is not None:
                individual.error_vector = error
            else:
                individual.error_vector = None
            # to log error vector
    
    # Evolution loop
    for generation in range(search_config.max_generations):
        try:
            evolve()
            # Print generation statistics
            print(f"{bold}Generation {generation + 1}:{endbold}")
            if len(custom_search.population) > 0:
                evaluated_individuals = [ind for ind in custom_search.population if ind.error_vector is not None]
                if evaluated_individuals:
                    # Calculate diversity
                    unique_genomes = set(genome_to_hashable(ind.genome) for ind in evaluated_individuals)
                    diversity = len(unique_genomes)
                    print(f"  Diversity: {diversity}/{len(evaluated_individuals)}")
                    print(f"  Mdn error: {bold}{np.median([np.mean(ind.error_vector) for ind in evaluated_individuals]):.2f}{endbold}")
                    best_individual = min(evaluated_individuals, key=lambda ind: np.mean(ind.error_vector))
                    layers, _ = genome_extractor(best_individual.genome)
                    best_individuals = sorted(evaluated_individuals, key=lambda ind: ind.total_error)[:int(len(evaluated_individuals) * 0.1)]
                    print(f"  Elite M error: {bold}{np.mean([np.mean(ind.error_vector) for ind in best_individuals]):.2f}{endbold}")
                    print(f"  Best individual: {input_size, layers, output_size}")
                    print(f"    M error: {bold}{np.mean(best_individual.error_vector):.2f}{endbold}\n")
                else:
                    print("  No evaluated individuals in population!")
            else:
                print("  No individuals in population!")
        except Exception as e:
            print(f"Error in generation {generation + 1}: {str(e)}")
            continue

if __name__ == '__main__':
    freeze_support()
    main()