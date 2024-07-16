from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.push.type_library import PushTypeLibrary
from pyshgp.push.atoms import Literal
from pyshgp.push.types import PushFloat, PushInt
from pyshgp.gp.individual import Individual
from pyshgp.gp.population import Population
from pyshgp.gp.search import SearchAlgorithm, SearchConfiguration
from pyshgp.gp.selection import Lexicase, Tournament
from pyshgp.gp.variation import LiteralMutation, VariationOperator, VariationStrategy, AdditionMutation, DeletionMutation, Alternation, Genesis, Cloning
from neural_network import NeuralNetwork, visualize_network
from neuromutations import FloatMutation, IntMutation, NullMutation, AdditionDeletionMutation
import numpy as np
import random
from datetime import datetime

# INITIALIZATION CONSTANTS
population_size = 100
max_generations = 3
MIN_LAYER_SIZE = 1
MAX_LAYER_SIZE = 16
MIN_WEIGHT_VALUE = -1.0
MAX_WEIGHT_VALUE = 1.0
MAX_HIDDEN_LAYERS = 3
NUM_WEIGHTS = MAX_LAYER_SIZE**2 * MAX_HIDDEN_LAYERS + MAX_LAYER_SIZE
TOTAL_GENES = MAX_HIDDEN_LAYERS + NUM_WEIGHTS
MIN_HIDDEN_LAYERS = 0
print_genomes = False
show_network = False
input_size = 4
output_size = 1

# XOR dataset
X = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
y = np.array([[0], [1], [1], [0], [1], [0], [0], [1],
                [1], [0], [0], [1], [0], [1], [1], [0]])

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

    def step(self):
        """
        Evolve the population by selecting parents and producing children.
        """
        # Filter out individuals with invalid error vectors
        valid_individuals = Population([ind for ind in self.population if np.isfinite(ind.error_vector).all() and len(ind.error_vector) > 0])

        # Elitism: Keep the best individual
        best_individual = min(valid_individuals, key=lambda ind: ind.total_error)

        # Adjust selection sizes based on the number of valid individuals
        total_size = len(self.population)
        random_size = int(total_size * 0.2)  # 20% random
        remaining_size = total_size # - random_size  # Leave room for the best individual?
        lexicase_size = remaining_size // 2  # Split remaining evenly between lexicase and tournament
        tournament_size = remaining_size - lexicase_size

        # downsize tournament size if necessary
        self.tournament_selector.tournament_size = min(self.tournament_selector.tournament_size, len(valid_individuals))

        parents_lexicase = self.lexicase_selector.select(valid_individuals, n=lexicase_size)
        parents_tournament = self.tournament_selector.select(valid_individuals, n=tournament_size)
        randomly_generated = [custom_spawn_genome(np.random.randint(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS + 1), NUM_WEIGHTS) for _ in range(random_size)]
        parents = parents_tournament + parents_lexicase # + randomly_generated

        print(f"{len(set(id(parent) for parent in parents))} unique parents, {len(parents)} selected")

        # Debugging to ensure selection with respect to error vectors
        # print("Selected parents error vectors:")
        # for parent in parents_lexicase[:3] + parents_tournament[:2]:  # Print a few from each selection method
        #     print(f"  {parent.error_vector:.4f}")
        
        children = [best_individual]
        if print_genomes:
            print(print_genome(best_individual.genome))
            print("Best individual")

        for _ in range(len(self.population) - 1):
            op = np.random.choice(self.variation_strategy.elements)
            num_parents = op.num_parents
            selected_parents = random.sample(parents, num_parents)
            child_genome = op.produce([p.genome for p in selected_parents], self.config.spawner)
            if print_genomes:
                print(print_genome(child_genome))
            child = Individual(child_genome, self.config.signature)
            children.append(child)
        self.population = Population(children)

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
            if gene.push_type == PushInt and len(layers):
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
    layers = ', '.join(int_values)
    weights = {float(value) for value in float_values}
    num_weights = len(weights)
    
    return f"\nLayers: {layers}\nweights: ({num_weights})"

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

            # for i in range(len(X)):
            #     prediction = network.predict(X[i])
            #     # Calculate mean squared error
            #     mse = np.mean((prediction - y[i]) ** 2)
            #     # Calculate binary cross-entropy loss
            #     bce = -np.mean(y[i] * np.log(prediction) + (1 - y[i]) * np.log(1 - prediction))

            predictions = network.predict(X)

            # Calculate mean squared error
            mse = np.mean((predictions - y) ** 2)
            # Calculate binary cross-entropy loss
            bce = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

            # Return as a 2d numpy array
            return np.array([bce])  # Return as a 2D numpy array with one element

        except Exception as e:
            # print(f"Error in fitness evaluation: {str(e)}")
            return None

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

    def variation_strategy(spawner):
        variation_strategy = VariationStrategy()
        
        # Fine-grained mutation operators
        # variation_strategy.add(AdditionDeletionMutation(addition_rate=0.1, deletion_rate=0.1), 0.000015)
        # variation_strategy.add(AdditionMutation(addition_rate=0.01), 0.25)
        # variation_strategy.add(DeletionMutation(deletion_rate=0.01), 0.25) # not a bad idea to track if an individual has been mutated
        variation_strategy.add(NullMutation(), 1)
        
        # Crossover operator
        # variation_strategy.add(Alternation(alternation_rate=0.1, alignment_deviation=5), 0.3)
        
        # Literal mutation (for fine-tuning weights and layers)
        # variation_strategy.add(FloatMutation(rate=0.5, std_dev=0.1), 0.2)
        # variation_strategy.add(IntMutation(rate=0.3, std_dev=1), 0.2)
        
        # Occasional genome reset (for maintaining diversity)
        # variation_strategy.add(Genesis(size=(TOTAL_GENES, TOTAL_GENES + 1)), 0.05)
        
        # Cloning (to preserve good solutions)
        # variation_strategy.add(Cloning(), 0.1)
        
        return variation_strategy

    # Create variation operator
    variation_strategy = variation_strategy(spawner)    

    # Create custom search
    custom_search = CustomSearch(search_config, variation_strategy)

    # Initialize the population
    population = []
    
    def spawn_eval():
        nonlocal population

        if not population: # If it's the first generation, initialize the population
            population = []
            for _ in range(search_config.population_size):
                HIDDEN_LAYERS = np.random.randint(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS + 1)
                # HIDDEN_LAYERS = 0
                genome = custom_spawn_genome(HIDDEN_LAYERS, NUM_WEIGHTS)
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
                print(f"  Evaluated: {len(evaluated_individuals)}/{len(custom_search.population)}")
                print(f"  Median error: {np.median([ind.total_error for ind in evaluated_individuals]):.2f}")
                best_individual = min(evaluated_individuals, key=lambda ind: ind.total_error)
                layers, _ = genome_extractor(best_individual.genome)
                print(f"  Best architecture: {input_size} {layers} {output_size}")
                print(f"    Error: {min(ind.total_error for ind in evaluated_individuals):.2f}\n")
            else:
                print("  No evaluated individuals in population!")
        else:
            print("  No individuals in population!")


if __name__ == '__main__':
    main()