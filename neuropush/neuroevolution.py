import numpy as np
import torch
import torch.nn as nn
from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner
from pyshgp.gp.selection import Lexicase
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.gp.population import Population
from pyshgp.gp.evaluation import Evaluator

# Define the function to parse PushGP stacks
def parse_pushgp_stack(stack):
    layer_sizes = []
    weights = []
    connections = []

    i = 0
    while i < len(stack):
        if isinstance(stack[i], int):
            layer_sizes.append(stack[i])
        elif isinstance(stack[i], float):
            weights.append(stack[i])
        elif isinstance(stack[i], bool):
            connections.append(stack[i])
        i += 1

    return layer_sizes, weights, connections

# Define the custom neural network class
class CustomNetwork(nn.Module):
    def __init__(self, layer_sizes, weights, connections):
        super(CustomNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.weights = weights
        self.connections = connections

        # Create layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Initialize weights and connections
        self._initialize_weights_and_connections()

    def _initialize_weights_and_connections(self):
        weight_index = 0
        connection_index = 0
        for layer in self.layers:
            with torch.no_grad():
                for i in range(layer.weight.size(0)):
                    for j in range(layer.weight.size(1)):
                        if connection_index < len(self.connections) and self.connections[connection_index]:
                            layer.weight[i, j] = self.weights[weight_index]
                            weight_index += 1
                        else:
                            layer.weight[i, j] = 0
                        connection_index += 1

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

# Define the function to evaluate an individual
def evaluate_individual(individual, X, y):
    interpreter = PushInterpreter()
    interpreter.execute(individual.program, inputs=[X[0]])

    int_stack = interpreter.state.int_stack
    float_stack = interpreter.state.float_stack
    bool_stack = interpreter.state.bool_stack

    # Combine all stacks into a single list
    pushgp_stack = int_stack + float_stack + bool_stack

    # Parse the stack to get layer sizes, weights, and connections
    layer_sizes, weights, connections = parse_pushgp_stack(pushgp_stack)

    # If the parsed stack is invalid, return a high error
    if len(layer_sizes) < 2:
        return float('inf')

    try:
        # Construct the network
        network = CustomNetwork(layer_sizes, weights, connections)

        # Evaluate the network
        input_data = torch.tensor(X, dtype=torch.float32)
        target_data = torch.tensor(y, dtype=torch.float32)
        output_data = network(input_data)

        # Compute the loss (mean squared error)
        loss = nn.MSELoss()(output_data, target_data)
        return loss.item()
    except Exception as e:
        # If there is an error in creating or evaluating the network, return a high error
        print(f"Error evaluating individual: {e}")
        return float('inf')

# Define the custom fitness function
def fitness(individual):
    return evaluate_individual(individual, X, y)

# Example data
X = np.random.rand(10, 1)  # Example input data
y = np.random.rand(10, 1)  # Example target data

# Define spawner and lexicase selector
spawner = GeneSpawner(
    n_inputs=1,
    instruction_set="core",
    literals=[],
    erc_generators=[lambda: np.random.uniform(-1, 1)]
)

lexicase = Lexicase(epsilon=True)

# Initialize the PushEstimator
est = PushEstimator(
    spawner=spawner,
    population_size=30,
    max_generations=10,
    verbose=2,
    selector=lexicase,
    fitness_function=fitness
)

# Fit the model
est.fit(X, y)

# Access the evaluated population
population = Population(individuals=est.population_size)

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
