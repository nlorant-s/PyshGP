from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner
from pyshgp.gp.selection import Lexicase
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.gp.population import Population
from pyshgp.gp.evaluation import Evaluator

from nn import SimpleNeuralNetwork, visualize_network
import numpy as np

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
    population_size=100,
    max_generations=5,
    verbose=2,
    selector=lexicase,
)

# Fit the model
est.fit(X, y)

'''
if __name__ == '__main__':
    # Define the network structure
    layer_sizes = [3, 2, 1] # to change

    # Calculate the total number of weights and biases needed
    total_weights = sum(layer_sizes[i] * layer_sizes[i-1] + layer_sizes[i] for i in range(1, len(layer_sizes)))
    # Example flattened_weights: randomly initialized weights and biases
    flattened_weights = np.random.randn(total_weights).tolist() # to change

    # Initialize the network
    network = SimpleNeuralNetwork(layer_sizes, flattened_weights)

    # Test the feedforward function with a sample input
    input_data = np.random.randn(3, 1)  # to change
    output_data = network.feedforward(input_data)

    print("Input Data:\n", input_data)
    print("Output Data:\n", output_data)

    # Visualize the network
    visualize_network(network, 'hide')
'''