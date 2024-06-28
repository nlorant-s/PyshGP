from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner, Genome
from pyshgp.gp.selection import Lexicase
from pyshgp.gp.individual import Individual # .genome()
from pyshgp.gp.search import SearchAlgorithm, GeneticAlgorithm # SearchAlgorithm.step()
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.gp.population import Population
from pyshgp.gp.evaluation import Evaluator

from nn import SimpleNeuralNetwork, visualize_network
import numpy as np
import torch
import torch.nn as nn

# Example data
X = np.random.rand(10, 1)  # Example input data
y = 5 * X  # Example target data

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
    population_size=120,
    max_generations=6,
    verbose=2,
    selector=lexicase,
)

# Fit the model
# est.fit(X, y)
print(Individual().genome())
