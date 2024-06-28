from pyshgp.gp.search import GeneticAlgorithm
from pyshgp.gp.individual import Individual
from pyshgp.gp.genome import Genome, GeneSpawner
from pyshgp.gp.estimators import PushEstimator

import numpy as np
import torch
import torch.nn as nn

X = np.random.rand(10, 1)  # Example input data
y = 5 * X  # Example target data

class NeuralNetSearchAlgorithm(GeneticAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        self.all_genomes = []

    def step(self):
        super().step()
        # After each step, store the genomes of the current population
        self.all_genomes.extend([ind.genome for ind in self.population])


class NeuralNetPushEstimator(PushEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search = NeuralNetSearchAlgorithm(self.search.config)

    def fit(self, X, y):
        super().fit(X, y)
        return self.search.all_genomes
    
spawner = GeneSpawner(
    n_inputs=X.shape[1],
    instruction_set=['add_layer', 'activation'],
    literals=[nn.ReLU(), nn.Tanh(), nn.Sigmoid()],
    erc_generators=[
        lambda: np.random.randint(1, 100),  # for layer sizes
    ]
)
estimator = NeuralNetPushEstimator(
    spawner=spawner,
    population_size=100,
    max_generations=50,
    # ... other parameters ...
)

genomes = estimator.fit(X, y)

def genome_to_neural_net(genome: Genome):
    # This is a simplified example. You'll need to define how your Push
    # instructions translate to neural network structure.
    layers = []
    for gene in genome:
        if gene.name == 'add_layer':
            layers.append(nn.Linear(gene.input_size, gene.output_size))
        elif gene.name == 'activation':
            layers.append(gene.activation_function)
    return nn.Sequential(*layers)

def test_neural_network(net, X_test, y_test):
    input_data = torch.tensor(X_test, dtype=torch.float32)
    target_data = torch.tensor(y_test, dtype=torch.float32)
    output_data = net(input_data)
    loss = nn.MSELoss()(output_data, target_data)
    print("Test loss:", loss.item())

# Convert and test each genome
for genome in genomes:
    net = genome_to_neural_net(genome)
    # Test the neural network
    test_neural_network(net, X, y)
'''
class AddLayerMutation(VariationOperator):
    def produce(self, parents, spawner):
        parent = parents[0]
        new_genome = parent.copy()
        new_genome.append(spawner.random_instruction('add_layer'))
        return new_genome

class ChangeActivationMutation(VariationOperator):
    def produce(self, parents, spawner):
        parent = parents[0]
        new_genome = parent.copy()
        activation_indices = [i for i, gene in enumerate(new_genome) if gene.name == 'activation']
        if activation_indices:
            index = np.random.choice(activation_indices)
            new_genome[index] = spawner.random_instruction('activation')
        return new_genome

# Add these to your variation strategy
variation_strategy = VariationStrategy()
variation_strategy.add(AddLayerMutation(), 0.3)
variation_strategy.add(ChangeActivationMutation(), 0.3)
variation_strategy.add(Alternation(), 0.4)
'''