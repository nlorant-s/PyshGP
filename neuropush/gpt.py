from pyshgp.push.config import PushConfig
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.gp.genome import GeneSpawner
from pyshgp.gp.search import GeneticAlgorithm
from pyshgp.gp.evaluation import DatasetEvaluator
from pyshgp.gp.selection import Lexicase
from pyshgp.gp.variation import VariationPipeline, VariationOperator
from pyshgp.gp.estimators import PushEstimator
import pandas as pd
import numpy as np

class CustomPushEstimator(PushEstimator):
    def __init__(self, spawner, search='GA', selector='lexicase', variation_strategy='umad',
                 population_size=300, max_generations=100, initial_genome_size=(20, 100),
                 simplification_steps=2000, interpreter='default', parallelism=False, verbose=0, **kwargs):
        super().__init__(spawner, search, selector, variation_strategy, population_size,
                         max_generations, initial_genome_size, simplification_steps,
                         interpreter, parallelism, verbose, **kwargs)
        self.genomes = []

    def fit(self, X, y):
        evaluator = DatasetEvaluator(X, y, interpreter=self.interpreter)
        self.search_config = {
            # "signature": evaluator,
            "evaluator": evaluator,
            "spawner": self.spawner,
            "selection": self.selector,
            "variation": self.variation_strategy,
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "initial_genome_size": self.initial_genome_size,
            "simplification_steps": self.simplification_steps,
            "parallelism": self.parallelism,
        }

        self.search_algo = GeneticAlgorithm(config=self.search_config)
        self.search_algo.init_population()

        for generation in range(self.max_generations):
            self.search_algo.step()
            self.log_genomes(self.search_algo.population)

        best_individual = self.search_algo.best_seen
        self._program = best_individual.program

    def log_genomes(self, population):
        for individual in population.individuals:
            self.genomes.append(individual.genome)

# Usage Example
if __name__ == "__main__":
    # Define your problem-specific configurations
    X = pd.DataFrame(np.random.randn(100, 5))  # Example input data
    y = np.random.randint(2, size=100)  # Example target data

    spawner = GeneSpawner(n_inputs=5, instruction_set='core', literals=[], erc_generators=[])
    custom_estimator = CustomPushEstimator(spawner=spawner, search='GA', population_size=300)
    custom_estimator.fit(X, y)

    # Access the captured genomes
    genomes = custom_estimator.genomes
    for genome in genomes:
        print(genome)
