import gym
import numpy as np
import random

from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner
from pyshgp.gp.selection import Lexicase
from pyshgp.gp.variation import VariationStrategy, Alternation, Genesis
from pyshgp.gp.evaluation import DatasetEvaluator
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.gp.population import Population
from pyshgp.push.atoms import Input, Literal

def symbolic_regression():
    # Step 2: Define the Dataset
    # Create input-output pairs for the function f(x) = x^2
    X = np.array([[x] for x in range(-5, 5)])
    y = np.array([x**2 for x in range(-5, 5)]).reshape(-1, 1)
    testX = np.array([[x] for x in range(-5, 5)])
    testy = np.array([x**2 for x in range(-5, 5)]).reshape(-1, 1)

    # Step 3: Set Up the GeneSpawner
    spawner = GeneSpawner(
        n_inputs=1,
        instruction_set="core",
        literals=[],
        erc_generators=[lambda: random.randint(-5, 5)]
    )

    variation_strategy = VariationStrategy()
    variation_strategy.add(Alternation(alternation_rate=0.01, alignment_deviation=10), 0.7)
    variation_strategy.add(Genesis(size=(5, 20)), 0.3)

    # Step 4: Configure the PushEstimator
    est = PushEstimator(
        spawner=spawner,
        population_size=500,
        max_generations=100,
        initial_genome_size=(20, 100),
        selector="lexicase",
        variation_strategy=variation_strategy,
        verbose=True
    )

    train = True
    # Step 5: Run or load the GP Algorithm
    if train:
        print("Fitting GP Algorithm...")
        est.fit(X, y)
        est.save("push_estimator.pkl")
        print("Model saved")
    else:
        print("Loading GP Algorithm...")
        est.load("push_estimator.pkl")

    # Display the best program found
    best_program = est.solution.program
    print("Best program found:")
    print(best_program.pretty_str())

    print("Predictions:")
    print(est.predict(testX))
    print("Test errors:")
    print(est.score(testX, testy))

def mujoco():
    # Create the environment
    env = gym.make('Walker2d-v4')

    def evaluate_program(program):
        interpreter = PushInterpreter()
        observation = env.reset()
        total_reward = 0
        done = False

        while not done:
            for obs in observation:
                interpreter.push_input(obs)
            
            interpreter.run(program)
            
            action = np.array([interpreter.extract_output() for _ in range(env.action_space.shape[0])])
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            observation, reward, done, _ = env.step(action)
            total_reward += reward
        
        return total_reward

    dataset = [(None, None)]
    evaluator = DatasetEvaluator(evaluate_program, dataset)

    atom_generators = [
        Input(i) for i in range(env.observation_space.shape[0])
    ] + [
        Literal(0.0), Literal(1.0), Literal(-1.0), Literal(0.1), Literal(-0.1)
    ] + PushInterpreter

    spawner = GeneSpawner(atom_generators=atom_generators)

    selector = Lexicase("lexicase")

    population = Population(
        genome_size=(10, 100),
        evaluator=evaluator,
        spawner=spawner,
        selector=selector,
        pop_size=50
    )

    generations = 40
    for gen in range(generations):
        print(f"Generation {gen}")
        population.step()
        best_individual = population.best
        best_fitness = best_individual.fitness
        print(f"Best fitness: {best_fitness}")

    print("Evolution completed.")

mujoco()