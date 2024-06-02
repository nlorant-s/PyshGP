import gym
import numpy as np
import random
from deap import base, creator, tools, algorithms

# Create the environment
env = gym.make('Walker2d-v4')

# Define evaluation function
def evaluate(individual):
    observation = env.reset()
    total_reward = 0
    done = False

    print(f'Observation shape: {observation.shape}')

    # Reshape the individual's genome to form a weight matrix
    weights = np.array(individual).reshape((env.observation_space.shape[0], env.action_space.shape[0]))

    for _ in range(1000):  # Run the environment for 1000 steps
        # Generate the action using the weights
        action = np.dot(observation, weights)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward,

# Setup DEAP
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attribute', random.uniform, -1.0, 1.0)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attribute, n=(env.observation_space.shape[0] * env.action_space.shape[0]))
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('mate', tools.cxBlend, alpha=0.5)
toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('evaluate', evaluate)

# Genetic Algorithm parameters
population_size = 50
generations = 40
crossover_probability = 0.5
mutation_probability = 0.2

# Create initial population
population = toolbox.population(n=population_size)

# Run the Genetic Algorithm
for gen in range(generations):
    print(f"Generation {gen}")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_probability:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutation_probability:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the new offspring
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Replace population with offspring
    population[:] = offspring

# Print the best individual
best_ind = tools.selBest(population, 1)[0]
print(f'Best individual: {best_ind}')
print(f'Fitness: {best_ind.fitness.values[0]}')
