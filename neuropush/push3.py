from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.push.type_library import PushTypeLibrary
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.gp.evaluation import DatasetEvaluator
import numpy as np

# Initialize PyshGP components
instruction_set = InstructionSet().register_core()
type_library = PushTypeLibrary(register_core=True)
spawner = GeneSpawner(
    n_inputs=1,
    instruction_set=instruction_set,
    literals=[],
    erc_generators=[]
)

# Generate some dummy data for this example
X = np.random.rand(100, 1)
y = X * 2 + 1 + np.random.normal(0, 0.1, (100, 1))

# Create the evaluator
evaluator = DatasetEvaluator(X, y)

# Create the PushEstimator with search configuration
estimator = PushEstimator(
    spawner=spawner,
    evaluator=evaluator,
    population_size=100,
    max_generations=50,
    initial_genome_size=(10, 50),
    selection="lexicase",
    variation="umad",
    error_threshold=0.1,  # This is now correctly placed in the search configuration
    verbose=2
)

# Fit the estimator (this will run the evolutionary process)
estimator.fit(X, y)

# After evolution, we can access the final population
final_population = estimator.search.population

# Print final population statistics
print(f"Final population size: {len(final_population)}")
print(f"Final median error: {final_population.median_error()}")
print(f"Final genome diversity: {final_population.genome_diversity()}")

# Get the best individual
best_individual = final_population.best()
print(f"Best individual error: {best_individual.total_error}")

# Interpret the best individual's program
interpreter = PushInterpreter(instruction_set=instruction_set)
program = best_individual.program
test_input = np.array([0.5])
output = interpreter.run(program, inputs=test_input.tolist())

print(f"Program output for input {test_input[0]}: {output}")

# Accessing the stacks after running the program
print("Int stack:", interpreter.state["int"])
print("Float stack:", interpreter.state["float"])
print("Bool stack:", interpreter.state["bool"])