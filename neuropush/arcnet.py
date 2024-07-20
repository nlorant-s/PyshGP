import tensorflow as tf
import numpy as np
import random
import numpy as np
from copy import deepcopy

class ARCPush(tf.keras.Model):
    def __init__(self, max_grid_size=30, num_rules=5):
        super(ARCPush, self).__init__()
        self.max_grid_size = max_grid_size
        self.num_rules = num_rules

    def build(self, input_shape):
        # Encoding layers
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        
        # Improved attention mechanism
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        
        # Rule inference layers
        self.rule_dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.rule_dense2 = tf.keras.layers.Dense(self.num_rules, activation='softmax')
        
        # Decoding layers
        self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same')
        self.conv_transpose2 = tf.keras.layers.Conv2DTranspose(10, 3, activation=None, padding='same')

        super(ARCPush, self).build(input_shape)
    def call(self, inputs):
        # Ensure input is float32 and has correct shape
        x = tf.cast(inputs, tf.float32)
        input_shape = tf.shape(x)
        x = tf.ensure_shape(x, (None, None, None, 10))
        
        # Encoding
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Attention
        attention_output1 = self.mha1(x, x)
        x = tf.keras.layers.Add()([x, attention_output1])
        x = tf.keras.layers.LayerNormalization()(x)
        
        attention_output2 = self.mha2(x, x)
        x = tf.keras.layers.Add()([x, attention_output2])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Rule inference
        pooled = tf.keras.layers.GlobalAveragePooling2D()(x)
        rule_repr = self.rule_dense1(pooled)
        rule_weights = self.rule_dense2(rule_repr)
        
        # Apply rules (weighted sum)
        x = tf.einsum('bijk,bl->bijk', x, rule_weights)
        
        # Decoding (with dynamic shape handling)
        x = self.conv_transpose1(x)
        x = tf.image.resize(x, (input_shape[1], input_shape[2]))
        x = self.conv_transpose2(x)
        
        return x

def preprocess_arc_input(grid):
    # Convert grid to one-hot encoded tensor
    one_hot = tf.one_hot(grid, depth=10)
    return one_hot

class Genome:
    def __init__(self):
        self.max_grid_size = random.randint(20, 40)
        self.num_rules = random.randint(3, 10)
        self.conv1_filters = random.randint(16, 64)
        self.conv2_filters = random.randint(32, 128)
        self.mha_num_heads = random.randint(4, 16)
        self.rule_dense1_units = random.randint(128, 512)
        self.fitness = None

    def mutate(self, mutation_rate=0.1):
        if random.random() < mutation_rate:
            self.max_grid_size += random.randint(-2, 2)
        if random.random() < mutation_rate:
            self.num_rules += random.randint(-1, 1)
        if random.random() < mutation_rate:
            self.conv1_filters += random.randint(-4, 4)
        if random.random() < mutation_rate:
            self.conv2_filters += random.randint(-8, 8)
        if random.random() < mutation_rate:
            self.mha_num_heads += random.randint(-1, 1)
        if random.random() < mutation_rate:
            self.rule_dense1_units += random.randint(-16, 16)
        
        self.max_grid_size = max(20, min(40, self.max_grid_size))
        self.num_rules = max(3, min(10, self.num_rules))
        self.conv1_filters = max(16, min(64, self.conv1_filters))
        self.conv2_filters = max(32, min(128, self.conv2_filters))
        self.mha_num_heads = max(4, min(16, self.mha_num_heads))
        self.rule_dense1_units = max(128, min(512, self.rule_dense1_units))

def create_model_from_genome(genome):
    model = ARCPush(
        max_grid_size=genome.max_grid_size,
        num_rules=genome.num_rules
    )
    model.conv1 = tf.keras.layers.Conv2D(genome.conv1_filters, 3, activation='relu', padding='same')
    model.conv2 = tf.keras.layers.Conv2D(genome.conv2_filters, 3, activation='relu', padding='same')
    model.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=genome.mha_num_heads, key_dim=64)
    model.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=genome.mha_num_heads, key_dim=64)
    model.rule_dense1 = tf.keras.layers.Dense(genome.rule_dense1_units, activation='relu')
    return model

def crossover(parent1, parent2):
    child = Genome()
    for attr in vars(child):
        if attr != 'fitness':
            setattr(child, attr, getattr(random.choice([parent1, parent2]), attr))
    return child

def evaluate_fitness(model, test_cases):
    total_correct = 0
    total_cases = 0
    for input_grid, expected_output in test_cases:
        preprocessed_input = preprocess_arc_input(input_grid)
        preprocessed_input = tf.expand_dims(preprocessed_input, 0)
        output = model(preprocessed_input)
        predicted_output = tf.argmax(output, axis=-1).numpy()[0]
        total_correct += np.sum(predicted_output == expected_output)
        total_cases += np.prod(expected_output.shape)
    return total_correct / total_cases

def evolutionary_algorithm(population_size, generations, test_cases):
    population = [Genome() for _ in range(population_size)]
    
    for generation in range(generations):
        # Evaluate fitness
        for genome in population:
            model = create_model_from_genome(genome)
            genome.fitness = evaluate_fitness(model, test_cases)
        
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Print best fitness
        print(f"Generation {generation + 1}, Best Fitness: {population[0].fitness}")
        
        # Select top half as parents
        parents = population[:population_size // 2]
        
        # Create new population
        new_population = parents.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child.mutate()
            new_population.append(child)
        
        population = new_population
    
    return population[0]  # Return best genome

# Example usage
test_cases = [
    (np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]])),  # Simple +1 to all elements
    (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]])),  # Flip 0 and 1
    # Add more test cases here
]

best_genome = evolutionary_algorithm(population_size=100, generations=10, test_cases=test_cases)
best_model = create_model_from_genome(best_genome)

# Test the best model
for input_grid, expected_output in test_cases:
    preprocessed_input = preprocess_arc_input(input_grid)
    preprocessed_input = tf.expand_dims(preprocessed_input, 0)
    output = best_model(preprocessed_input)
    predicted_output = tf.argmax(output, axis=-1).numpy()[0]
    print("Input:")
    print(input_grid)
    print("Expected Output:")
    print(expected_output)
    print("Predicted Output:")
    print(predicted_output)
    print()

if False:
    # Example usage
    model = ARCPush()

    # Example input (3x3 grid with values 0-9)
    example_input = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    preprocessed_input = preprocess_arc_input(example_input)
    preprocessed_input = tf.expand_dims(preprocessed_input, 0)  # Add batch dimension

    output = model(preprocessed_input)
    print("Output shape for 3x3 input:", output.shape)  # Should be (1, 3, 3, 10)

    # Example input (5x4 grid with values 0-9)
    example_input2 = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 0, 1, 2],
        [3, 4, 5, 6],
        [7, 8, 9, 0]
    ])

    preprocessed_input2 = preprocess_arc_input(example_input2)
    preprocessed_input2 = tf.expand_dims(preprocessed_input2, 0)  # Add batch dimension

    output2 = model(preprocessed_input2)
    print("Output shape for 5x4 input:", output2.shape)  # Should be (1, 5, 4, 10)