import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork: # needs to return score 0-1
    def __init__(self, layer_sizes, flattened_weights):
        """
        Initialize the neural network with given layer sizes and flattened weights.

        Args:
            layer_sizes (list): List of integers representing the number of neurons in each layer.
            flattened_weights (list): List of weights and biases flattened into a single list.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        
        # Unflatten weights and biases from the flat_weights list
        index = 0
        for i in range(1, self.num_layers):
            weight_matrix_size = layer_sizes[i] * layer_sizes[i-1]
            bias_vector_size = layer_sizes[i]
            
            weight_matrix = np.array(flattened_weights[index:index+weight_matrix_size]).reshape(layer_sizes[i], layer_sizes[i-1])
            index += weight_matrix_size
            
            bias_vector = np.array(flattened_weights[index:index+bias_vector_size]).reshape(layer_sizes[i], 1)
            index += bias_vector_size
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def relu(self, z):
        """
        Apply ReLU activation function.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying ReLU.
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        Compute the derivative of ReLU activation function.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Derivative of ReLU.
        """
        return (z > 0).astype(float)
    
    def feedforward(self, a):
        """
        Perform a feedforward operation through the network.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output of the network.
        """
        for weight, bias in zip(self.weights, self.biases):
            a = self.relu(np.dot(weight, a) + bias)
        return a

def visualize_network(network, display='hide'):
    layer_sizes = network.layer_sizes
    fig, ax = plt.subplots()
    
    # Set subplot parameters to remove padding
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    
    ax.axis('off')
    
    # Calculate the maximum layer size for scaling
    max_layer_size = max(layer_sizes)
    
    node_positions = []
    for i, layer_size in enumerate(layer_sizes):
        x = np.full(layer_size, i)
        y = np.linspace(0, 1, layer_size) * (layer_size / max_layer_size)
        y += (1 - (layer_size / max_layer_size)) / 2  # Center the layer vertically
        node_positions.append(list(zip(x, y)))
    
    # Draw nodes
    for layer in node_positions:
        for (x, y) in layer:
            circle = plt.Circle((x, y), radius=0.04, edgecolor='k', facecolor='k', lw=1)
            ax.add_patch(circle)
    
    # Draw connections
    for i in range(len(network.weights)):
        for j, start_pos in enumerate(node_positions[i]):
            for k, end_pos in enumerate(node_positions[i+1]):
                weight = network.weights[i][k, j]
                color = 'red' if weight < 0 else 'green'
                linewidth = 0.2 + abs(weight) # Scale linewidth for better visibility
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=color, lw=linewidth, alpha=0.6)
    
    # Set plot limits with some padding
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    ax.set_ylim(-0.1, 1.1)
    
    # Ensure the aspect ratio is appropriate for the network structure
    ax.set_aspect('equal', adjustable='box')
    
    if display == 'show':
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Define the network structure | this will change to be a list of the individials int stack
    # Define the input size
    input_size = 3

    # Generate random layer sizes
    num_hidden_layers = np.random.randint(1, 5)  # Random number of hidden layers (1 to 4)
    hidden_layer_sizes = [np.random.randint(2, 10) for _ in range(num_hidden_layers)]  # Random size for each hidden layer
    layer_sizes = [input_size] + hidden_layer_sizes + [1]  # Input layer + hidden layers + output layer
    
    print("Layer Sizes:", layer_sizes)

    # Calculate the total number of weights and biases needed
    num_weights = sum(layer_sizes[i] * layer_sizes[i-1] + layer_sizes[i] for i in range(1, len(layer_sizes)))

    # Example flattened_weights: randomly initialized weights and biases
    flattened_weights = np.random.randn(num_weights).tolist() # WILL BE CHANGED TO FLOAT STACK

    # Initialize the network
    network = SimpleNeuralNetwork(layer_sizes, flattened_weights)

    # Test the feedforward function with a sample input
    input_data = np.random.randn(3, 1)  # to change
    output_data = network.feedforward(input_data)

    print("Input:\n", input_data)
    print("Output:\n", output_data)

    # Visualize the network
    visualize_network(network, 'show')