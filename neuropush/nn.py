import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
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
    """
    Visualize the neural network structure.

    Args:
        network (SimpleNeuralNetwork): Instance of the neural network.
        display (str): Whether to show the plot ('show') or hide it ('hide').
    """
    layer_sizes = network.layer_sizes
    fig, ax = plt.subplots()
    ax.axis('off')
    
    # Define node positions
    node_positions = []
    for i, layer_size in enumerate(layer_sizes):
        x = np.full(layer_size, i)
        y = np.linspace(0, 1, layer_size)
        node_positions.append(list(zip(x, y)))
    
    # Draw nodes
    for layer in node_positions:
        for (x, y) in layer:
            circle = plt.Circle((x, y), radius=0.075, edgecolor='k', facecolor='skyblue', lw=1)
            ax.add_patch(circle)
    
    # Draw connections
    for i in range(len(network.weights)):
        for j, start_pos in enumerate(node_positions[i]):
            for k, end_pos in enumerate(node_positions[i+1]):
                weight = network.weights[i][k, j]
                color = 'red' if weight < 0 else 'green'
                linewidth = abs(weight) * 1  # Scale line width by weight magnitude
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=color, lw=linewidth)
    if display=='show':
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

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
    visualize_network(network, 'show')