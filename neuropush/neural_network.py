import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, flattened_weights):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        
        # print(f"Initializing network with layer sizes: {layer_sizes}")
        # print(f"Total flattened weights: {len(flattened_weights)}")
        
        index = 0
        for i in range(1, self.num_layers):
            weight_matrix_size = layer_sizes[i] * layer_sizes[i-1]
            bias_vector_size = layer_sizes[i]
            
            # print(f"Layer {i}: Weight matrix size: {weight_matrix_size}, Bias vector size: {bias_vector_size}")
            
            # if index + weight_matrix_size > len(flattened_weights):
            #     raise ValueError(f"Not enough weights for layer {i}. Need {weight_matrix_size}, but only {len(flattened_weights) - index} left.")
            
            weight_matrix = np.array(flattened_weights[index:index+weight_matrix_size]).reshape(layer_sizes[i], layer_sizes[i-1])
            index += weight_matrix_size
            
            # if index + bias_vector_size > len(flattened_weights):
            #     raise ValueError(f"Not enough weights for bias in layer {i}. Need {bias_vector_size}, but only {len(flattened_weights) - index} left.")
            
            bias_vector = np.array(flattened_weights[index:index+bias_vector_size]).reshape(layer_sizes[i], 1)
            index += bias_vector_size
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
        
        # if index < len(flattened_weights):
        #     print(f"Warning: {len(flattened_weights) - index} unused weights")
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, z):
        return np.maximum(0, z)
    
    def predict(self, X):
        """
        Perform a feedforward operation through the network.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Output of the network with shape (n_samples, n_outputs).
        """
        a = X

        ''' No last layer sigmoid
        for weight, bias in zip(self.weights, self.biases):
            a = self.relu(np.dot(a, weight.T) + bias.T)
        '''
        try:
            for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                z = np.dot(a, weight.T) + bias.T
                if i == len(self.weights) - 1:  # Last layer
                    a = self.sigmoid(z)
                else:
                    a = self.relu(z)
            return a
        except Exception as e:
            print("predict() error:", e)
            return None

def visualize_network(network, display='hide'):
    if display == 'show':
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
        
        plt.tight_layout()
        plt.show()

input_size = 4
output_size = 1

if __name__ == '__main__':
    num_hidden_layers = np.random.randint(1, 5)  # Random number of hidden layers (1 to 4)
    hidden_layer_sizes = [np.random.randint(2, 10) for _ in range(num_hidden_layers)]  # Random size for each hidden layer
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]  # Input layer + hidden layers + output layer
    
    print("Layer Sizes:", layer_sizes)

    num_weights = sum(layer_sizes[i] * layer_sizes[i-1] + layer_sizes[i] for i in range(1, len(layer_sizes)))

    # Example flattened_weights: randomly initialized weights and biases
    flattened_weights = np.random.randn(num_weights).tolist() # WILL BE CHANGED TO FLOAT STACK

    # Initialize the network
    network = NeuralNetwork(layer_sizes, flattened_weights)

    # Test the predict function with 4 bit input
    input_data = np.random.randint(0, 2, (input_size, 1))
    output_data = network.predict(input_data)

    print("Input:\n", input_data)
    print("Output:\n", output_data)

    # Visualize the network
    visualize_network(network, 'show')