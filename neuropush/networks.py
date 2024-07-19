import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, flattened_weights):
        super(NeuralNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.layers = nn.ModuleList()
        
        index = 0
        for i in range(1, self.num_layers):
            weight_matrix_size = layer_sizes[i] * layer_sizes[i-1]
            bias_vector_size = layer_sizes[i]
            
            weight_matrix = torch.tensor(flattened_weights[index:index+weight_matrix_size], dtype=torch.float32).reshape(layer_sizes[i], layer_sizes[i-1])
            index += weight_matrix_size
            
            bias_vector = torch.tensor(flattened_weights[index:index+bias_vector_size], dtype=torch.float32)
            index += bias_vector_size
            
            layer = nn.Linear(layer_sizes[i-1], layer_sizes[i])
            layer.weight.data = weight_matrix
            layer.bias.data = bias_vector
            
            self.layers.append(layer)
            if i < self.num_layers - 1:
                self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Sigmoid())
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def predict(self, X):
        """
        Perform a feedforward operation through the network.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Binary predictions of shape (n_samples, 1).
        """
        try:
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self(X)
            
            # Ensure the output is of shape (n_samples, 1) and contains only 0 and 1
            binary_output = (output >= 0.5).cpu().numpy().astype(int)
            return binary_output.reshape(-1, 1)
        
        except Exception as e:
            print("predict() error:", e)
            return np.zeros((X.shape[0], 1))  # Return a zero array of correct shape instead of None

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
        linear_layer_index = 0
        for i in range(len(layer_sizes) - 1):
            while linear_layer_index < len(network.layers) and not isinstance(network.layers[linear_layer_index], nn.Linear):
                linear_layer_index += 1
            
            if linear_layer_index >= len(network.layers):
                print(f"Warning: Not enough linear layers for visualization. Expected {len(layer_sizes) - 1}, found {linear_layer_index}")
                break
            
            layer = network.layers[linear_layer_index]
            for j, start_pos in enumerate(node_positions[i]):
                for k, end_pos in enumerate(node_positions[i+1]):
                    if j < layer.weight.shape[1] and k < layer.weight.shape[0]:
                        weight = layer.weight.data[k, j].item()
                        color = 'red' if weight < 0 else 'green'
                        linewidth = 0.2 + abs(weight) # Scale linewidth for better visibility
                        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=color, lw=linewidth, alpha=0.6)
            
            linear_layer_index += 1
        
        # Set plot limits with some padding
        ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
        ax.set_ylim(-0.1, 1.1)
        
        # Ensure the aspect ratio is appropriate for the network structure
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    nn = NeuralNetwork([4, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    visualize_network(nn, display='show')
    print(nn.predict(np.array([[1, 2, 3, 4]])))