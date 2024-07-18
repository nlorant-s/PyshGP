import numpy as np
from neural_network import NeuralNetwork, visualize_network

# XOR dataset
X = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
              [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
              [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
              [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
y = np.array([[0], [1], [1], [0], [1], [0], [0], [1],
              [1], [0], [0], [1], [0], [1], [1], [0]])

def load_best_individual(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Find the last entry in the log file
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Best Individual"):
            start_index = i
            break
    
    layers = eval(lines[start_index + 2].strip())
    weights = eval(lines[start_index + 3].strip())
    
    return layers, weights

def main():
    filename = 'logs.txt'
    
    print(f"Loading best individual from {filename}")
    
    layers, weights = load_best_individual(filename)
    
    network = NeuralNetwork(layers, weights)
    
    # Visualize the network
    visualize_network(network, 'show')
    
    # Make predictions
    predictions = network.predict(X)
    
    # Calculate accuracy
    accuracy = np.mean((predictions > 0.5) == y)
    
    print(f"Accuracy on XOR dataset: {accuracy * 100:.2f}%")
    
    # Print detailed results
    print("\nDetailed results:")
    print("Input | Target | Prediction")
    print("--------------------------")
    for i in range(len(X)):
        input_str = ' '.join(map(str, X[i]))
        target = y[i][0]
        prediction = predictions[i][0]
        print(f"{input_str} | {target}      | {prediction:.4f}")

if __name__ == "__main__":
    main()