import numpy as np
from neural_network import NeuralNetwork, visualize_network
from ast import literal_eval

X = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
              [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
              [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
              [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
y = np.array([[0], [1], [1], [0], [1], [0], [0], [1], [1], [0], [0], [1], [0], [1], [1], [0]])

def load_best_individual(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Find the last entry in the log file
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Best Individual"):
            start_index = i
            break
    
    # Parse the architecture, weights, and error vector
    architecture = list(lines[start_index + 4])
    print("arch:", architecture)
    weights = literal_eval(lines[start_index + 3].strip())
    
    # Handle the error vector parsing more carefully
    error_vector_str = lines[start_index + 4].strip()
    try:
        error_vector = literal_eval(error_vector_str)
    except:
        # If literal_eval fails, try to clean up the string and parse it manually
        error_vector_str = error_vector_str.replace('[', '').replace(']', '')
        error_vector = [float(x) for x in error_vector_str.split() if x]
    
    return architecture, weights, error_vector

def main():
    filename = 'logs.txt'
    
    print(f"Loading best individual from {filename}")
    
    architecture, weights, error_vector = load_best_individual(filename)
    
    print(f"Architecture: {architecture}")
    print(f"Number of weights: {len(weights)}")
    print(f"Error vector: {error_vector}")
    print(f"Mean error: {np.mean(error_vector)}")
    
    # Create the neural network
    network = NeuralNetwork(architecture, weights)
    
    # Visualize the network
    visualize_network(network, 'show')
    
    # Make predictions
    predictions = network.predict(X)
    
    # Calculate accuracy
    accuracy = np.mean((predictions > 0.5) == y)
    
    print(f"\nAccuracy on XOR dataset: {accuracy * 100}%")
    
    # Print detailed results
    print("\nDetailed results:")
    print("Input   | Target | Prediction | Error")
    print("------------------------------------")
    for i in range(len(X)):
        input_str = ' '.join(map(str, X[i]))
        target = y[i][0]
        prediction = predictions[i][0]
        error = error_vector[i]
        print(f"{input_str} | {target}      | {prediction}          | {error}")

if __name__ == "__main__":
    main()