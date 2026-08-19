import math
import sys

prev_activations = [0.5, 0.8, 0.2]

weights = [
    [0.4, -0.6, 0.9],
    [0.1, 0.7, -0.3],
    [0.6, -1.3, 0.8]
]

biases = [0.1, -0.2, -0.75]

new_activations = []

def sigmoid(x):
    return 1 / (1 + (math.exp(-x)))

def single_neuron(activations, weight_list, bias):
    result = 0
    for weight, activation in zip(weight_list, activations):
        result += (weight * activation)
    result += bias
    return sigmoid(result)


def main():
    for i in range(len(weights)):
        new_activations.append(single_neuron(prev_activations, weights[i], biases[i]))
    
    print(new_activations)
    
    

main()