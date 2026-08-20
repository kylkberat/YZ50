import math
import matplotlib.pyplot as plt

prev_activations = [0.5, 0.8, 0.2]
weights = [
    [0.4, -0.6, 0.9],
    [0.1, 0.7, -0.3],
    [0.6, -1.3, 0.8]
]
biases = [0.1, -0.2, -0.75]
targets = [1.0, 0.0, 0.0]


def sigmoid(x):
    return 1 / (1 + (math.exp(-x)))

def single_neuron(activations, weight_list, bias):
    result = 0
    for weight, activation in zip(weight_list, activations):
        result += (weight * activation)
    result += bias
    return sigmoid(result)

def compute_loss(outputs, targets):
    result = 0
    for output, target in zip(outputs, targets):
        result += ((output - target) ** 2)
    return result

def forward_and_loss(weights, activations, biases, targets):
    new_activations = []
    for i in range(len(weights)):
        new_activations.append(single_neuron(activations, weights[i], biases[i]))
    loss = compute_loss(new_activations, targets)
    return loss

def main():
    losses = []
    learning_rate = 0.1
    
    for _ in range(100):
        loss1 = forward_and_loss(weights, prev_activations, biases, targets)
        losses.append(loss1)
        weight_gradients = []
        for row in weights:
            weight_gradients.append([0] * len(row))

        h = 0.0001
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                original = weights[i][j]
                weights[i][j] = original + h
                loss2 = forward_and_loss(weights, prev_activations, biases, targets)
                weights[i][j] = original
                weight_gradients[i][j] = (loss2 - loss1) / h

        for i in range(len(biases)):
            bias_gradients = [0] * len(biases)
            original = biases[i]
            biases[i] = original + h
            loss2 = forward_and_loss(weights, prev_activations, biases, targets)
            biases[i] = original
            bias_gradients[i] = (loss2 - loss1) / h

        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = weights[i][j] - (learning_rate * weight_gradients[i][j])

        for i in range(len(biases)):
            biases[i] = biases[i] - (learning_rate * bias_gradients[i])
        
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Loss over Gradient Descent Iterations")
    plt.savefig("gradient_descent_loss.png")
    plt.show()

main()