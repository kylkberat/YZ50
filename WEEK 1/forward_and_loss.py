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
    loses = []
    w_values = []
    for i in range(-20, 21):
        w = i * 0.1
        weights[0][0] = w
        loss = forward_and_loss(weights, prev_activations, biases, targets)
        w_values.append(w)
        loses.append(loss)

    plt.plot(w_values, loses)
    plt.xlabel("weight[0][0] value")
    plt.ylabel("loss")
    plt.title("Loss vs Weight")
    plt.savefig("loss_curve.png")
    plt.show()

"""
Deney Notu: Tek katmanlı ağda her parametrenin loss eğrisi monoton + plato çıkıyor;
eğim yönü hedef değerine göre değişiyor. Parabol için çelişen etkiler gerekiyor.

"""

main()