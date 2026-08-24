import math

activation_list = [0.5, 0.5866175789173301, 0.2091593652130633]
targets = [1.0, 0.0, 0.0]

def compute_loss(outputs, targets):
    result = 0
    for output, target in zip(outputs, targets):
        result += ((output - target) ** 2)
    return result

def main():
    print(compute_loss(activation_list, targets))   


main()