import math

prev_activations = [0.5, 0.8, 0.2]
weights = [0.4, -0.6, 0.9]
bias = 0.1

def sigmoid(x):
    return 1 / (1 + (math.exp(-x)))

def main():
    result = 0
    for a, b in zip(prev_activations, weights):
        result += (a * b)
    result_with_bias = result + bias
    
    print(f"Weighted Sum without bias: {result:.4f}")
    print(f"Before Sigmoid Function: {result_with_bias:.4f}")
    print(f"Final Result: {sigmoid(result_with_bias):.4f}")

main()
