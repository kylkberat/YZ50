import math
import sys
input = sys.stdin.readline

n = int(input())
prev_act_list = list(map(float, input().split()))
weight_list = list(map(float, input().split()))
bias = float(input())

def sigmoid(x):
    return 1 / (1 + (math.exp(-x)))

def main():
    result = 0
    for a, b in zip(prev_act_list, weight_list):
        result += (a * b)
    result_with_bias = result + bias
    
    print(f"Weighted Sum without bias: {result:.4f}")
    print(f"Before Sigmoid Function: {result_with_bias:.4f}")
    print(f"Final Result: {sigmoid(result_with_bias):.4f}")

main()
