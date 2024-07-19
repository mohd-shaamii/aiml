import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from math import ceil, pi

def lowess(x, y, f, iterations):
    n = len(x) 
    r = int(ceil(f * n))  
    yest = np.zeros(n)  
    delta = np.ones(n)

    for _ in range(iterations): 
        for i in range(n):
            dist = np.abs(x - x[i])
            h = np.partition(dist, r)[r]  
            weights = np.clip(dist / h, 0.0, 1.0)  
            weights = (1 - weights ** 3) ** 3 

            b = np.array([np.sum(weights * y), np.sum(weights *y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)], [np.sum(weights * x), np.sum(weights * x * x)]])

            beta = solve(A, b) 
            yest[i] = beta[0] + beta[1] * x[i] 

    return yest

def main():
    x = np.linspace(0, 2 * pi, 100) 
    y = np.sin(x) + 0.3 * np.random.randn(len(x)) 
    yest = lowess(x, y,0.25,3) 
    plt.plot(x, y, "r.")
    plt.plot(x, yest, "b-")
    plt.show()  
    
main()
