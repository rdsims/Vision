import scipy.optimize as optimize
import math

def f(c):
    return math.sqrt(c[0]**2 + c[1]**2 + c[2]**2)

result = optimize.minimize(f, [1,1,1])
print(result)
print(result.x[1])