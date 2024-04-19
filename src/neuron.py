import math

def gelu(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
