import numpy as np
import pymannkendall as mk

def check_monotonicity():
    pass

# Data generation for analysis
data = np.random.rand(360,1)

result = mk.original_test(data)
print(result)

if __name__ == '__main__':
    pass