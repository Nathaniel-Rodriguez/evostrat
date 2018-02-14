import evostrat
import numpy as np
from test_objectives import *


def ga_test():
    """
    Runs some tests for the GAs
    """
    xo = np.array([0.5 for i in range(10)])
    test = evostrat.BasicGA(xo, verbose=True, objective=sphere)
    test(100)


if __name__ == '__main__':
    ga_test()