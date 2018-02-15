import evostrat
import numpy as np
from test_objectives import *


def es_test():
    """
    Runs some tests for the ESs
    """

    xo = np.random.uniform(0.3, 0.7, 10)
    # bounds = [(-5.0,5.0) for i in range(2) ] + [(0.01,5) for i in range(8)]
    test = evostrat.BasicES(xo, 0.1, verbose=False, objective=sphere)
    test(100)
    test.plot_cost_over_time()
    # test.save("test.es")

    # test = evostrat.BoundedRandNumTableES.load("test.es")
    # test(100, objective=sphere)
    # test.plot_cost_over_time()


if __name__ == '__main__':
    es_test()