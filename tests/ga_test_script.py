import evostrat
from test_objectives import *


def ga_test():
    """
    Runs some tests for the GAs
    """

    test = evostrat.BasicGA(evostrat.real_member_generator,
                            evostrat.real_mutator,
                            {'size':5,'bounds':[0.3,0.7]}, {'scale':0.005},
                            elite_fraction=0.125,
                            parent_fraction=0.375,
                            seed=1,
                            verbose=True, objective=sphere)
    test(5)
    test.plot_cost_over_time()
    # test.save("test.basicga")

    # test = evostrat.BasicGA.load("test.basicga")
    # test.assign_member_generating_function(evostrat.real_member_generator,
    #                                        {'size': 10, 'bounds': [-1, 1]})
    # test.assign_mutation_function(evostrat.real_mutator, {'scale':0.01})
    # test(100, objective=sphere)
    # test.plot_cost_over_time()


if __name__ == '__main__':
    ga_test()