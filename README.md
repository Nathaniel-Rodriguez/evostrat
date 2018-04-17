A package with evolutionary strategy and genetic algorithms for Python3.4+.

Requires [mpi4py](https://pypi.org/project/mpi4py/) to be installed. Suggest installing it prior so that it is configured properly with your installation of MPI (else it will be installed automatically when installing dependencies).

The code uses modified versions of the core algorithms for the evolutionary strategies come from [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf). While the core algorithms for the genetic algorithms comes from [Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/abs/1712.06567). The versions in `evostrat.py` and `genalg.py` were modified to improve performance and be easily distributed over MPI using mpi4py. These algorithms were tested out to around 6 million parameters and take-up virtually none of the wall-time, which is almost entirely loaded on the objective function. (e.g. a 1 million parameter run with a simple objective function takes only 2.5 minutes for 100 iterations for over 1000 ranks on BR2).

An example evolutionary run may look like:

```python
from evostrat import BoundedRandNumTableES

# <insert code for guess, bounds and objective>

es = BoundedRandNumTableES(xo=initial_guess,
                           step_size=1.0,
                           bounds=bounds,
                           objective=objective_function,
                           seed=1,
                           verbose=True,
                           rand_num_table_size=20000000)
es.run(num_iterations=150)
es.save("test_run.es")
```

Scripts should be run through MPI like so:

```bash
mpiexec -n 1000 python es_run.py
```

Runs are pickled and saved (except for the objective). They can be reloaded and ran further like so:

```python
es = BoundedRandNumTableES.load("test_run.es")
es(num_iterations=50, objective=objective_function)
es.save("test_run.es")
```

When running outside of MPI, the output can be plotted and the best members can be printed:

```python
es = BoundedRandNumTableES.load("test_run.es")
es.plot_cost_over_time("test", savefile=True, logy=False)
print(es.best)
print(es.centroid)
```

See additional documentation via `help` in python for parameters and usage of the various algorithms. 