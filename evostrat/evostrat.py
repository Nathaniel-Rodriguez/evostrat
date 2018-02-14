import numpy as np
import pickle
from functools import partial
from .sliceops import *


class BasicES:

    def __init__(self, xo, step_size, **kwargs):
        """
        xo: initial centroid
        step_size: float. the size of mutations
        num_mutations: number of dims to mutate in each iter (defaults to #dim)
        verbose: True/False. print info on run

        """
        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        # User input parameters
        self.objective = kwargs.get('objective', None)
        self.obj_args = kwargs.get('obj_args', ())
        self._step_size = np.float32(step_size)
        self._num_parameters = len(xo)
        self._num_mutations = kwargs.get('num_mutations', self._num_parameters)
        self._num_parents = kwargs.get('num_parents', int(self._size / 2))
        self._verbose = kwargs.get('verbose', False)
        self._global_seed = kwargs.get('seed', 1)

        # Internal parameters
        self._weights = np.log(self._num_parents + 0.5) - np.log(
            np.arange(1, self._num_parents + 1))
        self._weights /= np.sum(self._weights)
        self._weights.astype(np.float32, copy=False)

        # Internal data
        self._global_rng = np.random.RandomState(self._global_seed)
        self._seed_set = self._global_rng.choice(1000000, size=self._size,
                                                 replace=False)
        self._worker_rngs = [np.random.RandomState(seed)
                             for seed in self._seed_set]
        self._par_choices = np.arange(0, self._num_parameters, dtype=np.int32)
        self._generation_number = 0
        self._score_history = []

        # State
        self._centroid = np.array(xo, dtype=np.float32)
        self._old_centroid = self._centroid.copy()

    def __call__(self, num_iterations, objective=None, args=()):

        if (self.objective is not None) and (objective is None):
            objective = self.objective
            args = self.obj_args
        elif (self.objective is None) and (objective is not None):
            raise AttributeError("Error: No objective defined")

        partial_objective = partial(objective, *args)
        for i in range(num_iterations):
            if self._verbose and (self._rank == 0):
                print("Generation:", self._generation_number)
            self._update(partial_objective)
            self._generation_number += 1

    def _update(self, objective):

        # Perturb centroid
        perturbed_dimensions = self._global_rng.choice(self._par_choices,
                                                       size=self._num_mutations,
                                                       replace=False)
        local_perturbation = self._worker_rngs[self._rank].randn(
            self._num_mutations)
        local_perturbation *= self._step_size
        self._centroid[perturbed_dimensions] += local_perturbation

        # Run objective
        local_cost = np.empty(1, dtype=np.float32)
        local_cost[0] = objective(self._centroid)

        # Consolidate return values
        all_costs = np.empty(self._size, dtype=np.float32)
        self._comm.Allgather([local_cost, self._MPI.FLOAT],
                             [all_costs, self._MPI.FLOAT])
        self._update_log(all_costs)

        self._update_centroid(all_costs, local_perturbation, perturbed_dimensions)

    def _update_centroid(self, all_costs, local_perturbation, perturbed_dimensions):

        for parent_num, rank in enumerate(np.argsort(all_costs)):
            if parent_num < self._num_parents:
                if rank == self._rank:
                    local_perturbation *= self._weights[parent_num]
                    self._old_centroid[perturbed_dimensions] += local_perturbation
                else:
                    perturbation = self._worker_rngs[rank].randn(
                        self._num_mutations)
                    perturbation *= self._weights[parent_num] * self._step_size
                    self._old_centroid[perturbed_dimensions] += perturbation
            else:
                if rank != self._rank:
                    self._worker_rngs[rank].randn(self._num_mutations)

        self._centroid[perturbed_dimensions] = self._old_centroid[perturbed_dimensions]

    def _update_log(self, costs):

        self._score_history.append(costs)

    def get_centroid(self):

        return self._centroid.copy()

    def plot_cost_over_time(self, prefix='test', logy=True, savefile=False):
        """
        Plots the evolutionary history of the population's cost.
        Includes min cost individual for each generation the mean
        """
        if self._rank == 0:
            import matplotlib.pyplot as plt

            costs_by_generation = np.array(self._score_history)
            min_cost_by_generation = \
                np.min(costs_by_generation, axis=1)
            mean_cost_by_generation = \
                np.mean(costs_by_generation, axis=1)

            plt.plot(range(len(mean_cost_by_generation)),
                     mean_cost_by_generation,
                     marker='None', ls='-', color='blue', label='mean cost')

            plt.plot(range(len(min_cost_by_generation)),
                     min_cost_by_generation, ls='--', marker='None',
                     color='red', label='best')
            if logy:
                plt.yscale('log')
            plt.grid(True)
            plt.xlabel('generation')
            plt.ylabel('cost')
            plt.legend(loc='upper right')
            plt.tight_layout()

            if savefile:
                plt.savefig(prefix + "_evocost.png", dpi=300)
                plt.close()
                plt.clf()
            else:
                plt.show()
                plt.clf()

    @classmethod
    def load(cls, filename):
        pickled_obj_file = open(filename, 'rb')
        obj = pickle.load(pickled_obj_file)
        pickled_obj_file.close()

        return obj

    def save(self, filename):
        """
        objectives and their args are not saved with the ES
        """
        if self._rank == 0:
            pickled_obj_file = open(filename, 'wb')
            pickle.dump(self, pickled_obj_file, 2)
            pickled_obj_file.close()

    def __getstate__(self):

        state = {"_step_size": self._step_size,
                 "_num_parameters": self._num_parameters,
                 "_num_mutations": self._num_mutations,
                 "_num_parents": self._num_parents,
                 "_verbose": self._verbose,
                 "_global_seed": self._global_seed,
                 "_weights": self._weights,
                 "_global_rng": self._global_rng,
                 "_seed_set": self._seed_set,
                 "_worker_rngs": self._worker_rngs,
                 "_generation_number": self._generation_number,
                 "_score_history": self._score_history,
                 "_centroid": self._centroid}

        return state

    def __setstate__(self, state):

        for key in state:
            setattr(self, key, state[key])

        # Reconstruct larger structures and load MPI
        self._par_choices = np.arange(0, self._num_parameters, dtype=np.int32)
        self._old_centroid = self._centroid.copy()
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()
        self.objective = None
        self.obj_args = tuple()


class BoundedBasicES(BasicES):

    def __init__(self, xo, step_size, bounds, **kwargs):
        super().__init__(xo, step_size, **kwargs)

        self._boundary_type = kwargs.get("boundary_type", "clip")
        # Bounds is a 2D array with shape (num_params x 2) (low,high)
        self._bounds = np.array(bounds, dtype=np.float32)
        self._parameter_scale = (self._bounds[:, 1] - self._bounds[:, 0]) / 1.
        self._lower_bounds = self._bounds[:, 0]
        self._upper_bounds = self._bounds[:, 1]

        # bound initial centroid
        self._apply_bounds(self._centroid)
        self._old_centroid = self._centroid.copy()

    def _apply_bounds(self, search_values):

        if self._boundary_type == "clip":
            np.clip(search_values, self._lower_bounds, self._upper_bounds,
                    out=search_values)

        else:
            raise NotImplementedError("Error: " + self._boundary_type +
                                      " not implemented")

    def _update(self, objective):

        # Perturb centroid
        perturbed_dimensions = self._global_rng.choice(self._par_choices,
                                                       size=self._num_mutations,
                                                       replace=False)
        local_perturbation = self._worker_rngs[self._rank].randn(
            self._num_mutations)
        local_perturbation *= self._step_size
        self._centroid[perturbed_dimensions] += local_perturbation

        # Apply bounds
        self._apply_bounds(self._centroid)

        # Run objective
        local_cost = np.empty(1, dtype=np.float32)
        local_cost[0] = objective(self._centroid)

        # Consolidate return values
        all_costs = np.empty(self._size, dtype=np.float32)
        self._comm.Allgather([local_cost, self._MPI.FLOAT],
                             [all_costs, self._MPI.FLOAT])
        self._update_log(all_costs)

        self._update_centroid(all_costs, local_perturbation, perturbed_dimensions)

    def __getstate__(self):
        state = super().__getstate__()
        addition_states = {"_boundary_type": self._boundary_type,
                           "_bounds": self._bounds}
        state.update(addition_states)

        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        self._parameter_scale = (self._bounds[:, 1] - self._bounds[:, 0]) / 1.
        self._lower_bounds = self._bounds[:, 0]
        self._upper_bounds = self._bounds[:, 1]

        # bound initial centroid
        self._apply_bounds(self._centroid)
        self._old_centroid = self._centroid.copy()


class RandNumTableES(BasicES):
    """
    Creates a large RN table
    Draws a proper basic slice (start,stop,step) from RN table
    Draws a proper basic slice (start,stop,step) from parameter list
    Applies RNGs from sliced table to sliced parameters.
    Then each rank only needs to reconstruct the slice draws, and not
    the full rng list, which makes it less expensive
    The two step slice draw helps smooth out the impending correlations between
    RNs in the RN table space when finally applied to parameter space.
    Becomes more of an issue the larger the slice is with respect to the
    parameter space and the table space. Worst case is small table with large
    slice and many parameters.

    max_table_step is the maximum stride that an iterator will take over the
    table. max_param_step is the maximum stride that an iterator will take
    over the parameters, when drawing a random subset for permutation.
    A larger step will help wash out correlations in randomness. However,
    if the step is too large you risk overlapping old values as it wraps around
    the arrays.

    WARNING: If the # mutations approaches # parameters, make sure that the
    max_param_step == 1, else overlapping will cause actual # mutations to be
    less than the desired value.

    Currently, the global_rng draws perturbed dimensions for each iteration
    This could be changed to let workers draw their own dimensions to perturb
    """

    def __init__(self, xo, step_size, **kwargs):
        super().__init__(xo, step_size, **kwargs)
        self._rand_num_table_size = kwargs.get("rand_num_table_size", 20000000)
        self._rand_num_table = self._global_rng.randn(self._rand_num_table_size)
        self._max_table_step = kwargs.get("max_table_step", 5)
        self._max_param_step = kwargs.get("max_param_step", 1)

        # Fold step-size into table values
        self._rand_num_table *= self._step_size

    def __getstate__(self):
        state = super().__getstate__()
        addition_states = {"_rand_num_table_size": self._rand_num_table_size,
                           "_max_table_step": self._max_table_step,
                           "_max_param_step": self._max_param_step}
        state.update(addition_states)

        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        self._rand_num_table = self._global_rng.randn(self._rand_num_table_size)
        self._rand_num_table *= self._step_size

    def _draw_random_table_slices(self, rng):
        """
        Chooses a constrained slice subset of the RN table (start, stop, step)
        to give roughly num_mutations random numbers (less if overlap if
        step is too large)
        """

        return random_slices(rng, self._rand_num_table_size,
                             self._num_mutations, self._max_table_step)

    def _draw_random_parameter_slices(self, rng):
        """
        Chooses a constrained slice subset of the parameters (start, stop, step)
        to give roughly num_mutations perturbations (less if overlap if
        step is too large)
        """

        return random_slices(rng, self._num_parameters,
                             self._num_mutations, self._max_param_step)

    def _update(self, objective):

        # Perturb centroid
        unmatched_dimension_slices = self._draw_random_parameter_slices(
            self._global_rng)
        unmatched_perturbation_slices = self._draw_random_table_slices(
            self._worker_rngs[self._rank])

        # Match slices against each other
        dimension_slices, perturbation_slices = match_slices(
            unmatched_dimension_slices,
            unmatched_perturbation_slices)
        # Apply perturbations
        multi_slice_add(self._centroid, self._rand_num_table,
                        dimension_slices, perturbation_slices)

        # Run objective
        local_cost = np.empty(1, dtype=np.float32)
        local_cost[0] = objective(self._centroid)

        # Consolidate return values
        all_costs = np.empty(self._size, dtype=np.float32)
        self._comm.Allgather([local_cost, self._MPI.FLOAT],
                             [all_costs, self._MPI.FLOAT])
        self._update_log(all_costs)

        self._update_centroid(all_costs, unmatched_dimension_slices,
                              dimension_slices, perturbation_slices)

    def _reconstruct_perturbation(self, rank, master_dim_slices, parent_num):

        perturbation_slices = self._draw_random_table_slices(
            self._worker_rngs[rank])
        dimension_slices, perturbation_slices = match_slices(
            master_dim_slices,
            perturbation_slices)
        multi_slice_divide(self._old_centroid, self._weights[parent_num],
                           dimension_slices)
        multi_slice_add(self._old_centroid, self._rand_num_table,
                        dimension_slices, perturbation_slices)
        multi_slice_multiply(self._old_centroid, self._weights[parent_num],
                             dimension_slices)

    def _update_centroid(self, all_costs, master_dim_slices, local_dim_slices,
                         local_perturbation_slices):
        """
        Adds an additional multiply operation to avoid creating a new
        set of arrays for the slices. Not sure which would be faster
        """
        for parent_num, rank in enumerate(np.argsort(all_costs)):
            if parent_num < self._num_parents:
                if rank == self._rank:
                    multi_slice_divide(self._old_centroid, self._weights[parent_num],
                                       local_dim_slices)
                    multi_slice_add(self._old_centroid, self._rand_num_table,
                                    local_dim_slices, local_perturbation_slices)
                    multi_slice_multiply(self._old_centroid, self._weights[parent_num],
                                         local_dim_slices)

                else:
                    self._reconstruct_perturbation(rank, master_dim_slices, parent_num)

            else:
                if rank != self._rank:
                    self._draw_random_table_slices(self._worker_rngs[rank])

        multi_slice_assign(self._centroid, self._old_centroid,
                           master_dim_slices, master_dim_slices)


class BoundedRandNumTableES(RandNumTableES):
    """
    Currently support clipping and periodic bounds
    boundary_type: "clip"

    *note: periodic is troublesome to implement due to the variable
    windows at each iteration and the need to rescale the entire array
    rather than a subset. This is because the search needs to happen in the 
    folded real space (which gets folded onto (0,1)), but the objective needs 
    the rescaled space.
    """

    def __init__(self, xo, step_size, bounds, **kwargs):
        super().__init__(xo, step_size, **kwargs)

        self._boundary_type = kwargs.get("boundary_type", "clip")
        # Bounds is a 2D array with shape (num_params x 2) (low,high)
        self._bounds = np.array(bounds, dtype=np.float32)
        self._parameter_scale = (self._bounds[:, 1] - self._bounds[:, 0]) / 1.
        self._lower_bounds = self._bounds[:, 0]
        self._upper_bounds = self._bounds[:, 1]

        # bound initial centroid
        self._apply_bounds(self._centroid, [np.s_[:]])
        self._old_centroid = self._centroid.copy()

    def __getstate__(self):
        state = super().__getstate__()
        addition_states = {"_boundary_type": self._boundary_type,
                           "_bounds": self._bounds}
        state.update(addition_states)

        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        self._parameter_scale = (self._bounds[:, 1] - self._bounds[:, 0]) / 1.
        self._lower_bounds = self._bounds[:, 0]
        self._upper_bounds = self._bounds[:, 1]

        # bound initial centroid
        self._apply_bounds(self._centroid, [np.s_[:]])
        self._old_centroid = self._centroid.copy()

    def _rescale_search_parameters(self, search_values, slice_list):
        """
        Assuming values are in the range (0,1) this function rescales
        the parameters to their desired range
        """
        multi_slice_multiply(search_values, self._parameter_scale,
                             slice_list, slice_list)
        multi_slice_add(search_values, self._lower_bounds,
                        slice_list, slice_list)

    def _periodic_search_parameters(self, search_values, slice_list):
        """
        Places the parameters in a range between (0,1). When
        values exceed the range they are wrapped around back between that
        range. E.g. 1.5 -> 0.5, -1 -> 1, -2 -> 0, etc
        """
        multi_slice_mod(search_values, 2, slice_list)
        multi_slice_subtract(search_values, 1, slice_list)
        multi_slice_fabs(search_values, slice_list)
        multi_slice_multiply(search_values, -1, slice_list)
        multi_slice_add(search_values, 1, slice_list)

    def _apply_periodic_bounds(self, search_values, slice_list):
        """
        Rescales the parameters using periodic boundary conditions. 
        By default the search parameter space is bound between 0 and 1.
        """

        self._periodic_search_parameters(search_values, slice_list)
        self._rescale_search_parameters(search_values, slice_list)

    def _apply_bounds(self, search_values, slice_list):

        if self._boundary_type == "clip":
            multi_slice_clip(search_values, self._lower_bounds,
                             self._upper_bounds, slice_list, slice_list, slice_list)

        else:
            raise NotImplementedError("Error: " + self._boundary_type +
                                      " not implemented")

    def _update(self, objective):

        # Perturb centroid
        unmatched_dimension_slices = self._draw_random_parameter_slices(
            self._global_rng)
        unmatched_perturbation_slices = self._draw_random_table_slices(
            self._worker_rngs[self._rank])

        # Match slices against each other
        dimension_slices, perturbation_slices = match_slices(
            unmatched_dimension_slices,
            unmatched_perturbation_slices)

        # Apply perturbations
        multi_slice_add(self._centroid, self._rand_num_table,
                        dimension_slices, perturbation_slices)

        # Apply bounds
        self._apply_bounds(self._centroid, dimension_slices)

        # Run objective
        local_cost = np.empty(1, dtype=np.float32)
        local_cost[0] = objective(self._centroid)

        # Consolidate return values
        all_costs = np.empty(self._size, dtype=np.float32)
        self._comm.Allgather([local_cost, self._MPI.FLOAT],
                             [all_costs, self._MPI.FLOAT])
        self._update_log(all_costs)

        self._update_centroid(all_costs, unmatched_dimension_slices,
                              dimension_slices, perturbation_slices)

    def _update_centroid(self, all_costs, master_dim_slices, local_dim_slices,
                         local_perturbation_slices):
        """
        Adds an additional multiply opperation to avoid creating a new
        set of arrays for the slices. Not sure which would be faster

        Boundaries are clipped again at the end, to make sure the parameters in
        this window are fit for the objective in the next iteration (which will
        only clip params in the new window).
        """
        for parent_num, rank in enumerate(np.argsort(all_costs)):
            if parent_num < self._num_parents:
                if rank == self._rank:
                    multi_slice_divide(self._old_centroid, self._weights[parent_num],
                                       local_dim_slices)
                    multi_slice_add(self._old_centroid, self._rand_num_table,
                                    local_dim_slices, local_perturbation_slices)
                    multi_slice_multiply(self._old_centroid, self._weights[parent_num],
                                         local_dim_slices)

                else:
                    self._reconstruct_perturbation(rank, master_dim_slices, parent_num)

            else:
                if rank != self._rank:
                    self._draw_random_table_slices(self._worker_rngs[rank])

        # apply bounds in this window to prevent problems in next iteration
        self._apply_bounds(self._old_centroid, master_dim_slices)
        multi_slice_assign(self._centroid, self._old_centroid,
                           master_dim_slices, master_dim_slices)


if __name__ == '__main__':
    """test"""
    pass
