import numpy as np
import pickle
import math
from functools import partial

class BasicES():

    def __init__(self, xo, step_size, **kwargs):
        """
        xo: initial centroid
        step_size: float. the size of mutations
        num_mutations: number of dims to mutate in each iter (defaults to #dim)
        bounds: 2D numpy array. # parameter X 2 array, (low, high)
        bounardy_type: "repair" or "periodic"
        penalty_coef: float, size of penalty for repair boundary
        verbose: True/False. print info on run

        """
        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()

        # User input parameters
        self.objective = kwargs.get('objective', None)
        self.obj_args = kwargs.get('obj_args', ())
        self._step_size = step_size
        self._num_parameters = len(xo)
        self._num_mutations = kwargs.get('num_mutations', self._num_parameters)
        self._num_parents = kwargs.get('num_parents', int(self._size / 2))
        self._verbose = kwargs.get('verbose', False)
        self._global_seed = kwargs.get('seed', 1)

        # Internal parameters
        self._weights = np.log(self._num_parents + 0.5) - \
                        np.log(np.arange(1, self._num_parents + 1))
        self._weights /= np.sum(self._weights)
        self._weights.astype(np.float32, copy=False)

        # Internal data
        self._global_rng = np.random.RandomState(self._global_seed)
        self._seed_set = self._global_rng.choice(0, 1000000, size=self._size, 
                                                replace=False)
        self._worker_rngs = [ np.random.RandomState(seed) 
                                for seed in self._seed_set ]
        self._par_choices = np.arange(0, self._num_parameters, dtype=np.int32)
        self._generation_number = 0
        self._score_history = []

        # State
        self._centroid = np.array(xo, dtype=np.float32)
        self._old_centroid = self._centroid.copy()

    def __call__(self, num_iterations, objective_funct=None, args=()):

        if (self.objective != None) and (objective_funct != None):
            objective_funct = self.objective
            args = self.obj_args
        else:
            raise AttributeError("Error: No objective defined")

        partial_objective = partial(objective_funct, *args)
        for i in range(num_iterations):
            if self.verbose and (self._rank == 0):
                print("Generation:", self._generation_number)
            self._update(partial_objective)
            self._generation_number += 1

    def _update(self, objective):

        # Perturb centroid
        perturbed_dimensions = self._global_rng.choice(self._par_choices,
                                    size=self._num_mutations, replace=False)
        local_perturbation = self._worker_rngs[self._rank].randn(
                            self._num_mutations)
        local_perturbation *= self._step_size
        self._centroid[perturbed_dimensions] += local_perturbation

        # Run objective
        local_cost = np.empty(1, dtype=np.float32)
        local_cost[0] = objective(self._centroid)

        # Consolidate return values
        all_costs = np.empty(self._size, dtype=np.float32)
        self.comm.Allgather([local_cost, self.MPI.FLOAT], 
                            [all_costs, self.MPI.FLOAT])
        self._update_log(all_costs)

        self._update_centroid(all_costs, local_perturbation, perturbed_dimensions)

    def _update_centroid(self, all_costs, local_perturbation, perturbed_dimensions):

        for parent_num, rank in enumerate(np.argsort(costs)):
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

        self._centroid[perturbed_dimensions] = self._old_centroid[perturbed_dimensions] #care, copies old_cent

    def _update_log(self, costs):

        self._score_history.append(costs)

    def get_centroid(self):

        return self._centroid.copy()

    def plot_cost_over_time(self, prefix='test', logy=True, savefile=False):
        """
        Plots the evolutionary history of the population's cost.
        Includes min cost individual for each generation the mean
        """

        import matplotlib.pyplot as plt

        costs_by_generation = np.array(self._score_history)
        min_cost_by_generation = \
            np.min(costs_by_generation, axis=1)
        mean_cost_by_generation = \
            np.mean(costs_by_generation, axis=1)

        plt.plot(range(len(mean_cost_by_generation)), \
            mean_cost_by_generation,
            marker='None', ls='-', color='blue', label='mean cost')

        plt.plot(range(len(min_cost_by_generation)), \
            min_cost_by_generation, ls='--', marker='None', \
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
        pickled_obj_file = open(filename,'rb')
        obj = pickle.load(pickled_obj_file)
        pickled_obj_file.close()

        return obj

    def save(self, filename):

        if self._rank == 0 or self._rank == -1:
            pickled_obj_file = open(filename,'wb')
            try:
                pickle.dump(self, pickled_obj_file, 2)
            except TypeError:
                print("Can't pickle objective, setting to None")
                self.objective = None
                pickle.dump(self, pickled_obj_file, 2)

            pickled_obj_file.close()

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

    Currently, the global_rng draws perturbed dimensions for each iteration
    This could be changed to let workers draw their own dimensions to perturb
    """

    def __init__(self, xo, step_size, **kwargs):
        super().__init__(xo, step_size, **kwargs)
        self._rand_num_table_size = kwargs.get("rand_num_table_size", 20000000)
        self._rand_num_table = self._global_rng.randn(self._rand_num_table_size)

        # Fold step-size into table values
        self._rand_num_table *= self._step_size

        # 1 added because max is excluded in randint
        self._max_table_step = kwargs("max_table_step", 5) + 1
        self._max_param_step = kwargs("max_param_step", 1) + 1

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
                            self._num_mutations, self._max_table_step)

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
        self.comm.Allgather([local_cost, self.MPI.FLOAT], 
                            [all_costs, self.MPI.FLOAT])
        self._update_log(all_costs)

        self._update_centroid(all_costs, unmatched_dimension_slices, 
                                unmatched_perturbation_slices)

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

    def _update_centroid(self, all_costs, master_dim_slices, local_dim_slices, local_perturbation_slices):
        """
        Adds an additional multiply opperation to avoid creating a new
        set of arrays for the slices. Not sure which would be faster
        """
        for parent_num, rank in enumerate(np.argsort(costs)):
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
                            perturbed_dimensions, perturbed_dimensions)

class BoundedES(RandNumTableES):
    """
    Currently support clipping and periodic bounds
    boundary_type: "clip" or "periodic"
    """

    def __init__(self, xo, step_size, bounds, **kwargs):
        super().__init__(xo, step_size, **kwargs)

        self._boundary_type = kwargs("boundary_type", "clip")
        # Bounds is a 2D array with shape (num_params x 2) (low,high)
        self._bounds = np.array(bounds, dtype=np.float32)
        self._parameter_scale = (self._bounds[:,1] - self._bounds[:,0]) / 1.
        self._lower_bounds = self._bounds[:,0]
        self._upper_bounds = self._bounds[:,1]

    def _rescale_search_parameters(self, search_values, slice_list):

        multi_slice_multiply(search_values, self._parameter_scale,
            slice_list)
        multi_slice_add(search_values, self._lower_bounds, 
            slice_list, slice_list)

    def _periodic_search_parameters(self, search_values, slice_list):

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

        elif self._boundary_type == "periodic":
            self._apply_periodic_bounds(search_values, slice_list)

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
        self.comm.Allgather([local_cost, self.MPI.FLOAT], 
                            [all_costs, self.MPI.FLOAT])
        self._update_log(all_costs)

        self._update_centroid(all_costs, unmatched_dimension_slices, 
                                unmatched_perturbation_slices)

def multi_slice_add(x1_inplace, x2, x1_slices=[], x2_slices=[]):
    """
    Does an inplace addition on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] += x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] += x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace += x2

def multi_slice_subtract(x1_inplace, x2, x1_slices=[], x2_slices=[]):
    """
    Does an inplace addition on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] -= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] -= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace -= x2

def multi_slice_multiply(x1_inplace, x2, x1_slices=[], x2_slices=[]):
    """
    Does an inplace multiplication on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] *= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] *= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace *= x2

def multi_slice_divide(x1_inplace, x2, x1_slices=[], x2_slices=[]):
    """
    Does an inplace multiplication on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] /= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] /= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace /= x2

def multi_slice_assign(x1_inplace, x2, x1_slices=[], x2_slices=[]):
    """
    Does an inplace assignment on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] = x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] = x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace = x2

def multi_slice_mod(x1_inplace, x2, x1_slices=[], x2_slices=[]):
    """
    Does an inplace modulo on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] %= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] %= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace %= x2

def multi_slice_fabs(x1_inplace, x1_slices=[]):
    """
    Does an inplace fabs on x1 given a list of slice objects
    """

    if (len(x1_slices) != 0):
        for x1_slice in x1_slices:
            np.fabs(x1_inplace[x1_slice], out=x1_inplace[x1_slice])

    else:
        np.fabs(x1_inplace, out=x1_inplace)

def multi_slice_clip(x1_inplace, lower, upper, xslices, lslices=[], uslices=[]):
    """
    Does an inplace clip on x1
    """

    if (len(lslices) == 0) or (len(uslices) == 0):
        for xslice in xslices:
            np.clip(x1_inplace[xslice], lower, upper, out=x1_inplace[xslice])

    elif (len(lslices) != 0) and (len(uslices) != 0) \
            and (len(lslices) == len(uslices)):
        for i in range(len(xslices)):
            np.clip(x1_inplace[xslices[i]], lower[lslices[i]], upper[uslices[i]],
                out=x1_inplace[xslices[i]])

    elif (len(lslices) == 0) and (len(uslices) == 0) and (len(xslices) == 0):
        np.clip(x1_inplace, lower, upper, out=x1_inplace)

    else:
        raise NotImplementedError("Invalid arguments in multi_slice_clip")

def random_slices(rng, iterator_size, slice_size, max_step=1):
    """
    Returns a list of slice objects given the size of the iterator it
    will be used for and the number of elements desired for the slice
    This will return additional slice each time it wraps around the
    iterator

    iterator_size - the number of elements in the iterator
    slice_size - the number of elements the slices will cover
    max_step - the maximum number of steps a slice will take.
                This affects the number of slice objects created, as
                larger max_step will create more wraps around the iterator
                and so return more slice objects

    The number of elements is not garanteed when slices overlap themselves
    """

    step_size = rng.randint(1, max_step + 1) # randint is exclusive
    start_step = rng.randint(0, iterator_size)

    return build_slices(start_step, iterator_size, slice_size, step_size)

def build_slices(start_step, iterator_size, slice_size, step_size):
    """
    Given a starting index, the size of the total members of the window,
    a step size, and the size of the iterator the slice will act upon,
    this function returns a list of slice objects that will cover that full
    window. Upon reaching the endpoints of the iterator, it will wrap around.
    """

    end_step = start_step + step_size * slice_size
    slices = []
    slice_start = start_step
    for i in range(1 + (end_step - step_size) // iterator_size):
        remaining = end_step - i * iterator_size
        if (remaining > iterator_size):
            remaining = iterator_size
        
        slice_end = (slice_start + 1) + ((remaining - \
                    (slice_start + 1)) // step_size) * step_size
        slices.append(np.s_[slice_start:slice_end:step_size])
        slice_start = (slice_end - 1 + step_size) % iterator_size

    return slices

def match_slices(slice_list1, slice_list2):
    """
    Will attempt to create additional slices to match the # elements of 
    each slice from list1 to the corresponding slice of list 2.
    Will fail if the total # elements is different for each list
    """

    slice_list1 = list(slice_list1) 
    slice_list2 = list(slice_list2)
    if slice_size(slice_list1) == slice_size(slice_list2):
        slice_list1.reverse()
        slice_list2.reverse()
        new_list1_slices = []
        new_list2_slices = []

        while len(slice_list1) != 0 and len(slice_list2) != 0:
            slice_1 = slice_list1.pop()
            slice_2 = slice_list2.pop()
            size_1 = slice_size(slice_1)
            size_2 = slice_size(slice_2)

            if size_1 < size_2:
                new_slice_2, slice_2 = splice_slice(slice_2, size_1)
                slice_list2.append(slice_2)
                new_list2_slices.append(new_slice_2)
                new_list1_slices.append(slice_1)

            elif size_2 < size_1:
                new_slice_1, slice_1 = splice_slice(slice_1, size_2)
                slice_list1.append(slice_1)
                new_list1_slices.append(new_slice_1)
                new_list2_slices.append(slice_2)

            elif size_1 == size_2:
                new_list1_slices.append(slice_1)
                new_list2_slices.append(slice_2)

    else:
        raise AssertionError("Error: slices not compatible")

    return new_list1_slices, new_list2_slices

def splice_slice(slice_obj, num_elements):
    """
    Returns two slices spliced from a single slice.
    The size of the first slice will be # elements
    The size of the second slice will be the remainder
    """
    
    splice_point = slice_obj.step * (num_elements - 1) + slice_obj.start + 1
    new_start = splice_point - 1 + slice_obj.step
    return np.s_[slice_obj.start : splice_point : slice_obj.step], \
            np.s_[new_start : slice_obj.stop : slice_obj.step]

def slice_size(slice_objs):
    """
    Returns the total number of elements in the combined slices
    Also works if given a single slice
    """

    num_elements = 0

    try:
        for sl in slice_objs:
            num_elements += (sl.stop - (sl.start + 1)) // sl.step + 1
    except TypeError:
        num_elements += (slice_objs.stop - (slice_objs.start + 1)) // slice_objs.step + 1

    return num_elements

if __name__ == '__main__':
    """test"""
    sl1 = build_slices(7, 13, 10, 2)
    sl2 = build_slices(0, 13, 10, 2)
    print(sl1)
    print(sl2)
    new_sl1, new_sl2 = match_slices(sl1, sl2)
    print()
    print(new_sl1)
    print(new_sl2)

    for i in range(len(new_sl1)):
        print(slice_size(new_sl1[i]), slice_size(new_sl2[i]))
    print(slice_size(new_sl1), slice_size(new_sl2))

    a1 = np.arange(0, 13, 1)
    a2 = 2*np.ones(13)
    a3 = 5*np.ones(13)
    print(a1)
    print(a2)
    print()
    multi_slice_clip(a1, a2, a3, new_sl2, new_sl2, new_sl2)
    print(a1)
    print(a2)