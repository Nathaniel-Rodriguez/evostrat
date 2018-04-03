import numpy as np
import pickle
import random
from functools import partial
from .sliceops import *
from abc import ABC, abstractmethod


class BaseGA(ABC):
    """
    Contains basic members and functions common to derived GA classes
    """

    def __init__(self, **kwargs):
        """
        :param objective:
        :param obj_kwargs:
        :param verbose:
        :param parent_fraction:
        :param num_parents:
        :param elite_fraction:
        :param num_elite:
        :param seed:
        """

        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        self.objective = kwargs.get('objective', None)
        self.obj_kwargs = kwargs.get('obj_kwargs', {})
        self._verbose = kwargs.get('verbose', False)
        self._max_seed = 2 ** 32 - 1

        if 'parent_fraction' in kwargs:
            self._parent_fraction = kwargs.get('parent_fraction', 0.3)
            self._num_parents = int(self._size * self._parent_fraction)
        elif 'num_parents' in kwargs:
            self._num_parents = kwargs.get('num_parents', int(self._size / 2))
            self._parent_fraction = self._num_parents / self._size

        if 'elite_fraction' in kwargs:
            self._elite_fraction = kwargs.get('elite_fraction', 0.1)
            self._num_elite = int(self._size * self._elite_fraction)
        elif 'num_elite' in kwargs:
            self._num_elite = kwargs.get('num_elite', 1)
            self._elite_fraction = self._num_elite / self._size

        if self._num_elite < self._num_parents:
            raise AssertionError("Number of elite has to be less than the"
                                 " number of parents")

        self._global_seed = kwargs.get('seed', 1)
        self._py_rng = random.Random(self._global_seed)
        self._global_rng = np.random.RandomState(self._global_seed)
        self._initial_seed_list = self._py_rng.sample(range(self._max_seed),
                                                      self._size)
        self._seed_seed_list = self._py_rng.sample(range(self._max_seed),
                                                   self._size)
        self._mutation_rng = np.random.RandomState(self._initial_seed_list[self._rank])
        self._seed_rng = np.random.RandomState(self._seed_seed_list[self._rank])

        self._generation_number = 0
        self._cost_history = []
        self._member_genealogy = [self._initial_seed_list[self._rank]]
        self._member = self._make_member(self._mutation_rng,
                                         self._member_genealogy)
        self._population_genealogy = [[] for i in range(self._size)]

    def __call__(self, num_iterations, objective=None, kwargs=None):
        """
        :param num_iterations: how many generations it will run for
        :param objective: a full or partial version of function
        :param kwargs: key word arguments for additional objective parameters
        :return: None
        """
        if objective is not None:
            if kwargs is None:
                kwargs = {}
        elif (self.objective is None) and (objective is None):
            raise AttributeError("Error: No objective defined")
        else:
            objective = self.objective
            kwargs = self.obj_kwargs

        partial_objective = partial(objective, **kwargs)
        for i in range(num_iterations):
            if self._verbose and (self._rank == 0):
                print("Generation:", self._generation_number)
            self._update(partial_objective)
            self._generation_number += 1

        self._share_genealogy()

    @abstractmethod
    def member_generator(self, rng):
        """
        Creates a new member and returns it.
        :param rng: a numpy random number generator
        :return: a new member
        """
        pass

    @abstractmethod
    def mutator(self, member, rng):
        """
        Mutates the member in place or returns a new one. If mutated in-place,
        return the reference.
        :param member: Reference to member
        :param rng: a numpy random number generator
        :return: Reference to member or new member
        """
        pass

    @abstractmethod
    def _update(self, objective):
        """
        Method that updates the genetic algorithm using the objective.
        The objective is a partial function created from whatever kwargs were
        given either upon instantiation of object or __call__
        :param objective: partial objective
        :return: None
        """
        pass

    @property
    def best(self):
        """
        :return: generates and returns the best member of the population.
            Only Rank==0 should be accessing this property.
        """

        try:
            return self._make_member(self._mutation_rng,
                                     self._population_genealogy[
                                         np.argsort(self._cost_history[-1])[0]])
        except IndexError:
            raise IndexError("No score or population genealogy from which"
                             "to generate best. Run optimization first.")
        except TypeError:
            raise TypeError("Need to set mutation and member generating functions")

    @property
    def population(self):
        """
        :return: a list of all members of the current population.
            Only Rank==0 should be accessing this property.
        """

        try:
            return [self._make_member(self._mutation_rng, member)
                    for member in self._population_genealogy]
        except IndexError:
            raise IndexError("No score or population genealogy"
                             "Run optimization first.")
        except TypeError:
            raise TypeError("Need to set mutation and member generating functions")

    @property
    def costs(self):
        """
        :return: scores of all members of the current population. In same order
            as population list. Only Rank==0 should be accessing this property.
        """

        return self._cost_history[-1].copy()

    def _make_member(self, rng, seed_list):
        """
        Creates a member of the population for this rank
        """

        rng.seed(seed_list[0])
        new_member = self.member_generator(rng)
        for seed in seed_list[1:]:
            rng.seed(seed)
            new_member = self.mutator(new_member, rng)
        return new_member

    def _share_genealogy(self):
        """
        It is difficult to track the population due to the nature of this
        distributed algorithm. It is more efficient to wait until the end of
        the optimization period to share all of the genealogies so that
        all nodes know what the populations are. This allows the generation
        of the final population and the best member in it.

        ONLY THE RANK==0 member of the population gets the genealogies as
        that is the only one that is saved after pickling. Doing this reduces
        the send operations from N^2 to N. Also, there isn't much that should
        be done over MPI once the optimization is complete.

        This function will allow the use of the best and population properties.

        :return: None, it fills the _population_genealogy list
        """

        self._population_genealogy = self._comm.gather(self._member_genealogy,
                                                       root=0)

    def _update_log(self, costs):

        self._cost_history.append(costs)

    def _dispatch_messages(self, messenger_list):
        """
        Iterates through a messenger list and sends/receives the genealogies
        for each pair of ranks
        """
        for messenger in messenger_list:
            # something went terribly wrong if node is sending to itself
            assert (messenger[0] != messenger[1])

            # send/recv genealogy
            if self._rank == messenger[0]:
                self._comm.send(self._member_genealogy, dest=messenger[1])

            if self._rank == messenger[1]:
                self._member_genealogy = self._comm.recv(source=messenger[0])

    def _construct_received_members(self, messenger_list):
        """
        Iterates through the messenger list and for ranks that recieved
        a genealogy it builds a new member from it and replaces the current
        member
        """
        for messenger in messenger_list:
            if self._rank == messenger[1]:
                self._member = self._make_member(self._mutation_rng,
                                                 self._member_genealogy)

    def plot_cost_over_time(self, prefix='test', logy=True, savefile=False):
        """
        Plots the evolutionary history of the population's cost.
        Includes min cost individual for each generation the mean
        """
        if self._rank == 0:
            import matplotlib.pyplot as plt

            costs_by_generation = np.array(self._cost_history)
            min_cost_by_generation = np.min(costs_by_generation, axis=1)
            mean_cost_by_generation = np.mean(costs_by_generation, axis=1)

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

        state = {"_num_parents": self._num_parents,
                 "_parent_fraction": self._parent_fraction,
                 "_verbose": self._verbose,
                 "_elite_fraction": self._elite_fraction,
                 "_num_elite": self._num_elite,
                 "_max_seed": self._max_seed,
                 "_global_seed": self._global_seed,
                 "_global_rng": self._global_rng,
                 "_py_rng": self._py_rng,
                 "_initial_seed_list": self._initial_seed_list,
                 "_seed_seed_list": self._seed_seed_list,
                 "_mutation_rng": self._mutation_rng,
                 "_seed_rng": self._seed_rng,
                 "_generation_number": self._generation_number,
                 "_cost_history": self._cost_history,
                 "_member_genealogy": self._member_genealogy,
                 "_member": self._member,
                 "_population_genealogy": self._population_genealogy}

        return state

    def __setstate__(self, state):

        for key in state:
            setattr(self, key, state[key])

        # Reconstruct larger structures and load MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()
        self.objective = None
        self.obj_args = None


class BasicGA(BaseGA):
    """
    A modular GA that requires a mutation size (sigma), member size, and
    bounds for member initialization.

    If the GA is ever saved and reloaded the objective function needs to be
    assigned again.

    This GA maintains a list of seeds for each member of the population which
    denotes its construction and mutation history. When members of the population
    are replaced this list is transferred between nodes and new mutations under
    the unique seed of that node are applied, branching the lineage of that
    member.

    MPI broadcasts remain isolated to the fitness function, but site2site
    messages are sent between ranks that need a new member, which will be
    drawn uniformly at random from available parents.

    An elite fraction can be set which determines what fraction of the population
    is withheld from mutation. The top best % of the population will go onto
    the next unchanged.

    The parent_fraction can be set which determines the selection strength.
    The top parent_fraction of members will be retained, while the remaining
    will be replaced with a uniform random draw of those that were successfull.

    Following the run, the properties best, costs, and population can be
    accessed by rank==0 (or whatever object is loaded after saving).
    If the object is reloaded then member_generating_function and
    mutation_function properties need to be set (since functions can't be
    pickled) before best and population can be accessed.
    """

    def __init__(self, sigma, member_size, member_draw_bounds, **kwargs):
        """
        :param sigma: the standard deviation of the normal distribution of
            perturbations applied as mutations
        :param member_size: the number of parameters for each member
        :param member_draw_bounds: the low/high boundaries for the initial
            draw of parameters for a member. e.g. (-1.0, 1.0)
        :param elite_fraction: the fraction of members to withhold from mutation
        :param parent_fraction: the fraction of the population to maintain. remaining
            members are culled and replaced with more successful parents.
            This is the selection process.
        """

        super().__init__(**kwargs)
        self._sigma = sigma
        self._member_size = member_size
        self._member_draw_bounds = member_draw_bounds

    def member_generator(self, rng):
        """
        Can be used with BasicGAs

        Function that takes a numpy rng to generate a new member.
        :param rng: a numpy random number generator
        :param size: the size of the returned array
        :param bounds: (low, high)

        Draws from a uniform distribution between low and high.

        :return 1d numpy float32 array that represents member
        """

        return rng.uniform(self._member_draw_bounds[0],
                           self._member_draw_bounds[1],
                           self._member_size).astype(np.float32)

    def mutator(self, member, rng):
        """
        Can be used with BasicGAs

        :param member: a member to mutate
        :param rng: a numpy randomState
        :return: reference to member that is changed in-place
        """

        perturbation = rng.randn(member.size)
        perturbation *= self._sigma
        member += perturbation

        return member

    def _update(self, objective):
        """
        Updates the population
        :param objective: a partial function, takes only parameters as input
        :return: None
        """
        # determine fitness and broadcast
        local_cost = np.empty(1, dtype=np.float32)
        local_cost[0] = objective(self._member)
        all_costs = np.empty(self._size, dtype=np.float32)
        self._comm.Allgather([local_cost, self._MPI.FLOAT],
                             [all_costs, self._MPI.FLOAT])
        self._update_log(all_costs)
        # Apply mutations, elite selection, and broadcast genealogies
        self._update_population(all_costs)

    def _chose_random_rank_by_performance(self, l2g_ranks):
        """
        picks randomly one of the ranks which has a member with cost in the
        bottom _num_parents
        """
        best_ranks = l2g_ranks[:self._num_parents]
        return self._global_rng.choice(best_ranks)

    def _construct_message_list(self, l2g_ranks):
        """
        Builds a list of tuples with (send_rank, recv_rank) pairs
        Every rank must participate in building this so the global_rng
        stays in sync with everyone and so that everyone has the same
        messenger list
        """

        messenger_list = []
        # For the rejects, pick a random parent to copy
        for parent_num, rank in enumerate(l2g_ranks):
            if parent_num >= self._num_parents:
                # randomly pick who to copy from higher rank members
                chosen_rank = self._chose_random_rank_by_performance(l2g_ranks)
                messenger_list.append((chosen_rank, rank))

        return messenger_list

    def _mutate_population(self, l2g_ranks):
        """
        Preserve the top _num_elite members, mutate the rest
        """
        for parent_num, rank in enumerate(l2g_ranks):
            if parent_num >= self._num_elite:
                if rank == self._rank:
                    # draw mutation seed
                    seed = self._mutation_rng.randint(0, self._max_seed)
                    self._member_genealogy.append(seed)

                    # apply mutation
                    self._mutation_rng.seed(seed)
                    self._member = self.mutator(self._member,
                                                self._mutation_rng)

    def _update_population(self, all_costs):
        """
        determines the cost ranking of all the costs for each node(rank) and
        then determines which ranks need new members, which are randomly
        selected and sent from the set of surviving members.
        Then all members except for the elite are mutated.
        """

        # argsort returns the indexes of all_costs that would sort the array
        # This means the first value is the highest performing rank while the
        # last value is the lowest performing rank
        l2g_ranks = np.argsort(all_costs)
        messenger_list = self._construct_message_list(l2g_ranks)

        self._dispatch_messages(messenger_list)
        self._construct_received_members(messenger_list)
        self._mutate_population(l2g_ranks)

    def __getstate__(self):
        state = super().__getstate__()
        state["_sigma"] = self._sigma
        state["_member_size"] = self._member_size
        state["_member_draw_bounds"] = self._member_draw_bounds

        return state


class BoundedBasicGA(BasicGA):
    """
    A BasicGA with bounds.
    """
    def __init__(self, **kwargs):
        """
        member_generating_function: a function that takes a numpy RNG and table
            as an argument and will return a np.float32 array that can be passed
            to the fitness function.
        mutation_function: a function that takes a numpy RNG and table as an
            argument and will return a np.float32 array that can be added to a
            member to produce a mutation.
        """
        pass


class RandNumTableGA(BaseGA):
    """
    A basic GA that uses a cached random number table. This speeds up
    the mutation process considerably at the cost of memory. By using a table
    the algorithm can be made several times faster as random number generation
    is expensive.

    The table is a numpy array of normally distributed values: N(0,1).
    """
    def __init__(self, sigma, member_size, member_draw_bounds,
                 rand_num_table_size, max_table_step, max_param_step, **kwargs):
        """
        :param sigma: the standard deviation of mutation perturbations
        :param member_size: number of parameters per member
        :param member_draw_bounds: low/high of initial draw for member
        :param rand_num_table_size: the number of elements in the random table
        :param max_table_step: the maximum random stride for table slices
        :param kwargs: parameters from BasicGA
        """
        super().__init__(**kwargs)

        self._rand_num_table_size = kwargs.get("rand_num_table_size", 20000000)
        self._rand_num_table = self._global_rng.randn(self._rand_num_table_size)
        self._max_table_step = kwargs.get("max_table_step", 5)


class BoundedRandNumTableGA(RandNumTableGA):
    """
    A RandNumTable with bounds.
    """

    def __init__(self):
        """
        member_generating_function: a function that takes a numpy RNG and table
            as an argument and will return a np.float32 array that can be passed
            to the fitness function.
        mutation_function: a function that takes a numpy RNG and table as an
            argument and will return a np.float32 array that can be added to a
            member to produce a mutation.
        """

        pass