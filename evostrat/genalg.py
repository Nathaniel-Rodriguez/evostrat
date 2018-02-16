import numpy as np
import pickle
import random
from functools import partial
from .sliceops import *


def real_member_generator(rng, size, bounds):
    """
    Can be used with BasicGAs

    Function that takes a numpy rng to generate a new member.
    size: the size of the returned array
    bounds: (low, high)

    Draws from a uniform distribution between low and high.

    :return 1d numpy float32 array that represents member
    """

    return rng.uniform(bounds[0], bounds[1], size).astype(np.float32)


def real_mutator(member, rng, scale):
    """
    Can be used with BasicGAs

    :param member: a member to mutate
    :param rng: a numpy randomState
    :param scale: scale of normal mutation
    :return: reference to member that is changed in-place
    """

    perturbation = rng.randn(member.size)
    perturbation *= scale
    member += perturbation

    return member


class BasicGA:
    """
    A modular GA that requires a mutation and member generating function
    Genotypes can be variable in length.
    If the GA is ever saved and reloaded the member generating function
    and mutation function and objective function need to be assigned again.
    The former two have corresponding methods. The object can be set directly
    when calling the GA (just as in the ES)

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

    Mutation rate and other mutation related parameters must be specified
    in the mutation_function and are not part of the GA itself. These
    can be provided as mutation_function_args, or a partial function can be
    passed to the GA
    """

    def __init__(self, member_generating_function, mutation_function,
                 member_generating_function_kwargs=None,
                 mutation_function_kwargs=None,
                 **kwargs):
        """
        member_generating_function: a function that takes a numpy RNG as an
            argument and will return a np.float32 array that can be passed to
            the fitness function. E.g:
                new_member = f(rng, *args)

        mutation_function: a function that takes a population member and numpy
            RNG as an argument and will return a np.float32 array. If desired
            the new member can be a reference to a modified version of the
            input member, resulting in an inplace change. E.g:
                new_member = f(member, rng, *args)

        member_generating_function_args: if a partial function isn't given
            then one is made with the arguments passed here

        mutation_function_args: if a partial function isn't given then one is
            made with the arguments passed here

        elite_fraction: the fraction of members to withhold from mutation
        parent_fraction: the fraction of the population to maintain. remaining
            members are culled and replaced with more successful parents.
            This is the selection process.
        """

        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        # Assign properties
        if member_generating_function_kwargs is None:
            member_generating_function_kwargs = {}

        if mutation_function_kwargs is None:
            mutation_function_kwargs = {}

        self._member_generating_function = partial(member_generating_function,
                                                   **member_generating_function_kwargs)
        self._mutation_function = partial(mutation_function, **mutation_function_kwargs)

        self.objective = kwargs.get('objective', None)
        self.obj_kwargs = kwargs.get('obj_kwargs', ())
        self._parent_fraction = kwargs.get('parent_fraction', 0.3)
        self._num_parents = int(self._size * self._parent_fraction)
        if self._rank == 0: #################
            print("num_parents:",self._rank, self._num_parents) ########################
        self._verbose = kwargs.get('verbose', False)
        self._elite_fraction = kwargs.get('elite_fraction', 0.1)
        self._num_elite = int(self._size * self._elite_fraction)
        if self._rank == 0: ##########
            print("num_elite:",self._rank, self._num_elite)  ########################
        assert(self._num_elite < self._num_parents)

        self._max_seed = 2 ** 32 - 1
        self._global_seed = kwargs.get('seed', 1)
        self._py_rng = random.Random(self._global_seed)
        self._global_rng = np.random.RandomState(self._global_seed)
        self._initial_seed_list = self._py_rng.sample(range(self._max_seed),
                                                      self._size)
        self._seed_seed_list = self._py_rng.sample(range(self._max_seed),
                                                   self._size)
        self._mutation_rng = np.random.RandomState(
            self._initial_seed_list[self._rank])
        self._seed_rng = np.random.RandomState(self._seed_seed_list[self._rank])

        self._generation_number = 0
        self._score_history = []
        self._member_genealogy = [self._initial_seed_list[self._rank]]
        print("first genealogy:", self._rank, self._member_genealogy)  ########################
        self._member = self._make_member(self._mutation_rng,
                                         self._member_genealogy)

    def __call__(self, num_iterations, objective=None, kwargs=None):
        """
        :param num_iterations: how many generations it will run for
        :param objective: a full or partial version of function
        :param kwargs: key word arguments for additional objective parameters
        :return: None
        """
        if kwargs is None:
            kwargs = {}

        if (self.objective is not None) and (objective is None):
            objective = self.objective
            kwargs = self.obj_kwargs
        elif (self.objective is None) and (objective is None):
            raise AttributeError("Error: No objective defined")

        partial_objective = partial(objective, **kwargs)
        for i in range(num_iterations):
            if self._verbose and (self._rank == 0):
                print("Generation:", self._generation_number)
            self._update(partial_objective)
            self._generation_number += 1

    def _make_member(self, rng, seed_list):
        """
        Creates a member of the population for this rank
        """

        rng.seed(seed_list[0])
        new_member = self._member_generating_function(rng)
        for seed in seed_list[1:]:
            rng.seed(seed)
            new_member = self._mutation_function(new_member, rng)
        print("\tmake_member:", self._generation_number, self._rank, seed_list, new_member)  ########################
        return new_member

    def _update_log(self, costs):

        self._score_history.append(costs)

    def _update(self, objective):

        # determine fitness and broadcast
        local_cost = np.empty(1, dtype=np.float32)
        local_cost[0] = objective(self._member)
        all_costs = np.empty(self._size, dtype=np.float32)
        self._comm.Allgather([local_cost, self._MPI.FLOAT],
                             [all_costs, self._MPI.FLOAT])
        self._update_log(all_costs)
        print("\tcost:", self._generation_number, self._rank, local_cost, self._member)  ########################
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
                    self._member = self._mutation_function(self._member,
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
        if self._rank == 0: ###############################
            print("\t\torder:", self._generation_number, all_costs, l2g_ranks)  ########################
            print("\t\tmessages:", messenger_list) ###################

        self._dispatch_messages(messenger_list)
        self._construct_received_members(messenger_list)
        self._mutate_population(l2g_ranks)

    def plot_cost_over_time(self, prefix='test', logy=True, savefile=False):
        """
        Plots the evolutionary history of the population's cost.
        Includes min cost individual for each generation the mean
        """
        if self._rank == 0:
            import matplotlib.pyplot as plt

            costs_by_generation = np.array(self._score_history)
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

    def assign_member_generating_function(self, member_generating_function, kwargs=None):
        """
        If a GA is reloaded it needs to be reassigned a member generating
        function. Use this method to do so.
        """
        if kwargs is None:
            kwargs = {}

        self._member_generating_function = partial(member_generating_function,
                                                   **kwargs)

    def assign_mutation_function(self, mutation_function, kwargs=None):
        """
        If a GA is reloaded it needs to be reassigned a mutation function.
        Use this method to do so.
        """
        if kwargs is None:
            kwargs = {}

        self._mutation_function = partial(mutation_function, **kwargs)

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
                 "_score_history": self._score_history,
                 "_member_genealogy": self._member_genealogy,
                 "_member": self._member}

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
        self._member_generating_function = None
        self._mutation_function = None
        self.objective = None
        self.obj_args = None


class BoundedBasicGA(BasicGA):
    """
    A BasicGA with bounds.
    """
    def __init__(self, member_generating_function, mutation_function,
                 bounds, **kwargs):
        """
        member_generating_function: a function that takes a numpy RNG and table
            as an argument and will return a np.float32 array that can be passed
            to the fitness function.
        mutation_function: a function that takes a numpy RNG and table as an
            argument and will return a np.float32 array that can be added to a
            member to produce a mutation.
        """
        pass


class RandNumTableGA(BasicGA):
    """
    A basic GA that uses a chached random number table. This speeds up
    the mutation process considerably at the cost of memory. By using a table
    the algorithm can be made several times faster as random number generation
    is expensive.

    The member_generating_function and mutation_function must take an additional
    argument, called 'table' which will be a reference to the float32 random
    number table. I encourage using slices from sliceops to draw random
    segments from this table as values.

    The table is a numpy array of normally distributed values: N(0,1).
    """
    def __init__(self, member_generating_function, mutation_function, **kwargs):
        """
        member_generating_function: a function that takes a numpy RNG and table
            as an argument and will return a np.float32 array that can be passed
            to the fitness function.
        mutation_function: a function that takes a numpy RNG and table as an
            argument and will return a np.float32 array that can be added to a
            member to produce a mutation.
        """
        pass


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
