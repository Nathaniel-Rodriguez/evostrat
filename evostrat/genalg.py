import numpy as np
import pickle
import random
from functools import partial
from .sliceops import *
from abc import ABC, abstractmethod
from collections import Counter


class BaseGA(ABC):
    """
    Contains basic members and functions common to derived GA classes

    :note: self._member is not initialized in BaseGA, as mutator classes may
    need to initialize their state before mutations can be drawn to create
    the first member
    """

    def __init__(self, **kwargs):
        """
        :param initial_guess: a numpy array from which to generate perturbations from
            should be 1d numpy float32
        :param objective: the object function, returns a cost scalar
        :param obj_kwargs: key word arguments of the objective function (default {})
        :param verbose: True/False whether to print output (default False)
        :param elite_fraction: fraction of best performing that go unmutated to
            next generation (default 0.1)
        :param num_elite: same as above, can set one or the other
        :param seed: used to generate all seeds and random values (default: 1)
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

        if 'elite_fraction' in kwargs:
            self._elite_fraction = kwargs.get('elite_fraction', 0.1)
            self._num_elite = int(self._size * self._elite_fraction)
        elif 'num_elite' in kwargs:
            self._num_elite = kwargs.get('num_elite', 1)
            self._elite_fraction = self._num_elite / self._size

        self._initial_guess = kwargs['initial_guess'].astype(dtype=np.float32)
        self._generation_number = 0
        self._cost_history = []

        self._global_seed = kwargs.get('seed', 1)
        self._py_rng = random.Random(self._global_seed)
        self._global_rng = np.random.RandomState(self._global_seed)
        self._initial_seed_list = self._py_rng.sample(range(self._max_seed),
                                                      self._size)
        self._seed_seed_list = self._py_rng.sample(range(self._max_seed),
                                                   self._size)
        self._mutation_rng = np.random.RandomState(self._initial_seed_list[self._rank])
        self._seed_rng = np.random.RandomState(self._seed_seed_list[self._rank])
        self._member_genealogy = [self._initial_seed_list[self._rank]]
        self._population_genealogy = [[] for i in range(self._size)]

    def __call__(self, num_iterations, objective=None, kwargs=None,
                 save=True, save_every=None, save_filename="test.ga"):
        """
        Runs the genetic algorithm. It first sets the objective (since objectives
        are usually functions they aren't pickled with the GA, so have to be
        set each time the GA is loaded). It then runs for the designated number
        of iterations. At the end if shares the genealogy with the root node (
        usually node 0).

        :param num_iterations: how many generations it will run for
        :param objective: a full or partial version of function
        :param kwargs: key word arguments for additional objective parameters
        :param save: T/F, whether to save after the run is complete
        :param save_every: None or integer, shares genealogy and saves every X
            iterations.
        :param save_filename: name of savefile, default: test.ga)
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
                print("Generation:", self._generation_number, flush=True)
            self._update(partial_objective)

            # Shares genealogy and saves using root node.
            if not (save_every is None):
                if ((i % save_every) == 0) and (i != 0):
                    self._pre_save_configure()
                    self.save(save_filename)

            self._generation_number += 1

        if save:
            self._pre_save_configure()
            self.save(save_filename)

    def _pre_save_configure(self):
        """
        Carries out operations (such as MPI calls) to make sure all necessary
        information is saved. As ROOT is the only saving member.
        """
        self._share_genealogy()

    def member_generator(self, rng):
        """
        Uses initial guess to build initial members. This results in a
        somewhat different distribution of initial parameters as the BasicGA's
        draw.

        Draws from the perturbation distribution. If the member hasn't
        been created yet it creates a new one from initial guess, else it
        uses assignment to copy. This ensures allocation only occurs during
        initialization and not during the run.

        :return 1d numpy float32 array that represents member
        """

        if not hasattr(self, '_member'):
            self._member = self._initial_guess.copy()
        else:
            self._member[:] = self._initial_guess[:]

        # Mutation is applied here, as fitness is calculated first and mutation last
        self.mutator(self._member, rng)

        return self._member

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

    def _update(self, objective):
        """
        Method that updates the genetic algorithm using the objective.
        The objective is a partial function created from whatever kwargs were
        given either upon instantiation of object or __call__
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
        Creates a member of the population given a see list
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

    @abstractmethod
    def _construct_message_list(self, l2g_ranks):
        """
        Builds a list of tuples with (send_rank, recv_rank) pairs
        Every rank must participate in building this so the global_rng
        stays in sync with everyone and so that everyone has the same
        messenger list.

        Construct message should carry out member selection from the l2g_ranks
        list. Therefore, this function should define the algorithms selecion
        or replacement method.

        :param l2g_ranks: A list of ranks sorted by cost. This means the first
            value is the highest performing rank while the last value is the
            lowest performing rank.
        :return: A list of messages
        """
        pass

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

    def _mutate_population(self, l2g_ranks):
        """
        Preserve the top _num_elite members, mutate the rest
        """
        for cost_index, rank in enumerate(l2g_ranks):
            if cost_index >= self._num_elite:
                if rank == self._rank:
                    # draw mutation seed
                    seed = self._seed_rng.randint(0, self._max_seed)
                    self._member_genealogy.append(seed)

                    # apply mutation
                    self._mutation_rng.seed(seed)
                    self._member = self.mutator(self._member,
                                                self._mutation_rng)

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
            pickle.dump(self, pickled_obj_file, protocol=pickle.DEFAULT_PROTOCOL)
            pickled_obj_file.close()

    def __getstate__(self):

        state = {"_verbose": self._verbose,
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
                 "_population_genealogy": self._population_genealogy,
                 "_initial_guess": self._initial_guess}

        return state

    def __setstate__(self, state):
        """
        Member genealogy must be reset to the corresponding rank, since
        only rank0 is saved.
        """
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

        # Reassign member genealogies and make seed rng recent
        self._seed_rng = np.random.RandomState(self._seed_seed_list[self._rank])
        for i in range(self._generation_number):
            self._seed_rng.randint(0, self._max_seed)
        self._member_genealogy = self._population_genealogy[self._rank][:]


class TruncatedSelection(BaseGA):
    """
    Truncated selection works be selection the top # parents members and then
    replacing all other members with a uniform random draw from those top
    members.

    This introduces a new parameter: parent_fraction (num_parents)
    The smaller the num_parents, the string the selective force, as fewer members
    are preserved.
    """
    def __init__(self, **kwargs):
        """
        :param parent_fraction: fraction of number of members that will be
            parents for the truncated selection
        :param num_parents: see above (can be set instead of parent_fraction,
            only set one or the other)
        :param kwargs: BaseGA kwargs
        """
        super().__init__(**kwargs)

        if 'parent_fraction' in kwargs:
            self._parent_fraction = kwargs.get('parent_fraction', 0.3)
            self._num_parents = int(self._size * self._parent_fraction)
        elif 'num_parents' in kwargs:
            self._num_parents = kwargs.get('num_parents', int(self._size / 2))
            self._parent_fraction = self._num_parents / self._size

        if self._num_elite > self._num_parents:
            raise AssertionError("Number of elite has to be less than the"
                                 " number of parents")

    def _chose_rank_by_performance(self, l2g_ranks):
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
        for cost_index, rank in enumerate(l2g_ranks):
            # The higher the cost_index the lower performing the member
            # If this exceeds the num_parents, then the member is chosen
            # for replacement by a member of the parents (all low cost_index
            # members)
            if cost_index >= self._num_parents:
                # randomly pick who to copy from higher rank members
                chosen_rank = self._chose_rank_by_performance(l2g_ranks)
                messenger_list.append((chosen_rank, rank))

        return messenger_list

    def __getstate__(self):
        state = super().__getstate__()
        state["_num_parents"] = self._num_parents
        state["_parent_fraction"] = self._parent_fraction

        return state


class SusSelection(BaseGA):
    """
    Implements stochastic universal sampling with linear rank-based selection.
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: arguments for BaseGA
        """
        super().__init__(**kwargs)
        self._ranks = range(self._size)
        self._cost_rank_sum = self._size * (self._size + 1) / 2
        self._number_to_select = self._size - self._num_elite
        self._selection_probabilities = [self.linearly_scaled_member_rank(i)
                                         for i in self._ranks]
        self._num_expected_copies = [probability * self._number_to_select
                                     for probability in self._selection_probabilities]

    def linearly_scaled_member_rank(self, cost_index):
        """
        Scales the rank of an individual (cost_index)
        :param cost_index: 1 is best
        :return: scaled_cost_rank
        """
        return (self._size - cost_index) / self._cost_rank_sum

    def sus(self, rng, sorted_ranks):
        """
        Implements stochastic universal sampling and returns a list of
        selected ranks and is the same size as the population. There can be
        duplicate ranks which get selected more than once.

        Implements SUS. Enforces elitism.
        :param rng: a random number generator from which to determine the
            selected individuals
        :param sorted_ranks: a np.array of ranks that match the selection
            probabilities. These should be sorted from least->greatest cost.
        :return: a list of selected members to replace the population
        """

        selected = list(sorted_ranks[:self._num_elite])
        rv = rng.rand()
        count = 0
        for i in range(self._size):
            count += self._num_expected_copies[i]
            while rv < count:
                rv += 1
                selected.append(sorted_ranks[i])

        return selected

    def match_remainder(self, selected_rank_list):
        """
        Creates a message list based on a list of selected ranks. First,
        a matching process eliminates messaging between selected members and
        ranks that already have that member, keeping only duplicates. These
        duplicates represent the members that actually need copying between
        ranks. A list of remaining ranks available for being copied too is
        created and then assigned source ranks. This process prevents the issue
        of having a member replaced when that member is still needed later.

        :param selected_rank_list: list with ranks as elements
        :return: a list of messages
        """

        # Remove matched ranks where no move is required
        selected_rank_count = Counter(selected_rank_list)
        selected_rank_count.subtract(self._ranks)
        remaining_selected_ranks = list(selected_rank_count.elements())
        remaining_available_ranks = [key for key in selected_rank_count.keys()
                                     if selected_rank_count[key] < 0]

        # Create messages for remaining ranks
        return [(remaining_selected_ranks[i], remaining_available_ranks[i])
                for i in range(len(remaining_available_ranks))]

    def _construct_message_list(self, l2g_ranks):
        """
        Builds a list of tuples with (send_rank, recv_rank) pairs
        Every rank must participate in building this so the global_rng
        stays in sync with everyone and so that everyone has the same
        messenger list

        Implements Rank-based Stochastic Universal Sampling
        """

        # Implement selection process
        selected_rank_list = self.sus(self._global_rng, l2g_ranks)

        # Create messages from selected
        return self.match_remainder(selected_rank_list)

    def __getstate__(self):
        state = super().__getstate__()
        state["_ranks"] = self._ranks
        state["_cost_rank_sum"] = self._cost_rank_sum
        state["_number_to_select"] = self._number_to_select
        state["_selection_probabilities"] = self._selection_probabilities
        state["_num_expected_copies"] = self._num_expected_copies

        return state


class RealMutator(BaseGA):
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

    def __init__(self, **kwargs):
        """
        :param sigma: the standard deviation of the normal distribution of
            perturbations applied as mutations
        :param elite_fraction: the fraction of members to withhold from mutation
        :param parent_fraction: the fraction of the population to maintain. remaining
            members are culled and replaced with more successful parents.
            This is the truncation selection process.
        """

        super().__init__(**kwargs)

        self._sigma = kwargs.get('sigma', 1.0)
        self._member_size = len(self._member)
        self._member = self._make_member(self._mutation_rng,
                                         self._member_genealogy)

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

    def __getstate__(self):
        state = super().__getstate__()
        state["_sigma"] = self._sigma
        state["_member_size"] = self._member_size
        state["_member"] = self._member

        return state

    def __setstate__(self, state):
        """
        The members need to be generated from the member's genealogy,
        as only the rank=0 member is actually saved.
        The member genealogy is set to the proper rank by the baseGA
        """
        super().__setstate__(state)
        self._member = self._make_member(self._mutation_rng,
                                         self._member_genealogy)


class RandNumTableModule(BaseGA):
    """
    A basic GA that uses a cached random number table. This speeds up
    the mutation process considerably at the cost of memory. By using a table
    the algorithm can be made several times faster as random number generation
    is expensive.

    The table is a numpy array of normally distributed values: N(0,1).
    """
    def __init__(self, **kwargs):
        """
        :param sigma: the standard deviation of mutation perturbations
        :param rand_num_table_size: the number of elements in the random table
        :param max_table_step: the maximum random stride for table slices
        :param max_param_step: the maximum step size for parameter slices
        :param kwargs: parameters from BasicGA

        Parameter slices are always the same size as the size of the member, but
        by slicing it randomly you mitigate correlations in table slices by
        shifting which parameters those mutations are applied too, allowing the
        use of a smaller random table.
        """
        super().__init__(**kwargs)
        self._rand_num_table_seed = kwargs.get('rand_num_table_seed',
                                               self._py_rng.sample(
                                                    range(self._max_seed), 1)[0])
        self._table_rng = np.random.RandomState(self._rand_num_table_seed)
        self._rand_num_table_size = kwargs.get("rand_num_table_size", 20000000)
        self._rand_num_table = self._table_rng.randn(self._rand_num_table_size)
        self._sigma = kwargs['sigma']
        self._member_size = len(self._initial_guess)
        self._max_table_step = kwargs.get("max_table_step", 5)
        self._max_param_step = kwargs.get("max_param_step", 1)
        self._rand_num_table *= self._sigma
        self._member = self._initialize_member()

    def _initialize_member(self):
        """
        For initializing this rank's member
        :return: a new array
        """

        return self._make_member(self._mutation_rng, self._member_genealogy)

    def _draw_random_parameter_slices(self, rng):
        """
        Chooses a constrained slice subset of the parameters (start, stop, step)
        to give roughly num_mutations perturbations (less if overlap if
        step is too large)
        """

        return random_slices(rng, self._member_size, self._member_size,
                             self._max_param_step)

    def _draw_random_table_slices(self, rng):
        """
        Chooses a constrained slice subset of the RN table (start, stop, step)
        to give roughly num_mutations random numbers (less if overlap if
        step is too large)
        """

        return random_slices(rng, self._rand_num_table_size,
                             self._member_size, self._max_table_step)

    def mutator(self, member, rng):
        """
        Can be used with BasicGAs

        :param member: a member to mutate
        :param rng: a numpy randomState
        :return: reference to member that is changed in-place
        """

        param_slices = self._draw_random_parameter_slices(rng)
        table_slices = self._draw_random_table_slices(rng)
        param_slices, table_slices = match_slices(param_slices, table_slices)
        multi_slice_add(member, self._rand_num_table, param_slices, table_slices)

        return member

    def __getstate__(self):
        state = super().__getstate__()

        state["_sigma"] = self._sigma
        state["_member_size"] = self._member_size
        state["_rand_num_table_size"] = self._rand_num_table_size
        state["_max_table_step"] = self._max_table_step
        state["_max_param_step"] = self._max_param_step
        state["_rand_num_table_seed"] = self._rand_num_table_seed
        state["_member"] = self._member

        return state

    def __setstate__(self, state):
        """
        The RN table has to be re-created and the members for each rank need
        to be generated from the member's genealogy.
        The member genealogy is set to the proper rank by the baseGA
        """
        super().__setstate__(state)
        self._table_rng = np.random.RandomState(self._rand_num_table_seed)
        self._rand_num_table = self._table_rng.randn(self._rand_num_table_size)
        self._rand_num_table *= self._sigma
        self._member = self._initialize_member()


class AnnealingModule(BaseGA):
    """
    A version of the BaseGA that uses simulated annealing.
    Can be inherited to give access to cooling schedules.

    The mutator should be defined so that
    self._generation_number is fed to the schedule to output a scaling factor
    for the mutation rate or step size (usually sigma).
    """

    def __init__(self, **kwargs):
        """
        :param cooling_schedule: a function or string. If string, the class
            currently supports:
                "exp" : "initial_temperature", "cooling_factor"

        :param cooling_schedule_kwargs: default(None), dictionary of key word
            arguments for the schedule.
        :param kwargs: See BasicGA parameters
        """
        super().__init__(**kwargs)
        self._cooling_schedule = kwargs.get('cooling_schedule', "exp")
        self._cooling_schedule_kwargs = kwargs.get('cooling_schedule_kwargs', None)
        self.assign_cooling_schedule(self._cooling_schedule,
                                     self._cooling_schedule_kwargs)

    def _exponential_cooling_schedule(self, time_step, initial_temperature=1.0,
                                      cooling_factor=1.0):
        """
        Exponential decay cooling schedule
        :param time_step: current time-step
        :param initial_temperature: initial temperature (defaults 1.0)
        :param cooling_factor: A value between [0,1] (defaults 1.0)
        :return: New temperature
        """

        if (cooling_factor < 0) or (cooling_factor > 1):
            raise AssertionError("Invalid input: Cooling factor must be"
                                 " between 0 and 1.")
        if initial_temperature < 0:
            raise AssertionError("Invalid input: Initial temperature must be"
                                 " greater > 0")

        return initial_temperature * (cooling_factor ** time_step)

    def assign_cooling_schedule(self, cooling_schedule,
                                cooling_schedule_kwargs=None):
        """
        Assigns a cooling schedule to the evolutionary algorithm
        :param cooling_schedule: "string" or function. Supports: "exp".
        :param cooling_schedule_kwargs: key word arguments for cooling schedule
        :return: None
        """

        if (cooling_schedule_kwargs is None) and \
                (self._cooling_schedule_kwargs is None):
            self._cooling_schedule_kwargs = {}

        elif cooling_schedule_kwargs is not None:
            self._cooling_schedule_kwargs = cooling_schedule_kwargs

        if cooling_schedule == "exp":
            self.schedule_type = "exp"
            self._cooling_schedule = partial(self._exponential_cooling_schedule,
                                             **self._cooling_schedule_kwargs)
        else:
            self.schedule_type = "external"
            self._cooling_schedule = partial(cooling_schedule,
                                             **self._cooling_schedule_kwargs)

    def __getstate__(self):
        state = super().__getstate__()
        state["_cooling_schedule_kwargs"] = self._cooling_schedule_kwargs
        state["schedule_type"] = self.schedule_type

    def __setstate__(self, state):
        super().__setstate__(state)

        if self.schedule_type == "exp":
            self.assign_cooling_schedule("exp", self._cooling_schedule_kwargs)
        else:
            print("Warning: External cooling schedule defined, needs to be"
                  " set before running evolution.")


class AnnealingRandNumTableGA(RandNumTableModule, AnnealingModule):
    """
    Annealing requires a lot of specialization.
    """

    def __init__(self, **kwargs):
        """
        We need to initialize a
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._perturbation = np.zeros(self._member_size, dtype=np.float32)
        self._member_temperatures = [self._cooling_schedule(self._generation_number)]
        self._population_temperatures = [[] for i in range(self._size)]

    @property
    def best(self):
        """
        :return: generates and returns the best member of the population.
            Only Rank==0 should be accessing this property.
        """

        try:
            return self._make_member(self._mutation_rng,
                                     self._population_genealogy[
                                        np.argsort(self._cost_history[-1])[0]],
                                     self._population_temperatures[
                                         np.argsort(self._cost_history[-1])[0]])
        except IndexError:
            raise IndexError("No score or population genealogy from which"
                             "to generate best. Run optimization first.")

    @property
    def population(self):
        """
        :return: a list of all members of the current population.
            Only Rank==0 should be accessing this property.
        """

        try:
            return [self._make_member(self._mutation_rng, member,
                                      self._population_temperatures[i])
                    for i, member in enumerate(self._population_genealogy)]
        except IndexError:
            raise IndexError("No score or population genealogy"
                             "Run optimization first.")
        except TypeError:
            raise TypeError("Need to set mutation and member generating functions")

    def _initialize_member(self):
        """
        For initializing this rank's member
        :return: a new array
        """

        return self._make_member(self._mutation_rng, self._member_genealogy,
                                 self._member_temperatures)

    def member_generator(self, rng, *args, **kwargs):
        """
        Uses initial guess to build initial members. This results in a
        somewhat different distribution of initial parameters as the BasicGA's
        draw.

        Draws from the perturbation distribution. If the member hasn't
        been created yet it creates a new one from initial guess, else it
        uses assignment to copy. This ensures allocation only occurs during
        initialization and not during the run.

        :return 1d numpy float32 array that represents member
        """

        if not hasattr(self, '_member'):
            self._member = self._initial_guess.copy()
        else:
            self._member[:] = self._initial_guess[:]

        # Mutation is applied here, as fitness is calculated first and mutation last
        self.mutator(self._member, rng, kwargs['temperature'])

        return self._member

    def _pre_save_configure(self):
        """
        Add temperature sharing before saving so that ROOT can reconstruct each
        member.
        """
        super()._pre_save_configure()
        self._share_temperatures()

    def _dispatch_temperatures(self, messenger_list):
        """
        Iterates through a messenger list and sends/receives the genealogies
        for each pair of ranks
        """
        for messenger in messenger_list:
            # something went terribly wrong if node is sending to itself
            assert (messenger[0] != messenger[1])

            # send/recv temperatures
            if self._rank == messenger[0]:
                self._comm.send(self._member_temperatures, dest=messenger[1])

            if self._rank == messenger[1]:
                self._member_temperatures = self._comm.recv(source=messenger[0])

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
        self._dispatch_temperatures(messenger_list)
        self._construct_received_members(messenger_list)
        self._mutate_population(l2g_ranks)

    def _mutate_population(self, l2g_ranks):
        """
        Preserve the top _num_elite members, mutate the rest
        """
        for cost_index, rank in enumerate(l2g_ranks):
            if cost_index >= self._num_elite:
                if rank == self._rank:
                    # get temperature
                    temperature = self._cooling_schedule(self._generation_number)
                    self._member_temperatures.append(temperature)

                    # draw mutation seed
                    seed = self._seed_rng.randint(0, self._max_seed)
                    self._member_genealogy.append(seed)

                    # apply mutation
                    self._mutation_rng.seed(seed)
                    self._member = self.mutator(self._member,
                                                self._mutation_rng,
                                                temperature=temperature)

    def mutator(self, member, rng, *args, **kwargs):
        """
        :param member: array to perturb
        :param rng: seeded numpy rng
        :param temperature: the annealing scaling factor
        :return: member (reference to member passed in)
        """

        param_slices = self._draw_random_parameter_slices(rng)
        table_slices = self._draw_random_table_slices(rng)
        param_slices, table_slices = match_slices(param_slices, table_slices)
        # We assign the table values to the perturbation member first
        multi_slice_assign(self._perturbation, self._rand_num_table,
                           param_slices, table_slices)
        # Apply the annealing scaling factor, i.e. the temperature
        np.multiply(self._perturbation, kwargs['temperature'],
                    out=self._perturbation)
        # With perturbation member complete, we can add to member
        np.add(member, self._perturbation, out=member)

        return member

    def _make_member(self, rng, seed_list, *args, **kwargs):
        """
        :param rng: numpy rng
        :param seed_list: genealogy of member
        :param temperatures: temperature history of member
        :return: a new member
        """
        rng.seed(seed_list[0])
        new_member = self.member_generator(rng, temperature=kwargs['temperatures'][0])
        for i, seed in enumerate(seed_list[1:]):
            rng.seed(seed)
            new_member = self.mutator(new_member, rng,
                                      temperature=kwargs['temperatures'][i+1])

        return new_member

    def _construct_received_members(self, messenger_list):
        """
        Iterates through the messenger list and for ranks that recieved
        a genealogy it builds a new member from it and replaces the current
        member
        """
        for messenger in messenger_list:
            if self._rank == messenger[1]:
                self._member = self._make_member(self._mutation_rng,
                                                 self._member_genealogy,
                                                 self._member_temperatures)

    def _share_temperatures(self):
        """
        Shares each members temperature list, which corresponds with their
        genealogies so we know the multiplier for each perturbation.
        """

        self._population_temperatures = self._comm.gather(self._member_temperatures,
                                                          root=0)

    def __getstate__(self):
        state = super().__getstate__()
        state['_member_temperatures'] = self._member_temperatures
        state['_population_temperatures'] = self._population_temperatures

    def __setstate__(self, state):
        super().__setstate__(state)
        self._perturbation = np.zeros(self._member_size, dtype=np.float32)


class TruncatedAnnealingRandNumTableGA(AnnealingRandNumTableGA, TruncatedSelection):
    """
    Implements truncated random selection and a real valued random number table
    GA.

    :param initial_guess: numpy float32 array from which to draw perturbations around
    :param objective: the object function, returns a cost scalar
    :param obj_kwargs: key word arguments of the objective function (default {})
    :param verbose: True/False whether to print output (default False)
    :param elite_fraction: fraction of best performing that go unmutated to
        next generation (default 0.1)
    :param num_elite: same as above, can set one or the other
    :param seed: used to generate all seeds and random values (default: 1)
    :param parent_fraction: fraction of number of members that will be
        parents for the truncated selection
    :param num_parents: see above (can be set instead of parent_fraction,
        only set one or the other)
    :param initial_guess: numpy array from which to draw perturbations around
    :param sigma: the standard deviation of mutation perturbations
    :param rand_num_table_size: the number of elements in the random table
    :param max_table_step: the maximum random stride for table slices
    :param max_param_step: the maximum step size for parameter slices
    :param cooling_schedule: a function or string. If string, the class
            currently supports:
            "exp" : "initial_temperature", "cooling_factor"

    :param cooling_schedule_kwargs: default(None), dictionary of key word
        arguments for the schedule.
    """
    pass


class SusAnnealingRandNumTableGA(AnnealingRandNumTableGA, SusSelection):
    """
    Implements stochastic universal sampling and a real valued random number
    table GA.

    :param initial_guess: numpy float32 array from which to draw perturbations around
    :param objective: the object function, returns a cost scalar
    :param obj_kwargs: key word arguments of the objective function (default {})
    :param verbose: True/False whether to print output (default False)
    :param elite_fraction: fraction of best performing that go unmutated to
        next generation (default 0.1)
    :param num_elite: same as above, can set one or the other
    :param seed: used to generate all seeds and random values (default: 1)
    :param sigma: the standard deviation of mutation perturbations
    :param rand_num_table_size: the number of elements in the random table
    :param max_table_step: the maximum random stride for table slices
    :param max_param_step: the maximum step size for parameter slices
    :param cooling_schedule: a function or string. If string, the class
        currently supports:
            "exp" : "initial_temperature", "cooling_factor"

    :param cooling_schedule_kwargs: default(None), dictionary of key word
        arguments for the schedule.
    """
    pass


class TruncatedRandNumTableGA(RandNumTableModule, TruncatedSelection):
    """
    Implements truncated random selection and a real valued random number table
    GA.

    :param initial_guess: numpy float32 array from which to draw perturbations around
    :param objective: the object function, returns a cost scalar
    :param obj_kwargs: key word arguments of the objective function (default {})
    :param verbose: True/False whether to print output (default False)
    :param elite_fraction: fraction of best performing that go unmutated to
        next generation (default 0.1)
    :param num_elite: same as above, can set one or the other
    :param seed: used to generate all seeds and random values (default: 1)
    :param parent_fraction: fraction of number of members that will be
        parents for the truncated selection
    :param num_parents: see above (can be set instead of parent_fraction,
        only set one or the other)
    :param initial_guess: numpy array from which to draw perturbations around
    :param sigma: the standard deviation of mutation perturbations
    :param rand_num_table_size: the number of elements in the random table
    :param max_table_step: the maximum random stride for table slices
    :param max_param_step: the maximum step size for parameter slices
    """
    pass


class SusRandNumTableGA(RandNumTableModule, SusSelection):
    """
    Implements stochastic universal sampling and a real valued random number
    table GA.

    :param initial_guess: numpy float32 array from which to draw perturbations around
    :param objective: the object function, returns a cost scalar
    :param obj_kwargs: key word arguments of the objective function (default {})
    :param verbose: True/False whether to print output (default False)
    :param elite_fraction: fraction of best performing that go unmutated to
        next generation (default 0.1)
    :param num_elite: same as above, can set one or the other
    :param seed: used to generate all seeds and random values (default: 1)
    :param sigma: the standard deviation of mutation perturbations
    :param rand_num_table_size: the number of elements in the random table
    :param max_table_step: the maximum random stride for table slices
    :param max_param_step: the maximum step size for parameter slices
    """
    pass


class TruncatedRealMutatorGA(RealMutator, TruncatedSelection):
    """
    Implements truncated random selection and uses real valued perturbations
    drawn during optimization.

    :param initial_guess: numpy float32 array from which to draw perturbations around
    :param objective: the object function, returns a cost scalar
    :param obj_kwargs: key word arguments of the objective function (default {})
    :param verbose: True/False whether to print output (default False)
    :param elite_fraction: fraction of best performing that go unmutated to
        next generation (default 0.1)
    :param num_elite: same as above, can set one or the other
    :param seed: used to generate all seeds and random values (default: 1)
    :param parent_fraction: fraction of number of members that will be
        parents for the truncated selection
    :param num_parents: see above (can be set instead of parent_fraction,
        only set one or the other)
    :param sigma: the standard deviation of the normal distribution of
        perturbations applied as mutations
    :param member_size: the number of parameters for each member
    :param elite_fraction: the fraction of members to withhold from mutation
    :param parent_fraction: the fraction of the population to maintain. remaining
        members are culled and replaced with more successful parents.
        This is the truncation selection process.
    """
    pass
