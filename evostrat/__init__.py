"""
These classes implement simple evolutionary strategies for distribution over
a large cluster using MPI (mpi4py). It is very efficient and scales to 
millions of parameters. It uses numpy float32 types to reduce memory consumption
and relies heavily on numpy vectorization.

Even for many millions of parameters the total compute time of the ES amounts
only to a few minutes of total wall-time. Even for quickly evaluated fitness
functions the ES will likely only amount to a small portion of total wall-time.

The BasicES and BasicBoundedES are less efficient as they draw new random
values each iteration. The RandNumTableES and BoundedES use large static
tables of random numbers, only drawing slices each iteration, allowing a
10X+ speedup. Make sure to adjust table size to fit within the memory of each
node. If a node has 64GB and 32 ranks, then make sure to allocate less than
(64GB / 32) GB of RNG table memory (minus 32*size_of_parameters).

Additionally, all operations in RandNumTableES/BoundedES are done in-place, 
so no intermediate arrays are allocated. BasicES/BasicBoundedES have to
generate new random numbers so array allocation has to occur each iteration.

In order to smooth out correlations in RNG draws, different strides are randomly
drawn, and different sets of parameters can be selected each iteration for
mutation. So long as Table >> #pars it is unlikely to effect the ES.
Even a 20M table with 1M parameters doesn't inhibit ES progress.

If you subclass these ES, make sure any arithmetic you do uses dtype=float32 to
prevent time lost with type conversions.

The GA is different from the ES. 
The GA is modular and requires a mutation and member generating function
Genotypes can be variable in length.
If the GA is ever saved and reloaded the member generating function
and mutation function and objective function need to be assigned again.
The former two have corresponding methods. The object can be set directly
when calling the GA (just as in the ES)

This GA maintains a list of seeds for each member of the population which
denotes its construction and mutation history. When members of the population
are replaced this list is transfered between nodes and new mutations under
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

Mutation rate and other mulation related parameters must be specified
in the mutation_function and are not part of the GA itself. These
can be provided as mutation_function_args, or a partial function can be
passed to the GA
"""

from .evostrat import BasicES
from .evostrat import BoundedBasicES
from .evostrat import RandNumTableES
from .evostrat import BoundedRandNumTableES
from .genalg import BasicGA
from .genalg import BoundedBasicGA
from .genalg import RandNumTableGA
from .genalg import BoundedRandNumTableGA
from .genalg import real_member_generator
from .genalg import real_mutator
