"""
These classes implement simple evolutionary strategies for distribution over
a large cluster using MPI (mpi4py). It is very efficient and scales to 
millions of parameters. It uses numpy float32 types to reduce memory consumption
and relies heavily on numpy vectorization.

Even for many millions of parameters the total compute time of the ES amounts
only to a few minutes of total wall-time. Even for quickly evaluated fitness
functions the ES/GA will likely only amount to a small portion of total wall-time.

The BasicES and BasicBoundedES are less efficient as they draw new random
values each iteration. The RandNumTableES and BoundedES use large static
tables of random numbers, only drawing slices each iteration, allowing a
10X+ speedup (and a considerable speedup over the original table implementation
in the papers). Make sure to adjust table size to fit within the memory of each
node. If a node has 64GB and 32 ranks, then make sure to allocate less than
(64GB / 32) GB of RNG table memory (minus 32*size_of_parameters).

Additionally, all operations in RandNumTableES/BoundedES are done in-place, 
so no intermediate arrays are allocated. BasicES/BasicBoundedES have to
generate new random numbers so array allocation has to occur each iteration.

An additional mechanism was adopted to improve performance when using the
random number table. Instead of drawing random elements, which still requires
a large integer array to be created, random slices of both the parameters and
the random number table are drawn. This requires only drawing 6 random numbers
instead of a million+. In order to smooth out correlations in RNG draws,
different strides are randomly drawn, and different slices through the parameters
are selected each iteration for mutation. So long as Table >> #pars it is
unlikely to effect the ES. Even a 20M table with 1M parameters doesn't
inhibit ES/GA progress.

If you subclass these ES, make sure any arithmetic you do uses dtype=float32 to
prevent time lost with type conversions.

The GA is similar to the ES. It also can use random number tables, but instead
of a centroid, the GA keeps a log of a list of seeds for each member of the
population which denotes its construction and mutation history.
When members of the population are replaced this list is transferred between
nodes and new mutations under the unique seed of that node are applied,
branching the lineage of that member. At the end of execution the whole
lineage of all members are passed to the root node for logging. Also, the best
member is always retained in the root node for saving as well.

The GA as described in the paper does not implement a random number table (to
my knowledge) for parameter perturbations.

MPI broadcasts remain isolated to the fitness function, but site2site
messages are sent between ranks that need a new member, which will be 
drawn uniformly at random from available parents.

An elite fraction can be set which determines what fraction of the population
is withheld from mutation. The top best % of the population will go onto
the next unchanged.

Both truncated selection and stochastic universal sampling (based on linear
rank order) are supported for GAs.
"""

from .evostrat import BasicES
from .evostrat import BoundedBasicES
from .evostrat import RandNumTableES
from .evostrat import BoundedRandNumTableES
from .genalg import TruncatedRandNumTableGA
from .genalg import SusRandNumTableGA
from .genalg import TruncatedRealMutatorGA
from .genalg import TruncatedAnnealingRandNumTableGA
from .genalg import SusAnnealingRandNumTableGA
from .genalg import RescaledSusAnnealingRandNumTableGA
from .genalg import BoundedSusAnnealingRandNumTableGA
