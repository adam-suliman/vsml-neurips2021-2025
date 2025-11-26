import jax
import functools
from mpi4py import MPI
import mpi4jax


def only_rank(target_rank: int):
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    def decorator(fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            if my_rank == target_rank:
                return fn(*args, **kwargs)
            else:
                return None
        return decorated
    return decorator


def tree_all_reduce(tree, comm, **kwargs):
    def reduce_leaf_func(leaf):
        return mpi4jax.allreduce(leaf, **kwargs)
    return jax.tree_map(reduce_leaf_func, tree)


def tree_bcast(tree, comm, **kwargs):
    def reduce_leaf_func(leaf):
        return mpi4jax.bcast(leaf, **kwargs)
    return jax.tree_map(reduce_leaf_func, tree)
