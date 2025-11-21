# Numpy Buffer Map
A way to define arrays of an algorithm through a dependency tree, and model them by arrays that can and cannot share the same memory.

This library can:
- Reduce the total memory requirements of a procedure.
- Drastically improve the performance of compiled python, including but not limited to: Numba, Cython, Taichi.
- Express a unified outline of memory dependencies, useful for spotting issues or finding improvements in a complex procedure.

![distinct](renders/algo_mem_distinct.png)

After defining a tree, the total buffer size is calculated, allocated, and arrays are initialized at the offsets corresponding to
the shared and distinct nodes.

See the [demo](demo.md) for use and applications.  
See [extensions](demo.md) for future potential development directions.  

In a sense `bmap` does a portion of [JAX](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates)'s
automatic memory management. Only `bmap` is still an external tool and can be used for any array compatible language.
