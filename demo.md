```python

```

# Usage
We will begin our example by abbreviating the typical considerations in developing a numerical algorithm on CPU top to bottom:
1. We have RAM. The abstraction is usually referred to as the heap.
2. CPU caches usually L1-3. May be referred to as the stack.
3. The Core registry, where data is loaded by the (two) read ports, operated on by the other ports, and written out by the often singular write port. Modern ports can handle (roughly) one AVX512 instruction set per core cycle collectively as a single unit. [For more info on this](https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(client)#Scheduler_Ports_.26_Execution_Units).
    - Just from the different amount of read and write ports we can see that instructions groups will have bias. It means that if we can coax the compiler into utilizing every port of the core on vectorization, we could in theory achieve the most operation dense and fast kernel.
    - This leads to some interesting micro benchmarks on kernels that fully or partially subscribe to the [triad](https://www.intel.com/content/www/us/en/developer/articles/technical/optimizing-memory-bandwidth-on-stream-triad.html) template. Performing much better per operation than simpler kernels. However this is beyond the scope of this demo.  

The target of performance optimization in this library is #1 the heap. In compiled code, calling `malloc` in the default manner and releasing buffer allocations is slow; skipping the discussion on why that is, we realize a priority to reuse memory for dense numerical array operations. Following this reasoning we can group buffer memory into two parts:  
| Shareable Memory                                                                        | Distinct Memory                                                                             |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| All temporary arrays, e.g. used as scratch<br>and that do not require simultaneous use. | All arrays that contain data needed over<br>the lifetime of execution, e.g. looping bodies. |

What we find is that any *shareable memory* arrays can overlap and use the same buffer, except for the section holding *distinct memory*. While distinct memory holds arrays that cannot be shared with each other or the buffer holding the shared memory. Infact we can call the shared memory node an array (member) of the distinct node. These dependencies have the workings of a tree, and we can in fact represent this with **Numpy Buffermap** like so:





```python
import np_buffermap as npb
import numpy as np

# --- `ar_spec` returns an ArrayNode which contains all the arguments necessary to construct an array. 
#arrays that save their data
dist1=npb.ar_spec((69,),name='D1') #distinct array 1
dist2=npb.ar_spec((3,5,8),np.float32,name='D2',) #distinct array 2
dist3=npb.ar_spec((15,15),np.float32,name='D3',) #distinct array 3

#scratch arrays.
s1=npb.ar_spec((5,4,3),name='S1')
s2=npb.ar_spec((20,),name='S2')
s3=npb.ar_spec((8,9),np.int32,name='S3',order='F')

# --- Now we utilize ContainerNode which come in two flavors: 
#SharedNode sb_node - shared buffer node, 
#DistinctNode db_node - distinct buffer node.
mem_map=npb.db_node(dist1,dist2,dist3,name='Algo Mem',)
shared_mem=npb.sb_node(s1,s2,s3,name='Shared Mem')
mem_map.add(shared_mem)

print(mem_map)
#Calculates dimensions and ranges of buffer offsets, and places that info into the tree.
npb.build_bmap(mem_map,align=npb.BufferAlign.BYTE) #byte is equivalent to no extra padding of array alignments.


ot=npb.bmap_pyvis(mem_map,with_offsets=True,)
ot.prep_notebook()
ot.show('renders/algo_mem_1.html')

```

    |Node Algo Mem rule=D children=4|
    ╠══ |Array D1 (69) float64|
    ╠══ |Array D2 (3, 5, 8) float32|
    ╠══ |Array D3 (15, 15) float32|
    ╚══ |Node Shared Mem rule=S children=3|
        ╠══ |Array S1 (5, 4, 3) float64|
        ╠══ |Array S2 (20) float64|
        ╚══ |Array S3 (8, 9) int32|
    [<pydot.core.Dot object at 0x000001C03CC1AB10>]
    renders/algo_mem_1.html
    





<iframe
    width="100%"
    height="1000px"
    src="renders/algo_mem_1.html"
    frameborder="0"
    allowfullscreen

></iframe>





```python

```
