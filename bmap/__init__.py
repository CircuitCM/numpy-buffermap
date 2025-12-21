

"""Numpy Buffer Map
===================

A compact toolkit for laying out multiple NumPy arrays inside one or more
aligned backing buffers using a tree of nodes. The tree encodes *sharing
semantics* (shared vs. distinct regions), alignment policy, offsets, and
friendly labels. The pipeline is:

1) **Reduction**: merge adjacent containers that share the same rule.
2) **Size propagation & offset assignment**: compute `nbytes` per node and
   assign byte offsets (respecting alignment and container flags).
3) **Allocation**: allocate a single aligned top buffer and map `ArrayNode`s
   to views into it.

Core node types:
- **ContainerNode**: internal node that carries a rule: `BufferMap.SHARED`
  (children occupy the same region sized to the max child) or
  `BufferMap.DISTINCT` (children are concatenated, optionally without
  per-child alignment when `align=True`).
- **ArrayNode**: leaf that becomes a concrete NumPy array; supports optional
  *aligned leading dimension* (`align_ldim`) producing a padded backing shape
  (`bshape`) while exposing a trimmed logical `shape`.
- **ValueNode**: leaf that participates in traversal/flattening but does not
  consume buffer bytes.

Export helpers:
- **`bmap_todot`**: emit a Graphviz DOT representation using `anytree`.
- **`bmap_pyvis`** (defined later): visualize via PyVis/NetworkX/pydot.

Quickstart
----------
Build a tree with containers and arrays, then run `build_bmap` → `allocate_bmap`,
or call `ContainerNode.build_flatmap()` for the one-shot variant. Flattening
helpers (`flatten_items`, `flat_inits`) return leaves or initialized values in
preorder.
"""



from . import _util as _ #load it up
from ._util import BufferMap #rename to ContainerType
from ._util import BufferAlign, NativeTypes, aligned_buffer, c_orlen,buffer_expr,mk_buff_dict,bdict,buffer_symbols,dt_buff_exprs, eval_buff_exprs, eval_buff_expr, is_eqn, BButil, cse_codereduction
from ._nodes import BaseNode,ValueNode,ArrayNode,ContainerNode,ItemNodeT
from ._procedures import build_buffer_allocator, build_bmap, arrays_map, bmap_get,build_flatmap, oneshot_args, v_spec, ar_spec, f_arspec_i, f_arspec, array_arspec, ft_arspec, sb_node, db_node, bmap_pyvis, save_bmap_tree, clone_bmap 
#update build_flatmap, oneshot_args, flat_inits, 
