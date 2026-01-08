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

from . import _util as _  # load it first
from ._nodes import ArrayNode, BaseNode, ContainerNode, ItemNodeT, ValueNode
from ._procedures import (
    allocate_bmap,
    ar_spec,
    array_arspec,
    arrays_map,
    bmap_get,
    bmap_pyvis,
    build_bmap,
    build_buffer_allocator,
    build_flatmap,
    clone_bmap,
    db_node,
    f_arspec,
    f_arspec_i,
    ft_arspec,
    oneshot_args,
    save_bmap_tree,
    sb_node,
    v_spec,
)
from ._util import (
    BButil,
    BufferAlign,
    BufferMap,  # rename to ContainerType
    NativeTypes,
    aligned_buffer,
    bdict,
    buffer_expr,
    buffer_symbols,
    c_orlen,
    cse_codereduction,
    dt_buff_exprs,
    eval_buff_expr,
    eval_buff_exprs,
    is_eqn,
    mk_buff_dict,
    numb_syms,
)

# update build_flatmap, oneshot_args, flat_inits,

__all__ = [
    "ArrayNode",
    "BaseNode",
    "BButil",
    "BufferAlign",
    "BufferMap",
    "_",
    "ContainerNode",
    "ItemNodeT",
    "NativeTypes",
    "ValueNode",
    "aligned_buffer",
    "allocate_bmap",
    "ar_spec",
    "array_arspec",
    "arrays_map",
    "bdict",
    "bmap_get",
    "bmap_pyvis",
    "build_bmap",
    "build_buffer_allocator",
    "build_flatmap",
    "buffer_expr",
    "buffer_symbols",
    "c_orlen",
    "clone_bmap",
    "cse_codereduction",
    "db_node",
    "dt_buff_exprs",
    "eval_buff_expr",
    "eval_buff_exprs",
    "f_arspec",
    "f_arspec_i",
    "ft_arspec",
    "is_eqn",
    "mk_buff_dict",
    "numb_syms",
    "oneshot_args",
    "save_bmap_tree",
    "sb_node",
    "v_spec",
]
