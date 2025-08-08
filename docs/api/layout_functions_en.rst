==================================
API Reference: Layout Functions
==================================

This section documents the core functions responsible for calculating node positions using the Keçeci Layout algorithm. These functions are designed for maximum compatibility across different graph libraries.

.. automodule:: kececilayout
   :members: kececi_layout_v4, kececi_layout_v4_nx, kececi_layout_v4_ig, kececi_layout_v4_nk, kececi_layout_v4_gg, kececi_layout_v4_rx, kececi_layout_v4_pure, to_networkx
   :undoc-members:
   :show-inheritance:

------------------------
``kececi_layout_v4``
------------------------

.. autofunction:: kececilayout.kececi_layout_v4

This is the **primary and unified layout function** of the package. It automatically detects the type of the input graph (e.g., `NetworkX`, `igraph`, `rustworkx`, `networkit`, `graphillion`) and calculates its Keçeci Layout coordinates accordingly.

It serves as the main interface for users who want a single, simple function that works with any supported graph library.

**Key Features:**

- **Cross-Library Compatibility:** No need to worry about which library your graph comes from.
- **Deterministic Output:** The same graph and parameters always produce the same layout.
- **Expanding Zigzag:** By default (`expanding=True`), the secondary offset grows with distance, creating a triangular, spreading pattern.

**Example:**

.. code-block:: python

   import kececilayout as kl
   import networkx as nx

   G = nx.path_graph(10)
   pos = kl.kececi_layout_v4(G, primary_direction='top-down', expanding=True)
   # Returns: {0: (0.0, 0.0), 1: (-0.5, -1.0), 2: (0.5, -2.0), ...}

------------------------
``kececi_layout_v4_pure``
------------------------

.. autofunction:: kececilayout.kececi_layout_v4_pure

A lightweight, dependency-free version of the Keçeci Layout algorithm. This function uses only the standard Python library and the `math` module.

It is ideal for environments where installing additional packages like `networkx` or `numpy` is not possible or desired.

**Use Cases:**

- Embedded systems
- Minimalist deployments
- Educational purposes to study the algorithm's core logic

The input is a simple iterable of node identifiers (e.g., a list of integers or strings).

**Example:**

.. code-block:: python

   nodes = ['A', 'B', 'C', 'D']
   pos = kl.kececi_layout_v4_pure(nodes, primary_spacing=2.0, secondary_start='left')
   # Returns: {'A': (0.0, 0.0), 'B': (-2.0, 2.0), 'C': (2.0, 4.0), ...}

------------------------
``to_networkx``
------------------------

.. autofunction:: kececilayout.to_networkx

A utility function that converts a graph object from a supported library (`igraph`, `rustworkx`, `networkit`, `graphillion`) into a standard `NetworkX` graph.

This function is crucial for the internal operation of `draw_kececi` and `kececi_layout_v4`, as they often use `NetworkX` as a common intermediate format for processing.

**Returns:** A new `networkx.Graph` object with the same nodes and edges.

**Example:**

.. code-block:: python

   import igraph as ig
   ig_graph = ig.Graph.Ring(5)
   nx_graph = kl.to_networkx(ig_graph)
   print(type(nx_graph))  # <class 'networkx.classes.graph.Graph'>
