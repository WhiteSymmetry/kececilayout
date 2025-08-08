==================================
API Reference: Drawing Functions
==================================

This section documents the main user-facing function for visualizing graphs using the Keçeci Layout with advanced styles.

.. automodule:: kececilayout
   :members: draw_kececi
   :undoc-members:
   :show-inheritance:

------------------------
``draw_kececi``
------------------------

.. autofunction:: kececilayout.draw_kececi

This is the primary function for rendering graphs with the Keçeci Layout in various advanced styles. It acts as a unified interface that supports graphs from multiple libraries (`NetworkX`, `igraph`, `rustworkx`, `networkit`, `graphillion`) by automatically converting them to a `NetworkX` graph using `to_networkx()`.

The function creates a clean and expressive visualization by applying different drawing styles based on the `style` parameter.

**Supported Styles:**

- ``'curved'``: Draws edges as smooth curves using `arc3` connection style. Ideal for reducing visual clutter in dense graphs.
- ``'transparent'``: Adjusts edge opacity based on length (shorter edges are more opaque). This helps highlight local structure.
- ``'3d'``: Places nodes in a 3D helix pattern along the Z-axis. Requires a 3D projection axis.

**Key Features:**

- **Cross-library Compatibility:** Works with graphs from any supported library.
- **Flexible Styling:** All standard `matplotlib` and `networkx.draw()` parameters (e.g., `node_size`, `node_color`, `font_color`) can be passed via `**kwargs`.
- **Axis Control:** You can provide your own `matplotlib` axis (`ax`) for embedding in larger figures.
- **3D Support:** The `'3d'` style creates a dynamic spiral visualization, useful for showing sequential progression.

**Example:**

.. code-block:: python

   import kececilayout as kl
   import networkx as nx

   G = nx.path_graph(10)
   kl.draw_kececi(G, style='curved', node_color='lightblue', node_size=800)
   plt.show()

For more examples, see the `quickstart_en` guide.
