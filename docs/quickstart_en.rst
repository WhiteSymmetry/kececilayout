============
Quick Start
============

Learn how to use KeçeciLayout in under 5 minutes.

------------------------
1. Import Required Libraries
------------------------

.. code-block:: python

   import networkx as nx
   import matplotlib.pyplot as plt
   import kececilayout as kl

------------------------
2. Create a Graph
------------------------

Let's create a simple path graph:

.. code-block:: python

   G = nx.path_graph(8)  # 8-node linear graph

------------------------
3. Apply Keçeci Layout
------------------------

.. code-block:: python

   pos = kl.kececi_layout(
       G,
       primary_spacing=1.5,
       secondary_spacing=0.8,
       primary_direction='top_down',
       secondary_start='right',
       expanding=True
   )

- ``primary_spacing``: Distance between nodes along the primary axis.
- ``secondary_spacing``: Base offset for the zigzag.
- ``primary_direction``: Main direction (`top_down`, `bottom_up`, etc.).
- ``secondary_start``: Zigzag starts to `right` or `left`.
- ``expanding=True``: Zigzag amplitude increases with distance.

------------------------
4. Visualize the Graph
------------------------

.. code-block:: python

   plt.figure(figsize=(6, 10))
   nx.draw(
       G,
       pos=pos,
       with_labels=True,
       node_color='lightcoral',
       node_size=600,
       font_size=12,
       edge_color='gray',
       connectionstyle='arc3,rad=0.1'
   )
   plt.title("8-Node Path Graph with KeçeciLayout")
   plt.axis('equal')
   plt.show()

.. image:: https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nx-1.png?raw=true
   :alt: KeçeciLayout Example
   :align: center
   :width: 60%

------------------------
5. Use with Other Libraries
------------------------

KeçeciLayout supports multiple graph backends:

.. tabs::

   .. tab:: iGraph

      .. code-block:: python

         import igraph as ig
         G_ig = ig.Graph.Ring(8, circular=False)
         pos_ig = kl.kececi_layout(G_ig, primary_direction='left-to-right')
         layout = ig.Layout(pos_ig)
         ig.plot(G_ig, layout=layout, vertex_label=range(8))

   .. tab:: Rustworkx

      .. code-block:: python

         import rustworkx as rx
         G_rx = rx.generators.path_graph(8)
         pos_rx = kl.kececi_layout(G_rx, primary_direction='bottom_up')

   .. tab:: Graphillion

      .. code-block:: python

         import graphillion as gg
         universe = [(i, i+1) for i in range(1, 8)]
         gg.GraphSet.set_universe(universe)
         gs = gg.GraphSet()
         pos_gg = kl.kececi_layout(gs, secondary_start='left')

.. tip::
   For more examples, check the `examples/` folder or try live with `Binder <https://terrarium.evidencepub.io/v2/gh/WhiteSymmetry/kececilayout/HEAD>`_.
