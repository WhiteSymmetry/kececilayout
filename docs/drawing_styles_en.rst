======================
Drawing Styles
======================

KeçeciLayout supports advanced visualization styles.

------------------------
Curved Edges
------------------------

Edges are drawn as arcs.

.. code-block:: python

   kl.draw_kececi(G, style='curved', node_color='skyblue')

------------------------
Transparent Edges
------------------------

Edge opacity depends on length.

.. code-block:: python

   kl.draw_kececi(G, style='transparent', node_color='purple')

------------------------
3D Helix
------------------------

Nodes are placed in a 3D spiral.

.. code-block:: python

   kl.draw_kececi(G, style='3d', ax=plt.figure().add_subplot(projection='3d'))
