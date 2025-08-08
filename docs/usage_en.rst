===========
Usage Guide
===========

KeçeciLayout provides flexible parameters for advanced use.

------------------------
Parameters
------------------------

- ``primary_spacing``: Distance along the main axis.
- ``secondary_spacing``: Base zigzag offset.
- ``primary_direction``: `top-down`, `bottom-up`, `left-to-right`, `right-to-left`.
- ``secondary_start``: `right`, `left`, `up`, `down`.
- ``expanding``: If `True`, zigzag grows; if `False`, constant offset.

------------------------
Expanding vs. Parallel
------------------------

- ``expanding=True``: Creates a triangular, spreading pattern.
- ``expanding=False``: Creates parallel lines.

.. image:: https://github.com/WhiteSymmetry/kececilayout/blob/main/docs/_static/expanding_comparison.png?raw=true
   :width: 70%
   :alt: Expanding vs. Parallel

------------------------
Node Ordering
------------------------

Nodes are sorted numerically. For custom order, relabel nodes first:

.. code-block:: python

   G = nx.relabel_nodes(G, mapping_dict)
