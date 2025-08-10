==================
Hızlı Başlangıç
==================

KeçeciLayout'u 5 dakikadan kısa sürede nasıl kullanacağınızı öğrenin.

------------------------
1. Gerekli Kütüphaneleri İçe Aktar
------------------------

.. code-block:: python

   import networkx as nx
   import matplotlib.pyplot as plt
   import kececilayout as kl

------------------------
2. Bir Graf Oluştur
------------------------

Basit bir yol (path) grafiği oluşturalım:

.. code-block:: python

   G = nx.path_graph(8)  # 8 düğümlü doğrusal graf

------------------------
3. Keçeci Layout Uygula
------------------------

.. code-block:: python

   pos = kl.kececi_layout(
       G,
       primary_spacing=1.5,
       secondary_spacing=0.8,
       primary_direction='top-down',
       secondary_start='right',
       expanding=True
   )

- ``primary_spacing``: Ana eksen boyunca düğümler arası mesafe.
- ``secondary_spacing``: Zıgzag ofsetinin temel birimi.
- ``primary_direction``: Ana yön (`top-down`, `bottom-up`, vs.).
- ``secondary_start``: Zıgzagın başlangıç yönü (`right`, `left`, vs.).
- ``expanding=True``: Zıgzag genliği ilerledikçe artar.

------------------------
4. Grafi Görselleştir
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
   plt.title("KeçeciLayout ile 8 Düğümlü Yol Grafiği")
   plt.axis('equal')
   plt.show()

.. image:: https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nx-1.png?raw=true
   :alt: KeçeciLayout Örneği
   :align: center
   :width: 60%

------------------------
5. Diğer Kütüphanelerle Kullanım
------------------------

KeçeciLayout, farklı graf kütüphaneleriyle uyumludur:

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
         pos_rx = kl.kececi_layout(G_rx, primary_direction='bottom-up')

   .. tab:: Graphillion

      .. code-block:: python

         import graphillion as gg
         universe = [(i, i+1) for i in range(1, 8)]
         gg.GraphSet.set_universe(universe)
         gs = gg.GraphSet()
         pos_gg = kl.kececi_layout(gs, secondary_start='left')

.. tip::
   Daha fazla örnek için `examples/` klasörüne göz atın veya `Binder <https://terrarium.evidencepub.io/v2/gh/WhiteSymmetry/kececilayout/HEAD>`_ ile deneyin.
