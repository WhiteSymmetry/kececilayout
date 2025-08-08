==================
Kullanım Kılavuzu
==================

KeçeciLayout, gelişmiş kullanım için esnek parametreler sunar.

------------------------
Parametreler
------------------------

- ``primary_spacing``: Ana eksen boyunca mesafe.
- ``secondary_spacing``: Zıgzag ofseti.
- ``primary_direction``: `top-down`, `bottom-up`, `left-to-right`, `right-to-left`.
- ``secondary_start``: `right`, `left`, `up`, `down`.
- ``expanding``: `True` ise zıgzag büyür; `False` ise sabit kalır.

------------------------
Genişleyen vs. Paralel
------------------------

- ``expanding=True``: Üçgen, yayılan bir desen oluşturur.
- ``expanding=False``: Paralel çizgiler oluşturur.

.. image:: https://github.com/WhiteSymmetry/kececilayout/blob/main/docs/_static/expanding_comparison.png?raw=true
   :width: 70%
   :alt: Genişleyen vs. Paralel

------------------------
Düğüm Sıralaması
------------------------

Düğümler sayısal olarak sıralanır. Özel sıralama için önce yeniden etiketleyin:

.. code-block:: python

   G = nx.relabel_nodes(G, mapping_dict)
