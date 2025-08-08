======================
Gelişmiş Görselleştirme
======================

KeçeciLayout, `draw_kececi` fonksiyonu ile gelişmiş çizim stilleri sunar.

------------------------
1. Eğri Kenarlar (Curved)
------------------------

Kenarlar yay şeklinde çizilir.

.. code-block:: python

   kl.draw_kececi(G, style='curved', node_color='skyblue')

.. image:: https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nx-1.png?raw=true
   :width: 50%
   :alt: Eğri stil

------------------------
2. Şeffaf Kenarlar (Transparent)
------------------------

Kenar uzunluğuna göre şeffaflık ayarlanır (uzun kenarlar daha şeffaf).

.. code-block:: python

   kl.draw_kececi(G, style='transparent', node_color='purple')

.. image:: https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nk-1.png?raw=true
   :width: 50%
   :alt: Şeffaf stil

------------------------
3. 3D Heliks
------------------------

Düğümler 3D'de spiral (heliks) şeklinde yerleştirilir.

.. code-block:: python

   kl.draw_kececi(G, style='3d', ax=plt.figure().add_subplot(projection='3d'))

.. image:: https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/3d-helix.png?raw=true
   :width: 50%
   :alt: 3D stil

------------------------
Stil Parametreleri
------------------------

Tüm stiller, `matplotlib` parametrelerini kabul eder:

- ``node_size``
- ``node_color``
- ``font_color``
- ``edge_color``
- ``alpha``, vs.

Örnek:

.. code-block:: python

   kl.draw_kececi(
       G,
       style='curved',
       node_color='lightgreen',
       node_size=800,
       font_color='darkblue',
       edge_color='black'
   )
