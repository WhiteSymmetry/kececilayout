======================
Çizim Stilleri
======================

KeçeciLayout, gelişmiş görselleştirme stillerini destekler.

------------------------
Eğri Kenarlar
------------------------

Kenarlar yay şeklinde çizilir.

.. code-block:: python

   kl.draw_kececi(G, style='curved', node_color='skyblue')

------------------------
Şeffaf Kenarlar
------------------------

Kenar uzunluğuna göre şeffaflık ayarlanır.

.. code-block:: python

   kl.draw_kececi(G, style='transparent', node_color='purple')

------------------------
3D Heliks
------------------------

Düğümler 3D'de spiral şeklinde yerleştirilir.

.. code-block:: python

   kl.draw_kececi(G, style='3d', ax=plt.figure().add_subplot(projection='3d'))
