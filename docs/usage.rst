==================
Detaylı Kullanım
==================

KeçeciLayout, hem basit hem de gelişmiş senaryolar için esnek parametreler sunar.

------------------------
Parametreler
------------------------

``kececi_layout_v4`` fonksiyonunun tüm parametreleri:

- ``graph``: Desteklenen graf kütüphanesinden bir graf nesnesi.
- ``primary_spacing``: Ana eksen boyunca düğümler arası mesafe (varsayılan: 1.0).
- ``secondary_spacing``: Zıgzag ofseti (varsayılan: 1.0).
- ``primary_direction``: Ana yön (`top-down`, `bottom-up`, `left-to-right`, `right-to-left`).
- ``secondary_start``: Zıgzag yönü (`right`, `left`, `up`, `down`).
- ``expanding``: Zıgzag büyüklüğü artar mı? (``True``/``False``)

------------------------
Yön Kombinasyonları
------------------------

+-----------------------+----------------------+--------------------------+
| Ana Yön               | İkincil Başlangıç    | Sonuç                    |
+=======================+======================+==========================+
| ``top-down``          | ``right``            | Sağ-sol zıgzag           |
+-----------------------+----------------------+--------------------------+
| ``left-to-right``     | ``up``               | Yukarı-aşağı zıgzag      |
+-----------------------+----------------------+--------------------------+
| ``bottom-up``         | ``left``             | Sol-sağ zıgzag           |
+-----------------------+----------------------+--------------------------+
| ``right-to-left``     | ``down``             | Aşağı-yukarı zıgzag      |
+-----------------------+----------------------+--------------------------+

------------------------
İleri Seviye: `expanding` Parametresi
------------------------

``expanding=True`` (varsayılan): Zıgzag miktarı ilerledikçe büyür.

.. math::

   \text{offset} = \text{start\_mult} \times \lceil i/2 \rceil \times \text{side} \times \text{secondary\_spacing}

``expanding=False``: Sabit ofset (paralel çizgiler).

.. image:: https://github.com/WhiteSymmetry/kececilayout/blob/main/docs/_static/expanding_comparison.png?raw=true
   :alt: Expanding True vs False
   :align: center
   :width: 70%

------------------------
Düğüm Sıralaması
------------------------

KeçeciLayout, düğümleri sıralı olarak işler. Eğer düğüm ID'leriniz sayısal değilse:

- ``sorted(list(nodes))`` kullanılır.
- Sıralanamazsa orijinal sırada kalır.

Özel bir sıralama istiyorsanız, önce grafınızı yeniden etiketleyin:

.. code-block:: python

   G = nx.relabel_nodes(G, mapping_dict)

------------------------
Boş Graf ve Hata Durumları
------------------------

- Boş graf: Boş pozisyon sözlüğü döner `{}`.
- Desteklenmeyen graf türü: ``TypeError`` fırlatır.
- Geçersiz yön: ``ValueError`` fırlatır.

.. warning::
   ``graphillion`` kullanırken, evrenin doğru tanımlanmış olduğundan emin olun.
