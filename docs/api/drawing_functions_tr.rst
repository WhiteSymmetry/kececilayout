==================================
API Referansı: Çizim Fonksiyonları
==================================

Bu bölüm, Keçeci Yerleşimi'ni kullanarak grafikleri gelişmiş stillerde görselleştirmek için ana kullanıcı fonksiyonunu belgeler.

.. automodule:: kececilayout
   :members: draw_kececi
   :undoc-members:
   :show-inheritance:

------------------------
``draw_kececi``
------------------------

.. autofunction:: kececilayout.draw_kececi

Bu, Keçeci Yerleşimi'ni kullanarak grafikleri çeşitli gelişmiş stillerde oluşturmak için ana fonksiyondur. `to_networkx()` fonksiyonunu otomatik olarak kullanarak, `NetworkX`, `igraph`, `rustworkx`, `networkit`, `graphillion` gibi birden fazla kütüphaneden gelen grafiklerle uyumlu çalışır.

Fonksiyon, `style` parametresine göre farklı çizim stilleri uygulayarak temiz ve anlamlı görselleştirmeler oluşturur.

**Desteklenen Stiller:**

- ``'curved'``: Kenarları düzgün eğriler halinde çizer (`arc3` bağlantı stili). Yoğun grafiklerde görsel karışıklığı azaltmak için idealdir.
- ``'transparent'``: Kenarların saydamlığını uzunluğuna göre ayarlar (daha kısa kenarlar daha opaktır). Yerel yapıyı vurgulamaya yardımcı olur.
- ``'3d'``: Düğümleri Z ekseni boyunca 3D bir heliks (sarmal) desende yerleştirir. 3D projeksiyonlu bir eksen gerektirir.

**Temel Özellikler:**

- **Çoklu Kütüphane Uyumu:** Desteklenen herhangi bir kütüphanenin grafı ile çalışır.
- **Esnek Stil Seçenekleri:** `node_size`, `node_color`, `font_color` gibi tüm standart `matplotlib` ve `networkx.draw()` parametreleri `**kwargs` ile iletilebilir.
- **Eksen Kontrolü:** Daha büyük grafiklere yerleştirmek için kendi `matplotlib` ekseninizi (`ax`) sağlayabilirsiniz.
- **3D Görselleştirme:** `'3d'` stili, ardışık ilerlemeyi göstermek için dinamik bir sarmal görsel oluşturur.

**Örnek:**

.. code-block:: python

   import kececilayout as kl
   import networkx as nx

   G = nx.path_graph(10)
   kl.draw_kececi(G, style='curved', node_color='lightblue', node_size=800)
   plt.show()

Daha fazla örnek için `quickstart_tr` kılavuzuna bakın.
