==================================
API Referansı: Yerleşim Fonksiyonları
==================================

Bu bölüm, Keçeci Yerleşimi algoritmasını kullanarak düğüm konumlarını hesaplayan temel fonksiyonları belgeler. Bu fonksiyonlar, farklı graf kütüphaneleri arasında maksimum uyumluluk sağlayacak şekilde tasarlanmıştır.

.. automodule:: kececilayout
   :members: kececi_layout_v4, kececi_layout_v4_nx, kececi_layout_v4_ig, kececi_layout_v4_nk, kececi_layout_v4_gg, kececi_layout_v4_rx, kececi_layout_v4_pure, to_networkx
   :undoc-members:
   :show-inheritance:

------------------------
``kececi_layout_v4``
------------------------

.. autofunction:: kececilayout.kececi_layout_v4

Bu fonksiyon, paketin **birincil ve birleşik yerleşim fonksiyonudur**. Giriş grafiğinin türünü otomatik olarak algılar (örneğin `NetworkX`, `igraph`, `rustworkx`, `networkit`, `graphillion`) ve buna uygun Keçeci Yerleşimi koordinatlarını hesaplar.

Kullanıcıların, desteklenen herhangi bir graf kütüphanesiyle çalışabilen tek ve basit bir fonksiyona ihtiyaç duyduğu durumlar için ana arayüz görevi görür.

**Temel Özellikler:**

- **Çoklu Kütüphane Uyumu:** Grafin hangi kütüphaneden geldiğini düşünmenize gerek yok.
- **Deterministik Çıktı:** Aynı graf ve parametreler her zaman aynı yerleşimi üretir.
- **Genişleyen Zıgzag:** Varsayılan olarak (`expanding=True`), ikincil ofset mesafe arttıkça büyür ve üçgen, yayılan bir desen oluşturur.

**Örnek:**

.. code-block:: python

   import kececilayout as kl
   import networkx as nx

   G = nx.path_graph(10)
   pos = kl.kececi_layout_v4(G, primary_direction='top-down', expanding=True)
   # Döner: {0: (0.0, 0.0), 1: (-0.5, -1.0), 2: (0.5, -2.0), ...}

------------------------
``kececi_layout_v4_pure``
------------------------

.. autofunction:: kececilayout.kececi_layout_v4_pure

Keçeci Yerleşimi algoritmasının hafif, bağımlılık içermeyen bir sürümüdür. Bu fonksiyon sadece standart Python kütüphanesini ve `math` modülünü kullanır.

`networkx` veya `numpy` gibi ek paketlerin kurulmasının mümkün veya istenmediği ortamlar için idealdir.

**Kullanım Alanları:**

- Gömülü sistemler
- Minimalist dağıtımlar
- Algoritmanın temel mantığını incelemek için eğitim amaçlı

Girdi, düğüm kimliklerinin basit bir yinelenebiliridir (örneğin, tamsayılar veya dizelerin listesi).

**Örnek:**

.. code-block:: python

   nodes = ['A', 'B', 'C', 'D']
   pos = kl.kececi_layout_v4_pure(nodes, primary_spacing=2.0, secondary_start='left')
   # Döner: {'A': (0.0, 0.0), 'B': (-2.0, 2.0), 'C': (2.0, 4.0), ...}

------------------------
``to_networkx``
------------------------

.. autofunction:: kececilayout.to_networkx

Desteklenen bir kütüphaneden (`igraph`, `rustworkx`, `networkit`, `graphillion`) gelen bir graf nesnesini standart bir `NetworkX` grafiğine dönüştüren bir yardımcı fonksiyondur.

`draw_kececi` ve `kececi_layout_v4` fonksiyonlarının iç işleyişi için çok önemlidir çünkü genellikle `NetworkX`'i ortak bir ara format olarak kullanırlar.

**Döner:** Aynı düğüm ve kenarlara sahip yeni bir `networkx.Graph` nesnesi.

**Örnek:**

.. code-block:: python

   import igraph as ig
   ig_graph = ig.Graph.Ring(5)
   nx_graph = kl.to_networkx(ig_graph)
   print(type(nx_graph))  # <class 'networkx.classes.graph.Graph'>
