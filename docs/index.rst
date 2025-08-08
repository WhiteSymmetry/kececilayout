.. kececilayout documentation master file, created by
   sphinx-quickstart on Mon Apr  5 12:00:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================================
KeçeciLayout: Deterministic Zigzag Graph Layout
=================================================

**KeçeciLayout**, doğrusal, sıralı veya ardışık yapıdaki graf sistemlerini görselleştirmek için tasarlanmış, deterministik bir graf yerleşim algoritmasıdır. Düğümleri birincil eksende sıralar ve ikincil eksende artan bir "zıgzag" (zigzag) desenle düzenler. Bu yaklaşım, özellikle yollar, zincirler, kimyasal yapılar, çevresel ağlar ve sıralı süreçler gibi yapıların net bir şekilde görselleştirilmesini sağlar.

Bu paket, farklı graf kütüphaneleriyle (`NetworkX`, `igraph`, `rustworkx`, `networkit`, `graphillion`) uyumlu olacak şekilde tasarlanmıştır ve kullanıcıya yüksek esneklik sunar.

.. toctree::
   :maxdepth: 2
   :caption: Kullanıcı Rehberi

   quickstart
   installation
   usage
   drawing_styles

.. toctree::
   :maxdepth: 2
   :caption: API Referansı

   api/layout_functions
   api/drawing_functions

.. toctree::
   :maxdepth: 1
   :caption: Ek Bilgiler

   license
   citation

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🚀 Hızlı Başlangıç
      :class-title: sd-font-weight-bold
      :link: quickstart.html
      :link-type: doc

      KeçeciLayout'u 5 dakikada nasıl kullanacağınızı öğrenin.

   .. grid-item-card:: 🧰 Kurulum
      :class-title: sd-font-weight-bold
      :link: installation.html
      :link-type: doc

      Paketi kurmak için tüm yöntemler.

   .. grid-item-card:: 🎨 Görselleştirme
      :class-title: sd-font-weight-bold
      :link: drawing_styles.html
      :link-type: doc

      Eğri, şeffaf ve 3D stili ile gelişmiş çizimler.

   .. grid-item-card:: 📚 API
      :class-title: sd-font-weight-bold
      :link: api/layout_functions.html
      :link-type: doc

      Tüm fonksiyonların teknik dokümantasyonu.

.. note::
   Bu proje açık kaynaktır ve MIT lisansı altında dağıtılmaktadır. Katkıda bulunmak isterseniz, `GitHub deposuna <https://github.com/WhiteSymmetry/kececilayout>`_ göz atın.
