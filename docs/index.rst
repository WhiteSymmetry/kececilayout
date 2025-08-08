=================================================
KeçeciLayout: Deterministic Zigzag Graph Layout
=================================================
KeçeciLayout is a deterministic graph layout algorithm designed to visualize linear, sequential, or consecutive graph structures. It arranges nodes along a primary axis and orders them in an increasing "zigzag" pattern along a secondary axis. This approach enables clear visualization of structures such as paths, chains, chemical compounds, environmental networks, and sequential processes.

KeçeciLayout, doğrusal, sıralı veya ardışık yapıdaki graf sistemlerini görselleştirmek için tasarlanmış, deterministik bir graf yerleşim algoritmasıdır. Düğümleri birincil eksende sıralar ve ikincil eksende artan bir "zigzag" desenle düzenler. Bu yaklaşım, özellikle yollar, zincirler, kimyasal yapılar, çevresel ağlar ve sıralı süreçler gibi yapıların net bir şekilde görselleştirilmesini sağlar.

This package is designed to work seamlessly with various graph libraries (`NetworkX`, `igraph`, `rustworkx`, `networkit`, `graphillion`) and offers high flexibility for users.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart_en
   quickstart_tr
   installation_en
   installation_tr
   usage_en
   usage_tr
   drawing_styles_en
   drawing_styles_tr

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/layout_functions_en
   api/layout_functions_tr
   api/drawing_functions_en
   api/drawing_functions_tr

.. toctree::
   :maxdepth: 1
   :caption: Additional Info

   license_en
   license_tr
   citation_en
   citation_tr

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🚀 Quick Start
      :class-title: sd-font-weight-bold
      :link: quickstart_en.html
      :link-type: doc

      Learn how to use KeçeciLayout in 5 minutes.

   .. grid-item-card:: 🚀 Hızlı Başlangıç
      :class-title: sd-font-weight-bold
      :link: quickstart_tr.html
      :link-type: doc

      KeçeciLayout'u 5 dakikada nasıl kullanacağınızı öğrenin.

   .. grid-item-card:: 🧰 Installation
      :class-title: sd-font-weight-bold
      :link: installation_en.html
      :link-type: doc

      All methods to install the package.

   .. grid-item-card:: 🧰 Kurulum
      :class-title: sd-font-weight-bold
      :link: installation_tr.html
      :link-type: doc

      Paketi kurmak için tüm yöntemler.

   .. grid-item-card:: 🎨 Drawing Styles
      :class-title: sd-font-weight-bold
      :link: drawing_styles_en.html
      :link-type: doc

      Curved, transparent, and 3D styles.

   .. grid-item-card:: 🎨 Çizim Stilleri
      :class-title: sd-font-weight-bold
      :link: drawing_styles_tr.html
      :link-type: doc

      Eğri, şeffaf ve 3D stili ile gelişmiş çizimler.

.. note::
   This project is open-source and distributed under the MIT license. For contributions, visit the `GitHub repository <https://github.com/WhiteSymmetry/kececilayout>`_.
