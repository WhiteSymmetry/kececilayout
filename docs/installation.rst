============
Kurulum
============

KeçeciLayout, Python 3.8 ve üzeri sürümlerde çalışır. Aşağıdaki yöntemlerden biriyle kolayca kurulabilir.

------------------
Pip ile Kurulum
------------------

KeçeciLayout, PyPI üzerinden doğrudan kurulabilir:

.. code-block:: bash

   pip install kececilayout

Güncel sürümü kontrol etmek için:

.. code-block:: bash

   pip install --upgrade kececilayout

----------------------
Conda ile Kurulum
----------------------

KeçeciLayout, `Anaconda` kanalından da kurulabilir:

.. code-block:: bash

   conda install -c bilgi kececilayout

veya:

.. code-block:: bash

   mamba install -c bilgi kececilayout

-------------------------
Geliştirici Kurulumu
-------------------------

GitHub üzerinden geliştirici sürümünü kurmak için:

.. code-block:: bash

   git clone https://github.com/WhiteSymmetry/kececilayout.git
   cd kececilayout
   pip install -e .

Bu, projeyi geliştirme modunda kurar ve değişiklikleriniz otomatik olarak yansır.

-------------------------
Gereksinimler
-------------------------

KeçeciLayout, aşağıdaki kütüphanelere bağımlıdır:

- ``networkx`` (zorunlu)
- ``matplotlib`` (zorunlu)
- ``numpy`` (zorunlu)

Opsiyonel kütüphaneler (ilgili fonksiyonlar için gerekli):

- ``igraph``
- ``rustworkx``
- ``networkit``
- ``graphillion``

Tüm bağımlılıkları birlikte kurmak için:

.. code-block:: bash

   pip install kececilayout[all]

veya:

.. code-block:: bash

   conda install -c bilgi kececilayout

.. tip::
   Eğer sadece temel işlevleri kullanacaksanız, ``networkx`` ve ``matplotlib`` yeterlidir.
