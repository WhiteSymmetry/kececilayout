============
Kurulum
============

KeçeciLayout, Python 3.8 ve üzeri sürümlerde çalışır.

------------------
Pip ile Kurulum
------------------

.. code-block:: bash

   pip install kececilayout

Güncel sürümü kurmak için:

.. code-block:: bash

   pip install --upgrade kececilayout

----------------------
Conda ile Kurulum
----------------------

.. code-block:: bash

   conda install -c bilgi kececilayout

-------------------------
Geliştirici Kurulumu
-------------------------

GitHub üzerinden geliştirici sürümünü kurun:

.. code-block:: bash

   git clone https://github.com/WhiteSymmetry/kececilayout.git
   cd kececilayout
   pip install -e .

-------------------------
Gereksinimler
-------------------------

- **Zorunlu**: `networkx`, `matplotlib`, `numpy`
- **Opsiyonel**: `igraph`, `networkit`, `rustworkx`, `graphillion`

Tüm bağımlılıkları kurmak için:

.. code-block:: bash

   pip install kececilayout[all]

.. tip::
   Temel kullanım için sadece `networkx` ve `matplotlib` yeterlidir.
