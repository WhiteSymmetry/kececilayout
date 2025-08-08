============
Installation
============

KeçeciLayout requires Python 3.8 or higher.

------------------
Install via Pip
------------------

.. code-block:: bash

   pip install kececilayout

Upgrade to latest version:

.. code-block:: bash

   pip install --upgrade kececilayout

----------------------
Install via Conda
----------------------

.. code-block:: bash

   conda install -c bilgi kececilayout

-------------------------
Developer Installation
-------------------------

Clone and install in editable mode:

.. code-block:: bash

   git clone https://github.com/WhiteSymmetry/kececilayout.git
   cd kececilayout
   pip install -e .

-------------------------
Dependencies
-------------------------

- **Required**: `networkx`, `matplotlib`, `numpy`
- **Optional**: `igraph`, `networkit`, `rustworkx`, `graphillion`

Install all optional dependencies:

.. code-block:: bash

   pip install kececilayout[all]

.. tip::
   For basic use, only `networkx` and `matplotlib` are needed.
