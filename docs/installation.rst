Installation
============

.. code-block:: bash

   git clone https://github.com/candel-cosmo/CANDEL.git
   cd CANDEL

   python -m venv venv_candel
   source venv_candel/bin/activate
   python -m pip install --upgrade pip setuptools
   python -m pip install -e .
   for pkg in packages/*/; do python -m pip install -e "$pkg"; done

The core package runs without any probe package; install only the ones you
need (``candel-pv``, ``candel-ch0``, ``candel-trgb``, ``candel-mwcepheids``,
``candel-maser``). Each registers itself with the core through the
``candel.probes`` entry-point group.

For model-evidence computation, also install
`harmonic <https://github.com/astro-informatics/harmonic>`_.
