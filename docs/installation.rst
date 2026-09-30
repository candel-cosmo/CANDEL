Installation
============

.. code-block:: bash

   git clone https://github.com/candel-cosmo/CANDEL.git
   # Probe packages, each in its own repository, cloned beside the core:
   for pkg in candel-pv candel-ch0 candel-trgb candel-mwcepheids candel-maser; do
       git clone "https://github.com/candel-cosmo/$pkg.git"
   done
   cd CANDEL

   python -m venv venv_candel
   source venv_candel/bin/activate
   python -m pip install --upgrade pip setuptools
   python -m pip install -e .
   for pkg in ../candel-*/; do python -m pip install --no-deps -e "$pkg"; done

The core package runs without any probe package; install only the ones you
need (``candel-pv``, ``candel-ch0``, ``candel-trgb``, ``candel-mwcepheids``,
``candel-maser``). Each registers itself with the core through the
``candel.probes`` entry-point group. The probe repositories read ``data/``,
``results/`` and ``local_config.toml`` from the core checkout (Python through
``candel.util.CANDEL_ROOT``, shell scripts through ``$CANDEL_ROOT``, which
defaults to ``../CANDEL``).

For model-evidence computation, also install
`harmonic <https://github.com/astro-informatics/harmonic>`_.
