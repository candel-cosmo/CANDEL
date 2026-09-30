CANDEL
======

**CANDEL** is a framework for calibrating distance indicators in the local Universe.

CANDEL forward-models distance-indicator observables and redshift while
marginalising over latent variables such as distance and absolute magnitude.
Posterior sampling uses the No-U-Turn Sampler (NUTS) from
`NumPyro <https://github.com/pyro-ppl/numpyro>`_, with JAX providing automatic
differentiation and JIT compilation throughout.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   configuration
   models
   api/index

Tutorials
---------

For practical examples, see ``notebooks/example.ipynb`` in
`candel-pv <https://github.com/candel-cosmo/candel-pv>`_ and the ``papers/``
directory of each probe repository, which holds the scripts and notebooks
behind each publication, for example ``papers/CH0`` in
`candel-ch0 <https://github.com/candel-cosmo/candel-ch0>`_
for `Stiskalek et al. (2025) <https://arxiv.org/abs/2509.09665>`_.

HPC and Batch Processing
------------------------

CANDEL is designed for high-performance computing (HPC) environments. It
supports GPU acceleration via JAX and includes tools for generating large
batches of configuration files for parameter sweeps.

See the :doc:`configuration` guide for more information on the
``generate_tasks.py`` script and its template-based override system.
