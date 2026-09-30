Configuration Guide
===================

CANDEL experiments are defined using TOML configuration files. These files
specify the data paths, model parameters, priors, and inference settings.

Basic structure
---------------

A typical configuration file is organized into several sections:

.. code-block:: toml

   base = ["config.toml"]

   [inference]
   model = "TFRModel"
   num_warmup = 500
   num_samples = 1000
   num_chains = 4

   [io]
   catalogue_name = "CF4_W1"
   fname_output = "results/example/samples.hdf5"

   [pv_model]
   kind = "Vext"
   r_limits_malmquist = [0.01, 200.0]
   dr_malmquist = 0.5
   which_distance_prior = "empirical"

   [model]
   which_selection = "none"

Base paths are resolved relative to the TOML file, falling back to the shared
fragments in ``candel/configs`` (``config_paths.toml``, ``config_priors.toml``).
This compact form therefore assumes the run config lives in
``configs/`` of the ``candel-pv`` repository, next to its ``config.toml``.

The untracked ``local_config.toml`` at the repository root holds
machine-local settings (roots, field paths, reconstruction options). Values
are merged with increasing precedence: ``base`` fragments, then
``local_config.toml``, then the config file itself.

Path handling
-------------

- ``root_main``: The repository root (where the code lives). Required.
- ``root_data``: Base directory for input data files. Optional; defaults to
  ``<root_main>/data``.
- ``root_results``: Base directory for outputs (samples, plots, logs).
  Optional; defaults to ``<root_main>/results``.
- Relative paths in the configuration are automatically resolved against the
  appropriate root: data file keys against ``root_data``, output keys
  (``fname_output``) against ``root_results``.

Priors
------

Priors are specified in the ``[model.priors]`` section. Supported distributions
include:

- ``uniform``: Requires ``low`` and ``high``.
- ``normal``: Requires ``mean`` and ``std``.
- ``half_normal``: Requires ``std``.
- ``delta``: Fixed value, requires ``value``.
- ``log_normal``: Requires ``mean`` and ``std`` of the underlying normal.

Model settings
--------------

The ``[model]`` section contains settings specific to the distance indicator
and the physical model:

- ``name``: The name of the PV model class (e.g., ``"TFR"``, ``"SN"``, ``"PantheonPlus"``, ``"FP"``).
- ``which_run``: For non-PV runs, the probe to run (``"CH0"``,
  ``"EDD_TRGB"`` or ``"MWCepheids"``; any installed probe package may add
  more). Leave it unset for peculiar-velocity runs.
- ``Om``: Matter density parameter :math:`\Omega_m`.
- ``use_reconstruction``: Boolean, whether to use a reconstructed density/velocity field.
- ``which_selection``: Type of selection function to apply (e.g., ``"TRGB_magnitude"``, ``"redshift"``, or ``"none"``).

Model and Catalogue Names
-------------------------

When running PV models, the ``[inference]`` and ``[io]`` sections must specify
the model and catalogue:

.. code-block:: toml

   [inference]
   model = "TFRModel"

   [io]
   catalogue_name = "CF4_W1"

For joint inferences, these can be lists:

.. code-block:: toml

   [inference]
   model = ["TFRModel", "SNModel"]

   [io]
   catalogue_name = ["CF4_W1", "Foundation"]

Batch Generation
----------------

For large experiments (e.g., parameter sweeps), CANDEL includes a template-based
configuration generator:

.. code-block:: bash

   python scripts/runs/generate_tasks.py list
   python scripts/runs/generate_tasks.py build test

This script reads a template TOML and applies a named grid of overrides defined
in the ``specs.py`` module of each installed probe package. It generates:

1. A directory of ``.toml`` files, one for each combination of parameters.
2. A ``tasks_[index].txt`` file containing the paths to all generated configs.

Submit the task list with ``scripts/runs/submit.sh``. ``--batch N`` places N
tasks in one scheduler job and ``--parallel P`` runs P of them concurrently on a
CPU job, splitting its cores between them.

Inference settings
------------------

The ``[inference]`` section controls the NUTS sampler:

- ``num_warmup``: Number of adaptation steps.
- ``num_samples``: Number of posterior samples to keep per chain.
- ``num_chains``: Number of independent Markov chains.
- ``seed``: Random seed for reproducibility.
