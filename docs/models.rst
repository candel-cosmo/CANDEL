Supported models
================

Forward-modelling approach
--------------------------

CANDEL implements a forward-modelling approach for each distance indicator.
Given a comoving distance :math:`r` and the cosmological parameter :math:`h = H_0/100`, the
distance modulus is computed as:

.. math::

   \mu(r, h) = 5 \log_{10}\left[\frac{(1+z_{\rm cosmo})r}{10\,\mathrm{pc}}\right]

where :math:`z_{\rm cosmo}` is the cosmological redshift. The observed
observable :math:`y` (e.g., apparent magnitude, line width) is then modelled
as:

.. math::

   y \sim \mathcal{N}(M + \mu(r, h), \sigma^2)

where :math:`M` is the intrinsic absolute magnitude (or a function of latent
observables) and :math:`\sigma` accounts for both measurement error and
intrinsic scatter.

Peculiar-velocity models
------------------------

These models work in units of :math:`h^{-1}\,\mathrm{Mpc}` and can be analysed
jointly via :class:`~candel_pv.base_pv.JointPVModel` with user-specified
shared parameters.

- **Tully--Fisher relation** (:class:`~candel_pv.model_PV_TFR.TFRModel`):
  2MTF, SFI++, CF4-TFR
- **Type Ia supernovae (SALT2)** (:class:`~candel_pv.model_PV_SN.SNModel`):
  LOSS, Foundation
- **Pantheon+** (:class:`~candel_pv.model_PV_PantheonPlus.PantheonPlusModel`):
  Pantheon+ with full covariance matrix
- **Fundamental Plane** (:class:`~candel_pv.model_PV_FP.FPModel`):
  6dFGS-FP, SDSS-FP

:math:`H_0` inference
---------------------

- **Cepheid-calibrated** :math:`H_0`
  (:class:`~candel_ch0.model.CH0Model`):
  35 Cepheid host galaxies from SH0ES
- **TRGB-calibrated** :math:`H_0`
  (:class:`~candel_trgb.model.TRGBModel`):
  Tip of the Red Giant Branch (TRGB) distances from EDD
- **Milky Way Cepheid calibration**
  (:class:`~candel_mwcepheids.model.MWCepheidModel`)
- **Megamaser disk** :math:`H_0`
  (:class:`~candel_maser.model_H0_maser.MaserDiskModel`):
  spot-level warped disk fits, run with ``python -m candel_maser.run_maser``

Package structure
-----------------

- :doc:`candel <api/candel>` -- probe registry, inference, evidence, utilities
- :doc:`candel.model <api/candel.model>` -- base models, priors, quadrature, LOS and bias utilities
- :doc:`candel.cosmo <api/candel.cosmo>` -- cosmography
- :doc:`candel.field <api/candel.field>` -- field loading, LOS interpolation, field caches and 3D volume grids
- :doc:`Probe packages <api/probes>` -- the PV, CH0, TRGB, MW Cepheid and megamaser packages
