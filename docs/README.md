# Documentation

The data a training reads are described in [`data.md`](data.md). The analysis then runs in five stages; each page describes one stage with the same sections: what it does and reads, how to run it, its outputs, its metrics (each stored key, what it measures and its value when everything is in order) and its plots. Configuration keys are listed next to the example configs in `examples/<stage>/README.md`.

1. **Training** ([`training.md`](training.md)): one network per training point, nuisance direction or background process (`python -m topsbi.train`), with its closure metrics and plots.
2. **Ensemble** ([`ensemble.md`](ensemble.md)): the fit of the per-event quadratic coefficients over the networks of each seed, averaged over seeds; its cache under `{name}/ensemble/`, which validation and the scanner read; and the background caches for a mixture likelihood.
3. **Validation** ([`validation.md`](validation.md)): `python -m topsbi.validation`: bias test per component, attribution, suspect networks and the point search.
4. **Nuisance validation** ([`validation.md`](validation.md)): `python -m topsbi.nuisance_validation`, the last check before likelihood scans: closure of every nuisance network and of its interpolation in ν, factorisation at other WC points, and the learned model with the learned EFT ratio.
5. **Likelihood scans** (`likelihood_scans.md`, not yet written): the scanner, once it is a script.
