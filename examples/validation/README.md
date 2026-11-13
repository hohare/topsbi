## Configuration file fields

Run with `python -m topsbi.validation -c config.yml [--refit] [--backgrounds]`; `--backgrounds` builds only the background caches ([`docs/ensemble.md`](../../docs/ensemble.md), section 7). What the ensemble fit does and its cache: [`docs/ensemble.md`](../../docs/ensemble.md); what validation tests, its outputs and plots: [`docs/validation.md`](../../docs/validation.md). `config.yml` is an example with shortened network lists.

- `background_fit`: optional, a group name in every member's `networks`: the background columns of the fit come from that group's networks and the SM network alone ([`docs/ensemble.md`](../../docs/ensemble.md), section 1)
- `background_networks`: `--backgrounds` only, `{process: {member: path of a background training yml}}`, the networks $\hat r(x; p)$ of every process, for the same members
- `backgrounds`: `--backgrounds` only, `{process: nominal directory of its combined background sample}`; the ensemble and the background ratios are evaluated on its `validation.p`
- `bias`: optional settings of the bias test
    - `features`: network inputs to test, names from the feature configuration (default: all)
    - `min_neff`: minimum effective number of events per bin, $n_{\rm eff} = (\sum w)^2 / \sum w^2$ with $w = w^s(\theta_0)$; neighbouring bins are merged until each reaches it (default 30)
    - `n_bins`: equal-count bins per input before merging; an input with at most this many distinct values gets one bin per value (default 20)
    - `z_threshold`: a component is flagged if its largest $Z$ over the inputs exceeds it (default 5.0)
- `c0`: the reference point $\theta_0$ every network was trained against: $n_{\rm wcs}+1$ values, 1 (SM) followed by one value per WC in `wcs` order. It must match the networks' `c0`
- `data`: path to the dataset the ensemble is evaluated and tested on, a pickled `TensorDataset` whose first two tensors are the features and the structure constants $s_q(z)$ ([`docs/data.md`](../../docs/data.md)). Use events that no network was trained on, e.g. the `validation.p` produced alongside `train.p`
- `device`: device the networks are evaluated on: `cpu` (default) or `cuda`. The `device` in the networks' training configs is ignored
- `eras`: optional era names, indexed like `lumi`, for the tick labels of the era column in the closure plots (without it the eras are labelled by their index); as in the training configs
- `features`: path to the feature configuration yaml ([`examples/features/README.md`](../features/README.md)), or its contents; gives the column (`loc`) and label of every network input
- `lumi`: integrated luminosity of each era ($n_{\rm eras}$ values), indexed by the era column (`year_loc`); as in the training configs
- `n_threads`: torch threads of the main process (default 1)
- `n_top_networks`: number of networks listed under `top_networks` of each flagged component in `ensemble_diagnostics.yml` (default 5)
- `name`: output directory: the ensemble cache (`ensemble/`), `ensemble_diagnostics.yml`, `proposed_points.yml` and `plots/`
- `point_search`: optional settings of the point search
    - `G`: weights of the objective $\mathrm{tr}(\mathrm{Cov}(\hat\gamma)\, G)$: `flagged` (default), the relative variance reduction of the flagged components; or `scan`, the uncertainty of $\hat r$ averaged over a scan region given by `bounds`, `mode` and `shape`
    - `bounds`: `G: scan` only, `{wc: a}` per WC varied in the scan: half-range (uniform) or width (gaussian); WCs not listed are fixed at 0
    - `max_yield_ratio`: largest inclusive yield relative to the SM allowed for a proposed point (default: the largest of the existing training points)
    - `min_gain`: the search stops when adding a unit lowers the objective by less than this fraction (default 0.01)
    - `mode`: `G: scan` only, `box` (all listed WCs varied together, default) or `1d` (one WC at a time, averaged)
    - `n_units`: maximum number of units (sets of new points) proposed (default 10)
    - `shape`: `G: scan` only, `uniform` (default) or `gaussian`
    - `unflagged_weight`: `G: flagged` only, weight $g_q$ of the components not flagged (flagged ones have 1; default 0.1)
- `seeds`: the ensemble members, `{member: {networks: [...], sm: ...}}`; per member the paths of the training configs (`training.yml`) of its networks (a list, or groups `{group: [...]}`, see `background_fit`) and of its SM network. Each network's `c1` gives its training point and its `name` the directory with `model.pt` and `complete/performance.yml`. Every member must be trained at the same set of points, with at least as many networks as kept terms of $T(\theta)$, and every network with the same `wcs` and `zero_terms`. Changing the list (or retraining a network) makes the cache stale: rerun with `--refit`
- `suspects`: optional settings of the suspects and drop tests
    - `min_gain`: a drop is accepted only if the largest $Z$ of a component it was blamed for falls by at least this (default 2.0)
    - `seed_fraction`: a suspect network that closes badly in at least this fraction of the members is a drop candidate (default 0.5)
    - `share_threshold`: smallest share in a flagged component's bias that makes a network a suspect (default 0.5)
    - `tolerance`: a drop is rejected if another component ends above `bias.z_threshold` with its largest $Z$ risen by more than this (default 1.0)
    - `z_threshold`: a suspect closes badly in a member if its own largest $Z$ exceeds this (default 5.0)
- `wcs`: names of the $n_{\rm wcs}$ Wilson coefficients, in the order of the `coefs` columns; as in the training configs
- `workers`: number of members processed at once, one single-threaded process each, in the ensemble fit and in the per-member steps of validation (default 1). Memory grows with it ([`docs/ensemble.md`](../../docs/ensemble.md), Running it); give the run as many cores
- `year_loc`: optional column of the features holding the integer era index, used to look up `lumi` (default 21)

## Nuisance validation

Run with `python -m topsbi.nuisance_validation -c config.yml [--refit] [--plots-only]`.

- `--refit` builds the nuisance cache `{name}/nuisance/` and replaces it. Without it, a missing or stale cache stops the run.
- `--plots-only` redraws `plots/` from the stored results.

The settings shared with the EFT ensemble are `data`, `c0`, `wcs`, `lumi`, `eras`, `features`, `year_loc` and `bias`. They are taken from the EFT validation config that `ensemble` names; setting one of them here stops the run.

- `device`: device the nuisance networks are evaluated on (default `cpu`)
- `ensemble`: path to the EFT validation config; its cache `{name}/ensemble/` must exist and be current
- `factorisation`: optional
    - `points`: `{label: point}`, WC points (1 followed by one value per WC) at which the signal `weight_shift` networks and the learned model are tested (default `{sm: θ_SM}`)
- `n_threads`: torch threads of the main process (default 1)
- `name`: output directory: the nuisance cache (`nuisance/`), `nuisance_diagnostics.yml`, `nuisance_sums.npz` and `plots/`
- `problems`: optional thresholds of the possible problems in `nuisance_problems.yml` ([`docs/validation.md`](../../docs/validation.md)): `not_learned` (0.8), `closure_size` (1.0), `yield_n_sigma` (3.0), `yield_fraction` (0.2), `factorisation` (1.0), `combined_excess` (0.5)
- `seeds`: the members, `{member: {networks: [...]}}`: per member, the training configs of its nuisance networks, both directions of every nuisance. Every member must have the same nuisances, and every network must be trained against the point of `c0`. A network trained on a list of per-era systematics gives one nuisance per listed name
- `workers`: members processed at once in the cache build, one single-threaded process each (default 1)
