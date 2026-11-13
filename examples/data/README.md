## Data files

Examples of the yml files that accompany the training data, laid out as a combined data root (`<root>/nominal/` and one
directory per kinematic variation). The tensors themselves are described in [`docs/data.md`](../../docs/data.md).

## `nominal/systematics.yml`

Read by `weight_shift` trainings, next to the nominal `train.p` / `validation.p`.

- `syst_ratio`: names of the weight systematics, in the order of axis 1 of the `syst_ratio` tensor; each name in a
  config's `systematic` must be one of them
- `directions`: order of the tensor's last axis (default `[Up, Down]`); a config's `direction` must be one of them

## `nominal/sample.yml`, `<variation>/sample.yml`

Next to `train.p` / `validation.p` in every directory of the combined data.

- presence: marks the directory as combined data (an `eft` training on `<root>/nominal` then splits train / test by
  event key and uses `validation.p` as the default holdout)
- `reference`: the point at which `w_ref` is the event weight; must be the same in nominal and every variation
- `reference_point`: the value of that point, SM entry first, then one value per WC in the config's `wcs` order; read
  from the signal's nominal `sample.yml` by background trainings (written to `performance.yml`) and by the nuisance
  cache
- `salt_split`: the salt of the train / validation split; must be the same in nominal and every variation
- `eft`: `false` marks a background sample (default `true`, signal)
- `process`: the background process's name; required with `eft: false`
- any other keys are free and not read (e.g. `variation`, `columns`)

## `zero_terms.yml`

Referenced by the training config's `zero_terms` key; it depends on the samples, not on a single training.

- `zero_terms`: list of WC pairs (names as in the config's `wcs`, any order within a pair) whose quadratic term is
  structurally zero; their columns of the `get_lower_tri` expansion are dropped
- finding them: `topsbi.tools.data.find_zero_terms(coefs, wcs, lumi_per_event, features, n_bins=20, tol=1e-6)`
  returns the list for a sample. Every feature column is cut into `n_bins` equal-count quantile bins; in each bin a
  term's size is $\sum_i L |s_q(z_i)| / \sum_i L |s_{\rm SM}(z_i)|$ over the events in the bin, and a term is listed
  only if its largest size over all bins of all features is below `tol`. A term that is small overall but large in
  one region (a tail) is therefore kept. Structurally zero terms sit at the level of numerical noise in every bin,
  far below the kept terms; if the smallest kept and largest dropped sizes are close, inspect the terms near `tol`
  rather than trusting the cut.
- every network of an ensemble must use the same `zero_terms` (see [`docs/ensemble.md`](../../docs/ensemble.md)):
  find the list once per sample, before training, and regenerate it (and retrain) only if the samples change
