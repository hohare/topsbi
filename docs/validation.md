# Validation

The last check of an ensemble before it is used for likelihood scans: whether every coefficient of the learned ratio is unbiased, which training networks cause a bias, whether removing them helps, and which new training points would reduce the uncertainty that remains.

```sh
python -m topsbi.validation -c <config.yml> [--refit]
```

The validation of the nuisance networks and of the learned model with them is in section 10.

The configuration keys are listed in [`examples/validation/README.md`](../examples/validation/README.md). The ensemble it tests, its cache and `--refit` are described in [`ensemble.md`](ensemble.md); notation as there: $a$ a training network (row of $\Theta$), $e$ an ensemble member, $q$ a kept term of $T(\theta)$ (`k` in the code), $\langle \cdot \rangle_e$ the mean over members.

## 1. What it does and reads

1. **Ensemble.** Read the cache `{name}/ensemble/`, or fit it with `--refit`.
2. **Bias test** (section 3). Compare every component $\hat\gamma_q$ (row $q$ of $\hat\Gamma$) with its truth in bins of every network input, and flag the components with a significant bias.
3. **Attribution** (section 4). Split the bias of each flagged component exactly into the contributions of the training networks.
4. **Suspects and drop tests** (section 5). Networks that carry a large share of a bias and close badly on their own are proposed for dropping; each drop is tested by refitting without it.
5. **Point search** (section 6). Propose new training points that most reduce the uncertainty of the flagged components.

Besides the ensemble it reads `data` (the events the ensemble is evaluated on; the per-event structure constants $s_q(z)$ in it give the truth), `lumi` (luminosity per era, column `year_loc`), `c0` and `features` (the network inputs and their columns). Outputs go to `{name}/`: `ensemble_diagnostics.yml`, `proposed_points.yml` and `plots/`.

## 2. Running it

Without `--refit` the cache must exist and match the config ([`ensemble.md`](ensemble.md), section 3). The steps after the fit read it memory-mapped. `workers` (default 1) also runs the per-member steps in parallel, one process per member: the binned sums of every network (section 4), and the refits of the drop tests (section 5). Their memory per worker is of the order of one member's $R$ in float64; the needs of the fit itself are given in [`ensemble.md`](ensemble.md). The bias test, the attribution and the point search are fast compared with the fit.

## 3. Bias test

**Binning.** Each network input $f$ is divided into `bias.n_bins` equal-count bins (one bin per value for a discrete input), and neighbouring bins are merged until each holds `bias.min_neff` effective events, $n_{\rm eff} = (\sum w)^2 / \sum w^2$ with $w = w^s(\theta_0)$. These are the bins $K_b$, $b \in B_f$.

**Bias.** For component $q$ the true per-event contribution is $w_{q,i} = L\, s_q(z_i) \times 1000$ and the learned one $\hat w_{q,i} = w^s_i(\theta_0)\, \hat\gamma_{q,i}$. Per bin:

- $\Delta_{q,b} = \left\langle \sum_{i \in K_b} (\hat w_{q,i} - w_{q,i}) \right\rangle_e$, the bias of the ensemble mean (each member sums over the events it keeps);
- $\mathrm{Var}_{\rm MC} = \sum_{i \in K_b} (\langle \hat w_{q,i} \rangle_e - w_{q,i})^2$, its variance from the finite number of events;
- $\mathrm{Var}_{\rm seeds}$, the variance of the ensemble mean over members (the variance of the members' sums divided by their number).

**Z.** Per component and input, over the bins with events of the component, with $\rho_b = \mathrm{Var}_{\rm seeds} / \mathrm{Var}_{\rm MC}$:

$$\chi^2_{q,f} = \sum_{b \in B_f} \frac{\Delta_{q,b}^2}{\mathrm{Var}_{\rm MC}}, \qquad \mu = \sum_b (1 + \rho_b), \qquad V = 2 \sum_b (1 + \rho_b)^2, \qquad Z_{q,f} = \frac{\chi^2_{q,f} - \mu}{\sqrt{V}}.$$

Without bias each term is $(1 + \rho_b)$ times a $\chi^2_1$ variable, so $\mu$ and $V$ are the mean and variance of $\chi^2$ and $Z$ has mean 0 and variance 1. $Z$ is a screening statistic, not a Gaussian significance: the $\chi^2$ of a few tens of bins is right-skewed, so a large $Z$ overstates how unlikely the excess is. A component is flagged if its largest $Z$ over the inputs exceeds `bias.z_threshold`. The $Z$ of different inputs of one component are correlated, since all inputs bin the same events.

**rms bias.** The size of the bias, per component and input:

$$\text{rms bias}_{q,f} = \sqrt{\max\left(\left\langle \frac{\Delta_{q,b}^2 - \mathrm{Var}_{\rm MC} - \mathrm{Var}_{\rm seeds}}{\lambda_{K_b}(\theta_{\rm SM})^2} \right\rangle_{b \in B_f},\ 0\right)},$$

the noise-subtracted rms bias per bin as a fraction of the SM yield $\lambda_{K_b}(\theta_{\rm SM})$, per unit WC (or WC$^2$ for quadratic and cross terms). $Z$ measures significance and grows with the number of events; the rms bias measures size and does not. Read them together: a large $Z$ with a tiny rms bias is a real but negligible bias.

## 4. Attribution

Each member's coefficients are linear in its ratios, $\hat\Gamma = \Theta^- R$, with $\Theta^-$ the generalized inverse of the ensemble fit ($\Theta\,\Theta^-\Theta = \Theta$): the pseudo-inverse $\Theta^+$ for least squares over every network, or the inverse of the fit whose background columns come from a subset of the networks (`tools/build_ratio.py` `generalized_inverse`). The binned bias of a component is therefore exactly a sum over the training networks,

$$\Delta_{q,b} = \sum_a (\Theta^-)_{q,a}\, \Delta_{a,b},$$

with $\Delta_{a,b} = \sum_{i \in K_b} w^s_i(\theta_0) \left(\hat r_a(x_i) - r_a(x_i, z_i)\right)$ the binned misclosure of network $a$. The share of network $a$ in the bias of component $q$ is the projection of its term on $\Delta_q$, weighted by $1 / \lambda_{K_b}(\theta_{\rm SM})^2$ and averaged over members; the shares of a component sum to 1 and can be negative (a network that cancels part of the bias). `attribution_identity` checks that the terms rebuild $\Delta$ (up to float32 rounding).

Each network's own closure, from the same sums, gives its misclosure $m_a$: the noise-subtracted rms of $\Delta_{a,b} / \lambda_{K_b}(\theta_{\rm SM})$ over bins, with $m_a^2$ averaged over members.

## 5. Suspects and drop tests

A network is a suspect if its share in some flagged component reaches `suspects.share_threshold`. It closes badly in a member if its own $Z$ (the test of section 3 on $\Delta_{a,b}$) exceeds `suspects.z_threshold` there. Bad in at least `suspects.seed_fraction` of the members: the training point itself is the problem, and it is a drop candidate. Bad in fewer, the network stays in the fit.

Drop candidates are grouped (the points of one cross-term pair together, a single-WC point alone) and each group is tested by refitting every member without it. A drop is accepted if the largest $Z$ of some component it was blamed for falls by at least `suspects.min_gain`, and no component ends above `bias.z_threshold` with its largest $Z$ risen by more than `suspects.tolerance`. A group whose removal would leave $\Theta$ rank-deficient is not tested. Drops that pass alone may interact, so all accepted drops are then refitted together (`combined`).

The refits use the stored $R$, so no network is evaluated again.

## 6. Point search

With the misclosures as independent network errors, the propagated covariance of the coefficients for a design $\Theta$ is

$$\mathrm{Cov}(\hat\gamma) = H^+ \Theta^T \mathrm{diag}(m_a^2)\, \Theta\, H^+, \qquad H = \Theta^T \Theta,$$

and $\sqrt{\mathrm{Var}(\hat\gamma_q)}$ its diagonal (`sigma_propagated`). Components outside the range of $\Theta$ (a part of their direction in its null space) cannot be determined by the design; they are marked `constrained: false` and get no uncertainty.

The search adds candidate units one at a time, each time the one that most lowers the objective $\mathrm{tr}(\mathrm{Cov}(\hat\gamma)\, G)$, and stops after `point_search.n_units` units or when a step lowers it by less than the fraction `point_search.min_gain`.

- **Weights.** `G: flagged` (default): $G = \mathrm{diag}(g_q / \mathrm{Var}(\hat\gamma_q))$ with the variance of the current design, $g_q = 1$ for flagged components and `unflagged_weight` for the others, so the objective is the $g$-weighted sum of relative variance reductions. `G: scan`: $G = \langle T(\theta) T(\theta)^T \rangle$ over a scan region (`bounds` per WC, `mode` box or 1d, `shape` uniform or gaussian), the uncertainty of $\hat r$ averaged over the region.
- **Candidates.** For every cross term $\theta_i \theta_j$, units of the four points $(\pm\theta_i, \pm\theta_j)$ built from the existing single-WC training values, with the misclosure of the worse of the two source networks assumed for the new networks. Units whose points all exist, or with a point whose inclusive yield relative to the SM exceeds `max_yield_ratio` (default: the largest of the existing points), are left out.

The dropped networks leave the design before the search. `sigma_after_search` is the propagated uncertainty with the chosen units added. The search assumes the new networks close as well as existing ones; it cannot fix a bias that every network shares.

## 7. Outputs

### `ensemble_diagnostics.yml`

| Key | Measures | Healthy ensemble |
|---|---|---|
| `ensemble.path`, `created`, `seeds`, `rank` | the cache used; rank of each member's $\Theta$ | `rank` = number of kept terms |
| `ensemble.residual_outliers` | members whose median fit residual is more than 3 times the typical one | empty |
| `binning.<input>` | number of bins $K_b$ of each input after merging | |
| `flagged` | components with largest $Z$ above `bias.z_threshold`, largest first | empty |
| `attribution_identity` | max $\lvert$ rebuilt $-\ \Delta \rvert$ / max $\lvert \Delta \rvert$ (section 4) | float32 rounding, about $10^{-3}$ |
| `components.<q>.kind` | `sm`, `linear`, `quadratic` or `cross` | |
| `components.<q>.Z`, `max_Z`, `worst_feature` | $Z_{q,f}$ per input, its largest value and that input | $\lvert Z \rvert$ of a few at most, below the threshold |
| `components.<q>.rms_bias` | the largest rms bias over the inputs | small compared with the precision the analysis needs |
| `components.<q>.flagged` | in `flagged` | false |
| `components.<q>.constrained` | determined by the design (section 6) | true |
| `components.<q>.sigma_propagated`, `sigma_after_search` | $\sqrt{\mathrm{Var}(\hat\gamma_q)}$ for the current design and with the proposed points; null if unconstrained | below the rms bias the analysis can tolerate |
| `components.<q>.top_networks` | flagged components: the `n_top_networks` largest signed shares | |
| `components.<q>.refit.max_Z`, `refit.rms_bias` | after refitting without the accepted drops | |
| `suspects.drop` | networks dropped (accepted drop tests) | empty |
| `suspects.drop_tests.<group>.members`, `accepted`, `gain`, `harmed` | the networks of a group; accepted; largest fall of $Z$ among the components it was blamed for; {component: [Z before, Z after]} for harmed components, or why the group could not be tested | |
| `suspects.combined` | the same for all accepted drops at once | |
| `network_misclosure.<network>` | $m_a$ of every training network (section 4) | similar for all networks of a kind |
| `point_search.G` | the weights used | |
| `point_search.candidate_units` | number of candidate units | |
| `point_search.objective` | objective before the search and after each added unit | |
| `point_search.proposed` | the chosen units: kind, point names, largest yield ratio of their points | |

The settings of a run (thresholds, `n_units`, ...) are those of its config.

### `proposed_points.yml`

`wcs` (the WC order) and `points`, each with `name`, `kind` (`cross`) and `theta` (SM-inclusive, first entry 1): the new training points, ready to be turned into training configs.

## 8. Plots (`plots/`)

| File | Shows | Look for |
|---|---|---|
| `fit_residuals.png`, `fit_residuals_log.png` | distribution of the per-event fit residuals: median over members, band of their spread, outlying members as separate lines | a member far from the others |
| `z_heatmap_full.png`, `z_heatmap_compact.png` | $Z$ of every component (columns, grouped by kind) in every input (rows); cells above the threshold outlined. Compact: the flagged components and the next 10 | rows (inputs) or columns (components) with many outlined cells |
| `closure/flagged/<q>.png`, `closure/passed/<q>.png` | one component in its worst input: learned and true sums per bin, and $\Delta$ relative to the SM yield with the noise band $\sqrt{\mathrm{Var}_{\rm MC} + \mathrm{Var}_{\rm seeds}}$ | where in the input the bias sits, and whether it is a shape or a normalisation |
| `attribution/<q>.png` | flagged components: the signed shares of the 10 largest networks and the sum of the others | one dominant network (a candidate to drop) or many small shares (a bias the networks share) |
| `drop_tests_compact.png`, `drop_tests_full.png` | change of the largest $Z$ of the affected components for every tested group and for the combined refit; blamed cells outlined, accepted groups in dark red | gains where the group was blamed, no new outlined cells elsewhere |
| `network_misclosure.png` | $m_a$ of every network against the inclusive yield of its training point, by kind, with the spread over members; suspects marked | networks far above the others of their kind |
| `point_search_objective.png` | the objective after each added unit, relative to the current design | how fast the gain falls off: a flat curve means more points will not help |
| `point_search_uncertainty_flagged.png`, `point_search_uncertainty_full.png` | $\sqrt{\mathrm{Var}(\hat\gamma_q)}$ before and after the search, over the measured rms bias | a bias well above the propagated uncertainty: more points will not remove it |

## 9. Reading the results

- **Nothing flagged.** The ensemble closes in every term within the noise of the validation sample; it can be used.
- **A flagged component with one dominant network.** Check that network's own closure (`network_misclosure`, its training plots); accept the drop if the test does. A network whose share is large but whose copies close badly in only a few members is left in; its share is noise of those members.
- **A flagged component with many small shares** and a bias well above `sigma_propagated`: every network carries the same bias, so neither dropping nor adding points removes it; the networks themselves need to learn that term better (capacity, training points closer to where the term matters, more training events), or the term may vanish by construction and belong in `zero_terms`.
- **A bias comparable to `sigma_propagated`**: more training points help; `proposed_points.yml` lists them.
- **Remaining flags after all of this** limit what the ensemble can be trusted for: test their effect on the scan (bias of the scan minimum on Asimov data) before using it.

The bias test can only resolve biases larger than the noise of the validation sample; an unflagged component is consistent with zero bias at that precision, not proven unbiased.

## 10. Nuisance validation

The last check before likelihood scans: whether the nuisance networks close, whether their interpolation in $\nu$ reproduces the true Up and Down variations, whether a network trained at $\theta_{\rm ref}$ still holds at other WC points, and how much each nuisance adds to the error of the learned model the scans use.

```sh
python -m topsbi.nuisance_validation -c <config.yml> [--refit] [--plots-only]
```

The configuration keys are listed in [`examples/validation/README.md`](../examples/validation/README.md) (Nuisance validation); the learned model and the nuisance cache are described in [`ensemble.md`](ensemble.md), section 8. Notation as in the sections above, with $n$ a nuisance, $\nu$ its parameter and $\hat f^{\pm}$ the output of its Up and Down networks.

### 10.1 What it reads

- **The EFT validation config** named by `ensemble`: `data`, `c0`, `wcs`, `lumi`, `eras`, `features`, `year_loc` and `bias` are taken from it, so both ensembles see the same events and bins. Its cache `{name}/ensemble/` gives the learned EFT ratio.
- **The nuisance cache** `{name}/nuisance/`: $\hat f$ of every nuisance network of every member on every event, built with `--refit`. Every network must be trained at $\theta_{\rm ref}$, the point of `c0`; the run stops otherwise. $\theta_{\rm ref}$ is the reference point of the training samples (`reference` in their `sample.yml`: the generation point or the SM).
- **The targets.** `weight_shift`: the events of `data` with $w^s(\theta_{\rm ref}, \nu) = w^s(\theta_{\rm ref}, 0)\, r(x, z; \theta, \nu)$ (`syst_ratio`). `kinematic_shift`: the shifted events of the variation file in the network's training data, binned by their own $x'$ and matched to the nominal events by event keys.

Each nuisance is tested on the events of its term (the signal, or one background process) in its eras. `--plots-only` redraws `plots/` from the stored results.

### 10.2 Common definitions

**Bins.** The bins $K_b$ of the bias test (section 3), merged further until the events of every term hold `bias.min_neff` effective events in every bin, so the same bins serve every nuisance.

**Bias.** Per test and bin, as in section 3: $\Delta_b = \langle \hat w_b - w_b \rangle_e$, with $\hat w_b$ the learned and $w_b$ the true yield of the tested events; $\mathrm{Var}_{\rm MC}$, in which an event in both the learned and the target sample counts once; and $\mathrm{Var}_{\rm seeds}$ from the members' sums. $Z$ is the statistic of section 3, and a test is flagged when its largest $Z$ over the inputs exceeds `bias.z_threshold`. The era input is not tested for a nuisance of one era, whose events fill one of its bins.

**rms bias.** As in section 3, but relative to $\lambda_{K_b}(\theta, 0)$, the yield of every event (every term, every era) at the point of the test. All tests and nuisances share this normaliser, so their rms biases can be compared with each other and with that of the learned EFT ratio. The stored `rms_bias` is the largest over the inputs. Each input's value is noise-subtracted, which is unbiased only on average; with many inputs the largest picks up their upward fluctuations. Below the flagging threshold, differences of the order of the noise level carry no meaning.

**Kinematic truth.** For a `kinematic_shift` network an event moves between bins when its features are shifted, so its nominal and shifted rows enter different bins and do not cancel in $\mathrm{Var}_{\rm MC}$. Its closure is therefore noisier than that of a `weight_shift` network with the same number of events, whose target is the same events reweighted.

### 10.3 Tests

1. **Closure** (`down`, `up`). Per network, $\hat w_b = \sum_{i \in K_b} w^s_i(\theta_{\rm ref}, 0)\, e^{\hat f(x_i)}$ per member, against the target $w_b$. From the same sums, the misclosure of each member alone,
   $m_e = \sqrt{\langle \max(\Delta_{e,b}^2 - \mathrm{Var}_{\rm MC},\ 0) / \lambda_{K_b}^2 \rangle_b}$ over the bins of every tested input, combined as $m_a = \sqrt{\langle m_e^2 \rangle_e}$; and $m$, the same for the ensemble mean. $\mathrm{Var}_{\rm seeds}$ is not subtracted here: $m_a / m$ is about $\sqrt{n_e}$ when the members' errors are independent training noise that the average cancels, and about 1 when the members share one bias. `shift_size`, the rms of the true shift $(w_b(\pm 1) - w_b(0)) / \lambda_{K_b}$ over the same bins, sets the scale.
2. **Parts of the interpolation** (`fit_symmetry`, `fit_slope`). Combine interpolates a shape variation from $\delta^{\pm} = \hat f^{\pm}$: for $\lvert\nu\rvert \le 1$,
   $$\log \hat r(x; \nu) = \nu\, \hat f^{\rm slope} + h(\nu)\, \hat f^{\rm sym}, \qquad \hat f^{\rm slope} = \tfrac{1}{2}(\hat f^{+} - \hat f^{-}),\quad \hat f^{\rm sym} = \tfrac{1}{2}(\hat f^{+} + \hat f^{-}),\quad h(\nu) = \tfrac{1}{8}(15\nu^2 - 10\nu^4 + 3\nu^6),$$
   and linearly beyond. Each part is tested by its own closure, $\Delta^{\rm slope}_b = \tfrac{1}{2}(\Delta^{+}_b - \Delta^{-}_b)$ against the true $\tfrac{1}{2}(w^{+}_b - w^{-}_b)$, and $\Delta^{\rm sym}_b = \tfrac{1}{2}(\Delta^{+}_b + \Delta^{-}_b)$ against $\tfrac{1}{2}(w^{+}_b + w^{-}_b) - w^0_b$. A spurious $\hat f^{\rm sym}$ shows up in the second.
3. **Yield against $\nu$.** $\hat\kappa(\nu) = \hat\lambda(\nu) / \lambda(0)$ on the tested events, the weighted mean of $e^{\log \hat r(x; \nu)}$ with $\langle \hat f \rangle_e$, against the true $\kappa(\pm 1)$. $\kappa(\pm 1)$ are the Up and Down values a lnN uncertainty would carry. The truth exists only at $\nu = -1, 0, +1$; between them the interpolation follows Combine's convention, and its effect on a fit is seen in the likelihood scans.
4. **Factorisation** (`factorisation.<label>`), at the points of `factorisation.points` (default $\theta_{\rm SM}$). The events are reweighted from $\theta_{\rm ref}$ to $\theta$ by $\rho_i(\theta) = s(z_i) \cdot T(\theta) / s(z_i) \cdot T(\theta_{\rm ref})$, the target by the same factor, and the network trained at $\theta_{\rm ref}$ is closed again. This tests $r(x; \theta, \nu) / r(x; \theta, 0) \approx \hat r(x; \theta_{\rm ref}, \nu)$, the factorisation the learned model assumes. Only the signal `weight_shift` networks are tested: the variation files of `kinematic_shift` have no structure constants, and $\rho = 1$ on background events. If $\theta_{\rm ref} = \theta_{\rm SM}$, the default point repeats the closure; set other points.
5. **Combined** (`combined.<label>`). The learned model of the scans at each point and $\nu = \pm 1$, on the events of every term in the nuisance's eras: the nuisance multiplies the term of the learned EFT ratio it belongs to, the other terms enter unchanged, $\hat w = w^s(\theta_0, 0) \left[\hat r_{\rm term}(x; \theta)\, e^{\hat f} + \sum_{\rm other} \hat r_{\rm other}(x; \theta)\right]$, against $w^s(\theta, \nu)$. Its baseline is the learned EFT ratio alone at $\nu = 0$ on the same events (`eft_matched`), so `rms_bias_change` is what the nuisance adds to the error of the learned EFT ratio. `eft_only` is the learned EFT ratio alone on every event. Tested for the `weight_shift` networks.

### 10.4 Possible problems (`nuisance_problems.yml`)

Flags measure significance, which grows with the number of events. The possible problems measure size: each compares a misclosure with the shift it describes or with the error of the learned EFT ratio. Thresholds are tolerances set in `problems`, not derived values.

| Criterion | Value | Default |
|---|---|---|
| `not_learned` | $m$ / `shift_size`: the error of the ensemble mean as a fraction of the shift it learns | 0.8 |
| `closure_size` | closure `rms_bias` / `rms_bias` of the learned EFT ratio alone at $\theta_{\rm SM}$ (needs the point `sm`) | 1.0 |
| `yield_mismatch` | rate closure $\lvert\hat\kappa - \kappa\rvert$ at $\nu = \pm 1$ in units of $\sqrt{\mathrm{Var}_{\rm MC} + \mathrm{Var}_{\rm seeds}}$ of the rate, counted only when $\lvert\hat\kappa - \kappa\rvert$ exceeds `yield_fraction` times $\lvert\kappa - 1\rvert$ | `yield_n_sigma` 3.0, `yield_fraction` 0.2 |
| `factorisation` | factorisation `rms_bias` / `rms_bias` of the learned EFT ratio alone at that point | 1.0 |
| `combined_excess` | combined `rms_bias_change` / its baseline | 0.5 |

The rate closure sums every tested event in one bin, so its $\mathrm{Var}_{\rm MC}$ keeps the events that move between bins. Each problem has `nuisance`, `test` (the direction, or `<test>/<point>/<direction>`), `criterion`, `value`, `threshold` and `severity` = value / threshold; the file lists them largest severity first, with the thresholds used.

### 10.5 Outputs

#### `nuisance_diagnostics.yml`

| Key | Measures |
|---|---|
| `nuisance.path`, `created`, `seeds`, `theta_ref`, `ensemble` | the nuisance cache used, its reference point and the EFT cache |
| `binning.<input>` | number of bins $K_b$ of each input |
| `flagged` | network-directions whose closure is flagged, largest $Z$ first, named as their closure plots |
| `factorisation_points` | labels of the factorisation points |
| `eft_only.<label>` | the learned EFT ratio alone at that point: `flagged`, `max_Z`, `worst_feature`, `rms_bias`, `Z` |
| `nuisances.<n>.method`, `term`, `eras` | the nuisance |
| `nuisances.<n>.<direction>` | closure: `flagged`, `max_Z`, `worst_feature`, `rms_bias`, `Z` per input; `misclosure` ($m_a$), `member_misclosure` ($m_e$), `mean_misclosure` ($m$), `shift_size`, `members_flagged` (members whose own closure is flagged) |
| `nuisances.<n>.fit_symmetry`, `fit_slope` | closure of each part, keys as the closure |
| `nuisances.<n>.factorisation.<label>.<direction>` | factorisation test, keys as the closure, and `rms_bias_change` from the closure; null for the nuisances not tested |
| `nuisances.<n>.combined.<label>.<direction>` | combined test, keys as the closure, with `baseline` (`eft_only`), `baseline_rms_bias` and `rms_bias_change`; null for the nuisances not tested |

#### `nuisance_sums.npz`

The per-bin arrays behind the diagnostics, for redrawing plots or further checks: `nuisances` and `methods` (network tests have row $k = 2n + d$, $d$ = 0 Down, 1 Up); `binning/names`, `binning/feature`, `binning/discrete`, `binning/edges/<input>`; per test (`closure`, `fit_symmetry`, `fit_slope`, `factorisation/<label>`, `combined/<label>`, `eft_only/<label>`, `eft_matched/<label>`) the [rows, bins] arrays `D` ($\Delta$), `s2` ($\mathrm{Var}_{\rm seeds}$), `mc_var` ($\mathrm{Var}_{\rm MC}$), `truth` ($w_b$), `sm_yield` ($\lambda_{K_b}(\theta, 0)$), `nominal` ($w_b(0)$ of the tested events), `shift_var` and `learned_shift_var` (the MC variances of the true and the learned shift); for `closure`, `factorisation/<label>` and `combined/<label>` also the [rows] arrays `rate_mc_var` and `rate_s2` of the rate closure; and `interpolation/nu`, `interpolation/yield` ($\hat\kappa(\nu)$ per nuisance).

### 10.6 Plots (`plots/`)

Nuisances are ordered by method, then by name.

| File | Shows | Look for |
|---|---|---|
| `z_heatmap_full.png`, `z_heatmap_compact.png` | $Z$ of every network (Down beside Up, grouped by method) in every input; outlined above the threshold, hatched where not tested. Compact: the flagged networks and the next 10 | inputs or networks with many outlined cells |
| `z_heatmap_parts_full.png`, `z_heatmap_parts_compact.png` | the same for $\hat f^{\rm slope}$ and $\hat f^{\rm sym}$ | a part flagged where the networks are not |
| `size_vs_significance_networks.png`, `_slope.png`, `_symmetry.png` | rms bias against largest $Z$ per test, by method, with the threshold and the rms bias of the learned EFT ratio at $\theta_{\rm SM}$ | flagged tests with a large rms bias; flagged but tiny is real and negligible |
| `network_misclosure.png` | $m$ of every network against its `shift_size`, with each member's $m_e$ and their range; flagged networks edged; the line $m$ = shift | networks near or above the line; the large marker near the top of its range (a bias the members share) |
| `closure/{flagged,passed}/<n>_<direction>.png` | one network in its worst input: learned and true shift per bin, and $\Delta_b / \lambda_{K_b}$ with the noise band | where the misclosure sits, shape or normalisation |
| `closure_parts/{flagged,passed}/<n>_<part>.png` | the same for each part | a symmetric part the truth does not have |
| `interpolation_yield/<n>.png` | $\hat\kappa(\nu) - 1$ from the interpolation, with the true values at $\nu = -1, 0, +1$ and their MC noise | the curve missing a true point |
| `yield_summary_<method>.png` | per nuisance, the true and learned $\kappa - 1$ at $\nu = \pm 1$ (left) and the pull of the rate closure with a band at $\pm$`yield_n_sigma` (right); `yield_mismatch` problems in red | pulls outside the band on large shifts |
| `factorisation_<label>.png` | rms bias of each signal `weight_shift` network at $\theta_{\rm ref}$ (open) and at the point (filled); `factorisation` problems ringed; dashed line at the threshold times the rms bias of the learned EFT ratio at the point | long arrows to the right, filled markers past the line |
| `eft_combined_<label>.png` | rms bias of the learned EFT ratio alone (open) and of the learned model with the nuisance (filled), on the events of each test; `combined_excess` problems ringed; ticks at (1 + threshold) times the baseline | filled markers past their ticks |

In `factorisation_<label>.png`, $\hat r(x; \theta, \nu)$ is the nuisance ratio relative to $(\theta, 0)$; in `eft_combined_<label>.png`, it is the learned model relative to $\theta_0$. The open markers of `eft_combined_<label>.png` differ between nuisances because each baseline uses only the events of the nuisance's eras.

### 10.7 Reading the results

- **Nothing flagged and no possible problems.** The nuisance networks and their interpolation close within the noise of the validation sample, and the learned model holds at the tested points.
- **Flagged, but no possible problem.** A real misclosure that is small compared with the shift and with the error of the learned EFT ratio; it can usually be accepted.
- **`not_learned` or a large closure.** The network has not learned its shift: check its training plots ([`training.md`](training.md)); for a large ratio $m_a / m$ more members help, for a ratio near 1 they do not.
- **`yield_mismatch`.** The normalisation of the shift is wrong: compare with `yield_summary_<method>.png` and the closure of the parts.
- **`factorisation` or `combined_excess`.** The nuisance does not factorise from the WC dependence at that point: its shift depends on how the event mix changes with $\theta$. The model of the scans is then off by more than the learned EFT ratio's own error; test the effect on the scans (or model the nuisance differently) before using it.
- **A part flagged where the networks are not.** The parts combine the Up and Down misclosures: errors the two networks share add in the symmetric part and cancel in the slope, and the noise of the combination is smaller when the two samples share events. A flagged symmetric part is a spurious $\hat f^{\rm sym}$, which bends the interpolation inside $\lvert\nu\rvert < 1$.

The tests resolve misclosures larger than the noise of the validation sample; an unflagged nuisance is consistent with closure at that precision, not proven exact.
