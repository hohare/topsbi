from topsbi.tools.build_ratio import seed_map
from topsbi.tools.data import (
    NUISANCE_COLUMNS,
    event_keys,
    get_lower_tri,
    load_combined,
    matched_events,
    sample_weight,
    syst_ratio_column,
)
from topsbi.tools.ensembleCache import EnsembleCache, bin_events, binned_sums, merge_sparse, quantile_binning
from topsbi.tools.nuisanceCache import (
    DIRECTIONS,
    NuisanceCache,
    interpolate,
    nuisance_eras,
    nuisance_networks,
    nuisance_seed,
    reference_point,
    write_nuisance_cache,
)
from topsbi.tools.plots import (
    DIRECTION_TEX,
    METHOD_NAMES,
    plot_interpolation_shape,
    plot_interpolation_yield,
    plot_nuisance_closure,
    plot_rms_comparison,
    plot_size_significance,
    plot_z_heatmap,
)
from topsbi.validation import component_bias

import numpy as np

import argparse, os, shutil, sys, torch, yaml

# taken from the EFT validation config, so both ensembles see the same events, reference point and luminosities
SHARED_KEYS = ('data', 'c0', 'wcs', 'lumi', 'features', 'year_loc')
# max over nu of |d (nu S(nu)) / d nu| for Combine's smooth step, at nu = +-1/sqrt(3): log r_hat(x; nu) is monotonic in
# nu for every nu exactly when |f_hat^slope| >= MONOTONIC * |f_hat^sym|
MONOTONIC = 7 * np.sqrt(3) / 9
# the parts of the interpolation tested in check 2, with their TeX labels
PARTS = {'fit_slope': r'$\hat{f}\,^{\mathrm{slope}}$', 'fit_symmetry': r'$\hat{f}\,^{\mathrm{sym}}$'}


def load_config(config_path):
    """
    The nuisance validation config, completed from the EFT validation config it names (ensemble): data, c0, wcs,
    lumi, features and year_loc are taken from there (a nuisance config that sets one of them stops the run),
    eft_cache is the EFT ensemble cache, {name}/ensemble of that config, and eft_config that config itself.

    Args:
        config_path: nuisance validation config yml (name, ensemble, seeds, optional device and workers)
    Returns:
        config dict
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(config['ensemble']) as f:
        eft = yaml.safe_load(f)
    for key in SHARED_KEYS:
        if key in config:
            sys.exit(f'[error]: {key} is taken from the EFT validation config {config["ensemble"]}; remove it')
        if key in eft:
            config[key] = eft[key]
    if isinstance(config['features'], str):
        with open(config['features']) as f:
            config['features'] = yaml.safe_load(f)
    config['eft_cache'] = f'{eft["name"]}/ensemble'
    config['eft_config'] = eft
    return config


def load_nuisance_ensemble(config, features, refit=False):
    """
    The nuisance cache of config: read from {name}/nuisance when it matches config (EFT ensemble, data, networks and
    their model.pt), otherwise built and written, but only when refit is set. Its reference point theta_ref must be
    the EFT reference point c0, since the two ensembles are combined as log r_hat(x; theta) + log r_hat(x; nu).

    Args:
        config: nuisance validation config, completed by load_config
        features: [N, n_features] raw network inputs of config['data']
        refit: build the cache and replace it even if one exists
    Returns:
        NuisanceCache
    """
    path = f'{config["name"]}/nuisance'
    if not refit:
        if not os.path.exists(f'{path}/nuisance.yml'):
            sys.exit(f'[error]: no nuisance cache in {path}; rerun with --refit to build one')
        cache = NuisanceCache(path)
        if not cache.matches(config):
            sys.exit(
                f'[error]: {path} was built from another ensemble, data, networks or model.pt files; rerun with --refit'
            )
    else:
        nuisances, networks = nuisance_networks(config['seeds'])
        theta_ref = reference_point(nuisances, networks)
        year_loc = config.get('year_loc', 21)
        for nuisance, eras in zip(nuisances, nuisance_eras(nuisances, networks, year_loc)):
            nuisance['eras'] = eras
        tmp = f'{path}.tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        args = (networks, features, config.get('device', 'cpu'), tmp)
        finites = seed_map(nuisance_seed, list(networks), args, config.get('workers', 1))
        event_eras = features[:, year_loc].to(torch.int64).numpy()
        write_nuisance_cache(path, config, nuisances, networks, theta_ref, finites, event_eras)
        cache = NuisanceCache(path)
    if not np.allclose(cache.theta_ref, config['c0'], rtol=0, atol=1e-12):
        sys.exit(
            f'[error]: the nuisance networks were trained at theta_ref = {cache.theta_ref}, '
            f'the EFT ensemble at c0 = {config["c0"]}'
        )
    return cache


class ClosureTargets:
    """
    Target sample of every nuisance network for check 1 (nuisance_sums). weight_shift: the nominal events with
    w^s(theta_ref, nu) = w^s(theta_ref, 0) r(x, z; theta, nu) (syst_ratio). kinematic_shift: the events of the
    variation's file with w^s(theta_ref, nu), binned by their own x' in the bins K_b of the nominal events and matched
    to them by event_keys.
    """

    def __init__(self, config, cache, nominal, w0, binning):
        """
        Args:
            config: nuisance validation config, completed by load_config (data, lumi, features, year_loc)
            cache: NuisanceCache (nuisances with method and eras)
            nominal: tensors of config['data'] (load_combined with features, ids, event_id, syst_ratio, w_ref)
            w0: [N] w^s(theta_ref, 0) of the nominal events
            binning: binning of the nominal events (quantile_binning, then merge_sparse)
        """
        self.config, self.cache, self.binning = config, cache, binning
        # config['data'] is <root>/nominal/<file>; a variation's events are in <root>/<systematic><Up|Down>/<file>
        self.root = os.path.dirname(os.path.dirname(config['data']))
        self.file = os.path.basename(config['data'])
        self.year_loc = config.get('year_loc', 21)
        self.w0 = np.asarray(w0, dtype=np.float64)
        columns = dict(zip(NUISANCE_COLUMNS['nominal'], nominal))
        self.syst_ratio = columns['syst_ratio']
        self.event_eras = columns['features'][:, self.year_loc].to(torch.int64).numpy()
        self.keys0 = event_keys(nominal, 'nominal', self.year_loc)

    def __call__(self, n, d):
        """
        Target of nuisance n, direction d (0 = Down, 1 = Up), as nuisance_sums reads it: rows0, i0, w1, index1, i1.
        """
        rows0 = self.rows0(n)
        if self.cache.nuisances[n]['method'] == 'weight_shift':
            events = np.arange(len(self.w0))
            w1 = self.w0 * self.true_ratio(n, d)
            return {'rows0': rows0, 'i0': events, 'w1': w1, 'index1': self.binning['index'], 'i1': events}
        shifted = load_combined(self._path(n, d), 'variation', ('features', 'ids', 'event_id', 'w_ref'))
        i0, i1 = matched_events(self.keys0, event_keys(shifted, 'variation', self.year_loc))
        return {
            'rows0': rows0,
            'i0': i0.numpy(),
            'w1': sample_weight(shifted, 'variation', self.config).double().numpy(),
            'index1': bin_events(self.binning, shifted[0], self.config['features']),
            'i1': i1.numpy(),
        }

    def _path(self, n, d):
        """File of the shifted events of kinematic nuisance n, direction d."""
        return f'{self.root}/{self.cache.nuisances[n]["name"]}{DIRECTIONS[d].capitalize()}/{self.file}'

    def rows0(self, n):
        """[N] bool, the nominal events nuisance n is tested on: its eras, or every event."""
        eras = self.cache.nuisances[n].get('eras')
        return np.ones(len(self.w0), dtype=bool) if eras is None else np.isin(self.event_eras, eras)

    def true_ratio(self, n, d):
        """[N] r(x, z; theta, nu) of every nominal event for weight_shift nuisance n, direction d (syst_ratio)."""
        k, j = syst_ratio_column(self.root, self.cache.nuisances[n]['name'], DIRECTIONS[d])
        return self.syst_ratio[:, k, j].double().numpy()

    def pair(self, n):
        """
        Targets of nuisance n for Down and Up (as __call__), with one event index i over the nominal, Down and Up
        samples, so an event in several of them is one unit in matched_variance.
        """
        down, up = self(n, 0), self(n, 1)
        if self.cache.nuisances[n]['method'] == 'weight_shift':
            return down, up
        keys = [self.keys0] + [
            event_keys(
                load_combined(self._path(n, d), 'variation', ('features', 'ids', 'event_id')),
                'variation',
                self.year_loc,
            )
            for d in range(len(DIRECTIONS))
        ]
        _, events = np.unique(np.concatenate(keys), axis=0, return_inverse=True)
        events = events.reshape(-1)
        n0, n_down = len(keys[0]), len(keys[1])
        down['i0'] = up['i0'] = events[:n0]
        down['i1'], up['i1'] = events[n0 : n0 + n_down], events[n0 + n_down :]
        return down, up


class ThetaTargets:
    """
    Targets of ClosureTargets moved to the WC point theta (check 3): each weight scaled by
    rho_i(theta) = s_q(z_i) . T(theta) / s_q(z_i) . T(theta_ref) of its event, for weight_shift nuisances.
    """

    def __init__(self, targets, rho, rows=None):
        """
        Args:
            targets: ClosureTargets
            rho: [N] rho_i(theta) of every nominal event
            rows: [N] bool, nominal events to keep (check 4: those the EFT ensemble keeps); default every event
        """
        self.targets, self.rho, self.rows = targets, rho, rows

    def __call__(self, n, d):
        t = dict(self.targets(n, d))
        # weight_shift: the target events are the nominal events (i1 = i0), so rho applies event by event
        t['w1'] = t['w1'] * self.rho
        if self.rows is not None:
            t['rows0'] = t['rows0'] & self.rows
        return t


def factorisation_weights(coefs, theta_ref, points):
    """
    rho_i(theta) = s_q(z_i) . T(theta) / s_q(z_i) . T(theta_ref) per nominal event for every point of check 3.

    Args:
        coefs: [N, n_q] structure constants s_q(z) of the nominal events
        theta_ref: the networks' reference point (SM-inclusive)
        points: {label: theta} WC points (SM-inclusive)
    Returns:
        {label: [N] rho(theta), float64}
    """
    coefs = coefs.to(torch.float64)
    reference = coefs @ get_lower_tri(theta_ref, dtype=torch.float64)
    return {
        label: (coefs @ get_lower_tri(theta, dtype=torch.float64) / reference).numpy()
        for label, theta in points.items()
    }


def eft_ratio(eft, theta):
    """
    [N] learned EFT ratio r_hat(x; theta) = T(theta)[mask] <Gamma>_e of every event of the EFT cache's data, nan
    where the ensemble drops the event.
    """
    r_hat = np.full(eft.finite.shape, np.nan)
    r_hat[eft.finite] = eft(theta)
    return r_hat


def eft_closure(binning, w0, rho, r_hat, keep):
    """
    Check 4 baseline: closure of the learned EFT ratio alone at theta (nu = 0), hat w_b = sum_i w^s_i(theta_ref, 0)
    r_hat(x_i; theta) against w_b = sum_i w^s_i(theta, 0), with Var_MC = sum_i (hat w_i - w_i)^2 (no member scatter).

    Returns:
        (sums, component_bias result with one row)
    """
    index, n_bins = binning['index'], binning['n_bins']
    learned, true = (w0 * np.nan_to_num(r_hat))[keep], (w0 * rho)[keep]
    sums = {
        'bias': (binned_sums(learned[None], index[:, keep], n_bins) - binned_sums(true[None], index[:, keep], n_bins))[
            None
        ],
        'mc_var': binned_sums(((learned - true) ** 2)[None], index[:, keep], n_bins),
        'sm_yield': binned_sums(true[None], index[:, keep], n_bins)[0],
    }
    sums['truth'] = sums['sm_yield']
    return sums, component_bias(sums, binning)


def matched_variance(w_hat, index0, i0, w, index1, i1, n_bins):
    """
    Var_MC of Delta_b = w_hat_b - w_b per bin when the two samples share events: sum_i dw_ib^2, with dw_ib the sum of
    event i's rows in K_b (w_hat_i from the reweighted nominal rows, -w_i from the target rows), the diagonal of
    metrics.matched_closure for every feature of a binning at once. For paired data (i0 = i1, index0 = index1) it is
    sum_i (w_hat_i - w_i)^2 per bin.

    Args:
        w_hat, index0, i0: per nominal row, r_hat w^s(theta, 0), bins [F, N0] and event index i
        w, index1, i1: per target row, w^s(theta, nu), bins [F, N1] and event index i
        n_bins: binning['n_bins']
    Returns:
        [n_bins] Var_MC
    """
    n_features = index0.shape[0]
    events = np.tile(np.concatenate([i0, i1]).astype(np.int64), n_features)
    bins = torch.cat([index0, index1], dim=1).reshape(-1).numpy().astype(np.int64)
    signed = np.tile(np.concatenate([np.asarray(w_hat, np.float64), -np.asarray(w, np.float64)]), n_features)
    # rows of one event in one bin are summed first; bins of different features never coincide
    codes, entry = np.unique(events * n_bins + bins, return_inverse=True)
    dw = np.bincount(entry, weights=signed)
    return np.bincount(codes % n_bins, weights=dw**2, minlength=n_bins)


def nuisance_sums(cache, binning, w0, targets, methods=None, w_hat0=None, offset=None):
    """
    Binned sums for the closure of every nuisance network (nuisance n, direction d) in kinematic bins K_b.

    Args:
        cache: NuisanceCache
        binning: binning dict (quantile_binning, then merge_sparse with w0) on the nominal events of config['data']
        w0: [N] w^s(theta_ref, 0) per nominal event (w^s(theta, 0) in checks 3 and 4)
        targets: callable targets(n, d) (ClosureTargets, ThetaTargets) returning a dict with
            rows0: [N] bool, the nominal events the network is tested on (its eras)
            i0: [N] event index i of each nominal event
            w1, index1, i1: per target row, w^s(theta_ref, nu), bins [F, M] and event index i (weight_shift: the
                nominal events, w0 syst_ratio; kinematic_shift: the shifted events, bin_events of their x')
    Returns:
        list over (n, d), row k = 2 n + d, of dicts for validation.component_bias with
            bias: [S, 1, B] per member e, hat w_b - w_b with hat w_b = sum_i w^s_i(theta_ref, 0) exp f_hat_e(x_i)
            mc_var: [1, B] Var_MC with <f_hat>_e (matched_variance)
            sm_yield: [B] nominal yield sum_i w^s_i(theta_ref, 0) of the tested events (normalises rms_bias)
            truth: [B] target yield w_b
        None for the nuisances whose method is not in methods (default: every method)
        w_hat0: [N] base weight of the learned yield, hat w_b = sum_i w_hat0_i exp f_hat(x_i) (default w0; in check 4
            w^s(theta_ref, 0) r_hat(x_i; theta) with the learned EFT ratio)
        offset: [N] added to every event's learned weight, hat w_b = sum_i (w_hat0_i exp f_hat(x_i) + offset_i)
            (default 0; in the difference test of check 4 w^s(theta, 0) - w_hat0, which removes the closure of the
            learned EFT ratio alone)
    """
    w0 = np.asarray(w0, dtype=np.float64)
    w_hat0 = w0 if w_hat0 is None else np.asarray(w_hat0, dtype=np.float64)
    offset = np.zeros_like(w0) if offset is None else np.asarray(offset, dtype=np.float64)
    index, n_bins = binning['index'], binning['n_bins']
    # <f_hat>_e on every event (nan where some member's f_hat is not finite; those events are left out)
    mean = np.full(cache.f_hat.shape[:2] + cache.finite.shape, np.nan)
    mean[:, :, cache.finite] = cache.f_hat
    out = []
    for n in range(len(cache.nuisances)):
        for d in range(len(DIRECTIONS)):
            if methods is not None and cache.nuisances[n]['method'] not in methods:
                out.append(None)
                continue
            t = targets(n, d)
            rows0 = t['rows0'] & cache.finite
            # a shifted event whose nominal event is left out is left out too; events only in the shifted sample stay
            keep1 = ~np.isin(t['i1'], t['i0'][~rows0])
            truth = binned_sums(t['w1'][None, keep1], t['index1'][:, keep1], n_bins)[0]
            bias = [
                binned_sums(
                    (w_hat0 * np.exp(cache.seed_f_hat[e][n, d]) + offset)[None, rows0], index[:, rows0], n_bins
                )[0]
                - truth
                for e in cache.seeds
            ]
            w_hat = w_hat0[rows0] * np.exp(mean[n, d, rows0]) + offset[rows0]
            mc_var = matched_variance(
                w_hat, index[:, rows0], t['i0'][rows0], t['w1'][keep1], t['index1'][:, keep1], t['i1'][keep1], n_bins
            )
            out.append(
                {
                    'bias': np.stack(bias)[:, None],
                    'mc_var': mc_var[None],
                    'sm_yield': binned_sums(w0[None, rows0], index[:, rows0], n_bins)[0],
                    'truth': truth,
                }
            )
    return out


def pair_sums(cache, binning, w0, targets):
    """
    Binned sums for check 2: closure of the parts of the interpolation, the symmetric part
    f_hat^sym = (f_hat^Up + f_hat^Down) / 2 and the slope f_hat^slope = (f_hat^Up - f_hat^Down) / 2, as
    Delta^sym_b = (Delta^Up_b + Delta^Down_b) / 2 and Delta^slope_b = (Delta^Up_b - Delta^Down_b) / 2 in kinematic
    bins K_b.

    Args:
        cache, binning, w0: as nuisance_sums
        targets: ClosureTargets (its pair(n): Down and Up targets on one event index)
    Returns:
        list over nuisances n of {'fit_symmetry': sums, 'fit_slope': sums}, each a dict for validation.component_bias as nuisance_sums
    """
    w0 = np.asarray(w0, dtype=np.float64)
    index, n_bins = binning['index'], binning['n_bins']
    mean = np.full(cache.f_hat.shape[:2] + cache.finite.shape, np.nan)
    mean[:, :, cache.finite] = cache.f_hat
    out = []
    for n in range(len(cache.nuisances)):
        pair = targets.pair(n)
        rows0 = pair[0]['rows0'] & cache.finite
        # a shifted event whose nominal event is left out is left out too; events only in the shifted samples stay
        keep1 = [~np.isin(t['i1'], t['i0'][~rows0]) for t in pair]
        truth = [binned_sums(t['w1'][None, k], t['index1'][:, k], n_bins)[0] for t, k in zip(pair, keep1)]

        def learned(f_hat):
            return binned_sums((w0 * np.exp(f_hat))[None, rows0], index[:, rows0], n_bins)[0]

        sums = {}
        # the symmetric part adds the directions, the slope subtracts Down from Up
        for part, sign in (('fit_symmetry', 1), ('fit_slope', -1)):
            bias = [
                (learned(cache.seed_f_hat[e][n, 1]) - truth[1] + sign * (learned(cache.seed_f_hat[e][n, 0]) - truth[0]))
                / 2
                for e in cache.seeds
            ]
            w_hat = w0[rows0] * (np.exp(mean[n, 1, rows0]) + sign * np.exp(mean[n, 0, rows0])) / 2
            w = np.concatenate([pair[1]['w1'][keep1[1]], sign * pair[0]['w1'][keep1[0]]]) / 2
            index1 = torch.cat([pair[1]['index1'][:, keep1[1]], pair[0]['index1'][:, keep1[0]]], dim=1)
            i1 = np.concatenate([pair[1]['i1'][keep1[1]], pair[0]['i1'][keep1[0]]])
            sums[part] = {
                'bias': np.stack(bias)[:, None],
                'mc_var': matched_variance(w_hat, index[:, rows0], pair[0]['i0'][rows0], w, index1, i1, n_bins)[None],
                'sm_yield': binned_sums(w0[None, rows0], index[:, rows0], n_bins)[0],
                'truth': (truth[1] + sign * truth[0]) / 2,
            }
        out.append(sums)
    return out


def interpolation_shape(cache, w0, targets, n_sigma=3.0):
    """
    Check 2, shape of the interpolation in nu per nuisance on its eras, from the slope
    f_hat^slope = (f_hat^Up - f_hat^Down) / 2 and the symmetric part f_hat^sym = (f_hat^Up + f_hat^Down) / 2 averaged
    over members, weighted by w^s(theta_ref, 0): the yield-weighted sizes <|f_hat^slope|> and <|f_hat^sym|>, the noise
    on the symmetric part, <sqrt(Var_seeds(f_hat^sym))> with Var_seeds the variance of its member mean per event, and
    the yield fractions of events with a significant symmetric part (|f_hat^sym| > n_sigma sqrt(Var_seeds(f_hat^sym)))
    where log r_hat(x; nu) is not monotonic in nu (|f_hat^slope| < MONOTONIC |f_hat^sym|) and where f_hat^Up and
    f_hat^Down have the same sign (|f_hat^sym| > |f_hat^slope|). Var_seeds holds the scatter between members only, not
    MC noise of the training sample common to every member; with one member no symmetric part counts as significant.

    Args:
        cache: NuisanceCache
        w0: [N] w^s(theta_ref, 0) per nominal event
        targets: ClosureTargets (its rows0(n): the nuisance's eras)
        n_sigma: significance of the symmetric part required to count an event
    Returns:
        list over nuisances n of {'slope_size', 'symmetry_size', 'symmetry_noise', 'non_monotonic', 'same_sign'}
    """
    w0 = np.asarray(w0, dtype=np.float64)
    n_seeds = len(cache.seeds)
    out = []
    for n in range(len(cache.nuisances)):
        rows = targets.rows0(n) & cache.finite
        weight = w0[rows] / w0[rows].sum()
        # [S, 2, N_n]: f_hat^Down, f_hat^Up of every member on the tested events
        members = np.stack([np.asarray(cache.seed_f_hat[e][n], dtype=np.float64)[:, rows] for e in cache.seeds])
        slope = (members[:, 1] - members[:, 0]).mean(0) / 2
        symmetry_members = (members[:, 1] + members[:, 0]) / 2
        symmetry = symmetry_members.mean(0)
        if n_seeds > 1:
            noise = np.sqrt(symmetry_members.var(0, ddof=1) / n_seeds)
        else:
            noise = np.full_like(symmetry, np.inf)
        significant = np.abs(symmetry) > n_sigma * noise
        out.append(
            {
                'slope_size': float(weight @ np.abs(slope)),
                'symmetry_size': float(weight @ np.abs(symmetry)),
                'symmetry_noise': float(weight @ noise) if n_seeds > 1 else None,
                'non_monotonic': float(weight[significant & (np.abs(slope) < MONOTONIC * np.abs(symmetry))].sum()),
                'same_sign': float(weight[significant & (np.abs(symmetry) > np.abs(slope))].sum()),
            }
        )
    return out


def interpolation_yield(cache, w0, targets, nu):
    """
    Check 2, yield of every nuisance on its eras against nu: <exp f(nu)> weighted by w^s(theta_ref, 0), with f(nu) the
    interpolation of <f_hat>_e (nuisanceCache.interpolate).

    Args:
        cache: NuisanceCache
        w0: [N] w^s(theta_ref, 0) per nominal event
        targets: ClosureTargets (its rows0(n): the nuisance's eras)
        nu: [G] values of nu
    Returns:
        [n_n, G] yield relative to nu = 0
    """
    w0 = np.asarray(w0, dtype=np.float64)[cache.finite]
    out = np.empty((len(cache.nuisances), len(nu)))
    for n in range(len(cache.nuisances)):
        rows = targets.rows0(n)[cache.finite]
        weight = w0[rows] / w0[rows].sum()
        f_hat = np.asarray(cache.f_hat[n], dtype=np.float64)[None, :, rows]
        out[n] = [weight @ np.exp(interpolate(f_hat, [value])) for value in nu]
    return out


def test_arrays(sums, biases, n_bins):
    """
    Per-bin arrays of one closure test over its rows (networks, nuisances or one row), nan for rows not tested.

    Args:
        sums: list of nuisance_sums / pair_sums / eft_closure sums, None for rows not tested
        biases: list of their component_bias results, None for rows not tested
        n_bins: binning['n_bins']
    Returns:
        dict of [rows, B] arrays D (Delta_b), s2 (Var_seeds), mc_var (Var_MC), truth (w_b), sm_yield
    """
    out = {key: np.full((len(sums), n_bins), np.nan) for key in ('D', 's2', 'mc_var', 'truth', 'sm_yield')}
    for row, (s, bias) in enumerate(zip(sums, biases)):
        if s is None:
            continue
        out['D'][row], out['s2'][row] = bias['D'][0], bias['s2'][0]
        out['mc_var'][row], out['truth'][row], out['sm_yield'][row] = s['mc_var'][0], s['truth'], s['sm_yield']
    return out


def write_sums(name, cache, binning, tests, nu, yields):
    """
    Write nuisance_sums.npz to name, the per-bin arrays behind nuisance_diagnostics.yml. Every key is listed in
    docs/validation.md:
        nuisances, methods: [n_n] names and methods; network tests have row k = 2 n + d
        binning/names, binning/feature, binning/discrete, binning/edges/<feature>: the binning K_b
        <test>/D, s2, mc_var, truth, sm_yield: [rows, B] (test_arrays) for the tests closure, fit_symmetry,
            fit_slope, factorisation/<label>, combined/<label>, difference/<label>, eft_only/<label>
        interpolation/nu [G], interpolation/yield [n_n, G] (interpolation_yield)

    Args:
        name: output directory
        cache: NuisanceCache
        binning: the binning of the closure
        tests: {test: (list of sums, list of component_bias results)}
        nu, yields: grid and output of interpolation_yield
    """
    arrays = {
        'nuisances': np.array([nuisance['name'] for nuisance in cache.nuisances]),
        'methods': np.array([nuisance['method'] for nuisance in cache.nuisances]),
        'binning/names': np.array(binning['names']),
        'binning/feature': np.asarray(binning['feature']),
        'binning/discrete': np.asarray(binning['discrete']),
        'interpolation/nu': np.asarray(nu),
        'interpolation/yield': yields,
    }
    for feature, edges in zip(binning['names'], binning['edges']):
        arrays[f'binning/edges/{feature}'] = np.asarray(edges)
    for test, (sums, biases) in tests.items():
        for key, value in test_arrays(sums, biases, binning['n_bins']).items():
            arrays[f'{test}/{key}'] = value
    np.savez(f'{name}/nuisance_sums.npz', **arrays)


def closure_entry(bias, features, z_threshold):
    """Summary of one closure test (a component_bias result with one row): flagged, max_Z and its feature, rms_bias, Z."""
    z, rms_bias = bias['Z'][0], bias['rms_bias'][0]
    return {
        'flagged': bool(np.nanmax(z) > z_threshold),
        'max_Z': float(np.nanmax(z)),
        'worst_feature': features[int(np.nanargmax(z))],
        'rms_bias': float(np.nanmax(rms_bias)),
        'Z': {f: float(v) for f, v in zip(features, z)},
    }


def write_results(
    name, cache, binning, closure, parts, shape, factorisation, combined, difference, eft_only, z_threshold
):
    """
    Write nuisance_diagnostics.yml to name: per nuisance its method and eras, per direction the closure of its
    network averaged over members (check 1), the closure of the symmetric part and the slope of the interpolation and
    its shape in nu (check 2), in every network input. Every key is listed in docs/validation.md.

    Args:
        name: output directory
        cache: NuisanceCache
        binning: the binning of the closure
        closure: component_bias result of every network, row k = 2 n + d
        parts: per nuisance n, {'fit_symmetry': component_bias result, 'fit_slope': component_bias result}
        shape: per nuisance n, the sizes and fractions of interpolation_shape
        factorisation: {label: component_bias result of every network at that WC point, row k = 2 n + d, None for
            the networks not tested} (check 3)
        combined: as factorisation, with the learned EFT ratio of the ensemble cache (check 4)
        difference: as combined, the nuisance shift alone on top of the learned EFT ratio (check 4, difference test)
        eft_only: {label: component_bias result of the learned EFT ratio alone at that point} (check 4)
        z_threshold: a test is flagged when its Z exceeds this in some feature
    """
    features = binning['names']
    nuisances = {}
    for n, nuisance in enumerate(cache.nuisances):
        entry = {'method': nuisance['method'], 'eras': nuisance.get('eras')}
        for d, direction in enumerate(DIRECTIONS):
            entry[direction] = closure_entry(closure[len(DIRECTIONS) * n + d], features, z_threshold)
        for part in ('fit_symmetry', 'fit_slope'):
            entry[part] = closure_entry(parts[n][part], features, z_threshold)
        entry['shape'] = shape[n]
        entry['factorisation'] = {}
        for label, results in factorisation.items():
            if results[len(DIRECTIONS) * n] is None:
                entry['factorisation'][label] = None
                continue
            entry['factorisation'][label] = {}
            for d, direction in enumerate(DIRECTIONS):
                test = closure_entry(results[len(DIRECTIONS) * n + d], features, z_threshold)
                test['rms_bias_change'] = test['rms_bias'] - entry[direction]['rms_bias']
                entry['factorisation'][label][direction] = test
        entry['combined'] = {}
        for label, results in combined.items():
            if results[len(DIRECTIONS) * n] is None:
                entry['combined'][label] = None
                continue
            entry['combined'][label] = {}
            for d, direction in enumerate(DIRECTIONS):
                test = closure_entry(results[len(DIRECTIONS) * n + d], features, z_threshold)
                test['rms_bias_change'] = test['rms_bias'] - entry['factorisation'][label][direction]['rms_bias']
                entry['combined'][label][direction] = test
        entry['difference'] = {}
        for label, results in difference.items():
            if results[len(DIRECTIONS) * n] is None:
                entry['difference'][label] = None
                continue
            entry['difference'][label] = {}
            for d, direction in enumerate(DIRECTIONS):
                test = closure_entry(results[len(DIRECTIONS) * n + d], features, z_threshold)
                test['rms_bias_change'] = test['rms_bias'] - entry['factorisation'][label][direction]['rms_bias']
                entry['difference'][label][direction] = test
        nuisances[nuisance['name']] = entry
    results = {
        'cache': f'{name}/nuisance',
        'theta_ref': cache.theta_ref,
        'seeds': cache.seeds,
        'closure': {'z_threshold': z_threshold, 'features': features, 'n_bins': int(binning['n_bins'])},
        'factorisation_points': list(factorisation),
        'eft_only': {label: closure_entry(bias, features, z_threshold) for label, bias in eft_only.items()},
        'nuisances': nuisances,
    }
    with open(f'{name}/nuisance_diagnostics.yml', 'w') as f:
        yaml.safe_dump(results, f, sort_keys=False)


def plot_results(name, feature_config=None, n_sigma=3.0):
    """
    Plots of the validation from nuisance_diagnostics.yml and nuisance_sums.npz under name, written to name/plots
    (replaced): Z of every network and of the parts of the interpolation in every network input (z_heatmap_networks,
    z_heatmap_parts), rms bias against Z (size_vs_significance), the closure of every network in its worst feature
    (closure/{flagged,passed}) and of the flagged parts (closure_parts), the yield against nu (interpolation_yield),
    the shape metrics (interpolation_shape), and per factorisation point the rms bias of check 3 against check 1
    (factorisation_<label>) and of check 4 against check 3 (eft_difference_<label>, eft_combined_<label>). Nuisances
    are ordered by method, then by name. Every file is listed in docs/validation.md.

    Args:
        name: output directory of the validation
        feature_config: contents of the features yml, for the TeX labels of the features
        n_sigma: significance of the symmetric part in the shape metrics (config shape.n_sigma), for the label
    """
    with open(f'{name}/nuisance_diagnostics.yml') as f:
        diag = yaml.safe_load(f)
    sums = np.load(f'{name}/nuisance_sums.npz')
    entries, z_threshold = diag['nuisances'], diag['closure']['z_threshold']
    features = [str(f) for f in sums['binning/names']]
    binning = {
        'names': features,
        'feature': sums['binning/feature'],
        'edges': [sums[f'binning/edges/{f}'] for f in features],
    }
    nuisances, methods = [str(n) for n in sums['nuisances']], [str(m) for m in sums['methods']]
    # by method (weight_shift first), then by name; rows 2 n + d of the network tests follow n
    order = sorted(range(len(nuisances)), key=lambda n: (list(METHOD_NAMES).index(methods[n]), nuisances[n]))
    names = [nuisances[n] for n in order]
    plots = f'{name}/plots'
    shutil.rmtree(plots, ignore_errors=True)
    for sub in ('closure/flagged', 'closure/passed', 'closure_parts/flagged'):
        os.makedirs(f'{plots}/{sub}', mode=0o755)

    def test(key):
        return {k: sums[f'{key}/{k}'] for k in ('D', 's2', 'mc_var', 'truth', 'sm_yield')}

    # plot 1: Z of every network (Down beside Up) and of the parts, grouped by method / part, in the shared order
    labels = [f'{n} {DIRECTION_TEX[d]}' for n in names for d in DIRECTIONS]
    Z = np.array([[entries[n][d]['Z'][f] for f in features] for n in names for d in DIRECTIONS])
    kinds = [METHOD_NAMES[entries[n]['method']] for n in names for d in DIRECTIONS]
    plot_z_heatmap(
        Z,
        labels,
        kinds,
        features,
        f'{plots}/z_heatmap_networks.png',
        z_threshold,
        feature_config=feature_config,
        kind_order=list(METHOD_NAMES.values()),
        tick_labels=labels,
        sort=False,
    )
    Z = np.array([[entries[n][p]['Z'][f] for f in features] for p in PARTS for n in names])
    kinds = [PARTS[p] for p in PARTS for n in names]
    labels = names * len(PARTS)
    plot_z_heatmap(
        Z,
        labels,
        kinds,
        features,
        f'{plots}/z_heatmap_parts.png',
        z_threshold,
        feature_config=feature_config,
        kind_order=list(PARTS.values()),
        tick_labels=labels,
        sort=False,
    )

    # plot 2: rms bias against Z of the networks and of the parts
    def size(key):
        return {n: (entries[n][key]['max_Z'], entries[n][key]['rms_bias']) for n in names}

    panels = {'networks': {d: size(d) for d in DIRECTIONS}}
    panels.update({PARTS[p]: {'part': size(p)} for p in PARTS})
    reference = None
    if 'sm' in diag['eft_only']:
        reference = (r'EFT ensemble at $\theta_{\mathrm{SM}}$', diag['eft_only']['sm']['rms_bias'])
    plot_size_significance(
        panels,
        {n: entries[n]['method'] for n in names},
        z_threshold,
        f'{plots}/size_vs_significance.png',
        reference=reference,
    )

    # plot 3: closure of every network and of the flagged parts in their worst feature
    closure = test('closure')
    for n in order:
        entry = entries[nuisances[n]]
        for d, direction in enumerate(DIRECTIONS):
            result = entry[direction]
            group = 'flagged' if result['flagged'] else 'passed'
            plot_nuisance_closure(
                closure,
                len(DIRECTIONS) * n + d,
                features.index(result['worst_feature']),
                result['max_Z'],
                binning,
                f'{plots}/closure/{group}/{nuisances[n]}_{direction}.png',
                f'{nuisances[n]} {DIRECTION_TEX[direction]} ({METHOD_NAMES[entry["method"]]})',
                feature_config=feature_config,
            )
    for p in PARTS:
        part = test(p)
        for n in order:
            result = entries[nuisances[n]][p]
            if result['flagged']:
                plot_nuisance_closure(
                    part,
                    n,
                    features.index(result['worst_feature']),
                    result['max_Z'],
                    binning,
                    f'{plots}/closure_parts/flagged/{nuisances[n]}_{p}.png',
                    f'{nuisances[n]} {PARTS[p]}',
                    nominal=False,
                    feature_config=feature_config,
                )

    # plot 4: yield against nu, with the true yields at nu = -1, +1 from the closure of the networks (one feature
    # counts every tested event once)
    first = binning['feature'] == 0
    sm = closure['sm_yield'][:, first].sum(1)
    truth = (closure['truth'][:, first].sum(1) / sm).reshape(-1, len(DIRECTIONS))
    truth_err = (np.sqrt(closure['mc_var'][:, first].sum(1)) / sm).reshape(-1, len(DIRECTIONS))
    plot_interpolation_yield(
        sums['interpolation/nu'],
        sums['interpolation/yield'][order],
        truth[order],
        truth_err[order],
        names,
        [methods[n] for n in order],
        f'{plots}/interpolation_yield.png',
    )

    # plot 5: shape metrics of the interpolation
    plot_interpolation_shape(
        [entries[n]['shape'] for n in names],
        names,
        [methods[n] for n in order],
        f'{plots}/interpolation_shape.png',
        n_sigma,
    )

    # plots 6, 7: rms bias of check 3 against check 1, and of check 4 against check 3, per factorisation point
    for label in diag['factorisation_points']:
        tested = [n for n in names if entries[n]['factorisation'][label] is not None]
        theta = r'$\theta_{\mathrm{SM}}$' if label == 'sm' else label

        def rms(key):
            return np.array(
                [
                    [(entries[n] if key is None else entries[n][key][label])[d]['rms_bias'] for d in DIRECTIONS]
                    for n in tested
                ]
            )

        def flagged(key):
            return np.array([[entries[n][key][label][d]['flagged'] for d in DIRECTIONS] for n in tested])

        plot_rms_comparison(
            tested,
            rms(None),
            rms('factorisation'),
            flagged('factorisation'),
            (r'check 1 at $\theta_{\mathrm{ref}}$', f'check 3 at {theta}'),
            f'{plots}/factorisation_{label}.png',
        )
        plot_rms_comparison(
            tested,
            rms('factorisation'),
            rms('difference'),
            flagged('difference'),
            (f'check 3 at {theta}', f'difference test at {theta}'),
            f'{plots}/eft_difference_{label}.png',
        )
        plot_rms_comparison(
            tested,
            rms('factorisation'),
            rms('combined'),
            flagged('combined'),
            (f'check 3 at {theta}', f'combined at {theta}'),
            f'{plots}/eft_combined_{label}.png',
            reference=(f'EFT ensemble alone at {theta}', diag['eft_only'][label]['rms_bias']),
        )


def main(config_path, refit=False):
    """
    Run the validation of the nuisance ensemble (docs/validation.md): check 1, the closure of every nuisance network
    averaged over members in every network input; check 2, the closure of the symmetric part and the slope of the
    interpolation in nu and its shape; check 3, the closure of the weight_shift networks at other WC points
    (factorisation in theta); check 4, the closure of the combined learned ratio r_hat(x; theta) r_hat(x; nu) of the
    scans at those points, and of the nuisance shift alone on top of the learned EFT ratio; writes
    nuisance_diagnostics.yml, nuisance_sums.npz and plots/ (plot_results) under config['name'].

    Args:
        config_path: nuisance validation config yml
        refit: build the nuisance cache and replace it
    """
    config = load_config(config_path)
    torch.set_num_threads(config.get('n_threads', 1))
    name = config['name']
    os.makedirs(name, mode=0o755, exist_ok=True)
    if 'lumi' not in config:
        sys.exit(f'[error]: {config["ensemble"]} needs lumi, the luminosity of each era')

    nominal = load_combined(config['data'], 'nominal', ('features', 'ids', 'event_id', 'syst_ratio', 'w_ref'))
    features = nominal[0]
    w0 = sample_weight(nominal, 'nominal', config).double()
    cache = load_nuisance_ensemble(config, features, refit)

    # check 1: closure of every network, averaged over members, in merged equal-count bins of every network input
    cfg = config.get('closure', {})
    binning = quantile_binning(features, config['features'], cfg.get('n_bins', 20), cfg.get('features'))
    binning = merge_sparse(binning, features, config['features'], w0, cfg.get('min_neff', 30))
    targets = ClosureTargets(config, cache, nominal, w0, binning)
    closure_sums = nuisance_sums(cache, binning, w0, targets)
    closure = [component_bias(s, binning) for s in closure_sums]
    tests = {'closure': (closure_sums, closure)}

    # check 2: closure of the symmetric part and the slope of the interpolation, and its shape in nu
    pairs = pair_sums(cache, binning, w0, targets)
    parts = [{part: component_bias(s, binning) for part, s in pair.items()} for pair in pairs]
    for part in ('fit_symmetry', 'fit_slope'):
        tests[part] = ([pair[part] for pair in pairs], [p[part] for p in parts])
    shape = interpolation_shape(cache, w0, targets, config.get('shape', {}).get('n_sigma', 3.0))
    nu = np.linspace(-2, 2, 41)
    yields = interpolation_yield(cache, w0, targets, nu)

    # check 3: closure at other WC points theta with the networks trained at theta_ref (factorisation in theta), in
    # the bins of theta_ref; kinematic_shift is not tested, its variation files have no structure constants
    points = config.get('factorisation', {}).get('points', {'sm': [1.0] + [0.0] * len(config['wcs'])})
    coefs = load_combined(config['data'], 'nominal', ('coefs',))[1]
    factorisation = {}
    weights = factorisation_weights(coefs, cache.theta_ref, points)
    del coefs
    for label, rho in weights.items():
        sums = nuisance_sums(cache, binning, w0.numpy() * rho, ThetaTargets(targets, rho), methods=('weight_shift',))
        factorisation[label] = [None if s is None else component_bias(s, binning) for s in sums]
        tests[f'factorisation/{label}'] = (sums, factorisation[label])

    # check 4: the combined learned ratio of the scans, r_hat(x; theta) of the EFT ensemble times r_hat(x; nu) at
    # nu_n = +-1, at the points of check 3 (weight_shift only); the EFT-only closure at nu = 0 for comparison
    eft = EnsembleCache(config['eft_cache'])
    if not eft.matches(config['eft_config']):
        sys.exit(f'[error]: {config["eft_cache"]} is stale; rerun the EFT validation with --refit')
    combined, difference, eft_only = {}, {}, {}
    for label, rho in weights.items():
        r_hat = eft_ratio(eft, points[label])
        keep = np.isfinite(r_hat)
        sums = nuisance_sums(
            cache,
            binning,
            w0.numpy() * rho,
            ThetaTargets(targets, rho, keep),
            methods=('weight_shift',),
            w_hat0=w0.numpy() * np.nan_to_num(r_hat),
        )
        combined[label] = [None if s is None else component_bias(s, binning) for s in sums]
        tests[f'combined/{label}'] = (sums, combined[label])
        # difference test: the nuisance shift on top of the learned EFT ratio, w0 r_hat (e^f_hat - 1) against
        # w^s(theta, nu) - w^s(theta, 0) per event, so the closure and the scatter of r_hat(x; theta) cancel
        w_hat0 = w0.numpy() * np.nan_to_num(r_hat)
        sums = nuisance_sums(
            cache,
            binning,
            w0.numpy() * rho,
            ThetaTargets(targets, rho, keep),
            methods=('weight_shift',),
            w_hat0=w_hat0,
            offset=w0.numpy() * rho - w_hat0,
        )
        difference[label] = [None if s is None else component_bias(s, binning) for s in sums]
        tests[f'difference/{label}'] = (sums, difference[label])
        sums, eft_only[label] = eft_closure(binning, w0.numpy(), rho, r_hat, keep)
        tests[f'eft_only/{label}'] = ([sums], [eft_only[label]])

    z_threshold = cfg.get('z_threshold', 5.0)
    write_results(
        name, cache, binning, closure, parts, shape, factorisation, combined, difference, eft_only, z_threshold
    )
    write_sums(name, cache, binning, tests, nu, yields)
    plot_results(name, config['features'], config.get('shape', {}).get('n_sigma', 3.0))
    n_flagged = sum(bool(np.nanmax(bias['Z']) > z_threshold) for bias in closure)
    print(f'[info]: {n_flagged} of {len(closure)} nuisance networks flagged; see {name}/nuisance_diagnostics.yml')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='nuisance validation configuration yml file')
    parser.add_argument('--refit', action='store_true', help='build the nuisance cache and replace it')
    parser.add_argument(
        '--plots-only',
        action='store_true',
        help='redraw the plots from nuisance_diagnostics.yml and nuisance_sums.npz, without the checks',
    )
    args = parser.parse_args()
    if args.plots_only:
        config = load_config(args.config)
        plot_results(config['name'], config['features'], config.get('shape', {}).get('n_sigma', 3.0))
    else:
        main(args.config, args.refit)
