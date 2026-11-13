from topsbi.tools.build_ratio import seed_map
from topsbi.tools.data import (
    NUISANCE_COLUMNS,
    event_keys,
    expansion,
    load_combined,
    matched_events,
    sample_weight,
    syst_ratio_column,
)
from topsbi.tools.ensembleCache import EnsembleCache, bin_events, binned_sums, quantile_binning
from topsbi.tools.metrics import merge_sparse_bins
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
    plot_interpolation_yield,
    plot_nuisance_closure,
    plot_nuisance_misclosure,
    plot_rms_comparison,
    plot_size_significance,
    plot_yield_summary,
    plot_z_heatmap,
)
from topsbi.validation import component_bias, era_names

import numpy as np

import argparse, os, shutil, sys, torch, yaml

# taken from the EFT validation config, so both ensembles see the same events, reference point, luminosities and
# bias-test settings (bins, min_neff, z_threshold)
SHARED_KEYS = ('data', 'c0', 'wcs', 'lumi', 'eras', 'features', 'year_loc', 'bias')
# the parts of the interpolation in nu whose closure is tested, with their TeX labels
PARTS = {'fit_slope': r'$\hat{f}\,^{\mathrm{slope}}$', 'fit_symmetry': r'$\hat{f}\,^{\mathrm{sym}}$'}
# legend text of the closure plots of the parts, (truth, learned): the yield equivalents of Combine's
# (delta^+ - delta^-) / 2 and (delta^+ + delta^-) / 2, with w_b^0 the nominal and w_b^+- the Up / Down yields
PART_LABELS = {
    'fit_slope': (r'$(w_b^{+} - w_b^{-})\,/\,2$', r'$(\hat{w}_b^{+} - \hat{w}_b^{-})\,/\,2$'),
    'fit_symmetry': (r'$(w_b^{+} + w_b^{-})\,/\,2 - w_b^{0}$', r'$(\hat{w}_b^{+} + \hat{w}_b^{-})\,/\,2 - w_b^{0}$'),
}
# largest number of columns of the compact Z heatmaps
COMPACT_COLUMNS = 30
# sign of each direction in the legend text of the network closure plots
DIRECTION_SIGN = {'down': '-', 'up': '+'}


def load_config(config_path):
    """
    The nuisance validation config, completed from the EFT validation config it names (ensemble): data, c0, wcs,
    lumi, features, year_loc and the bias-test settings bias (n_bins, min_neff, z_threshold, features) are taken from
    there (a nuisance config that sets one of them stops the run), eft_cache is the EFT ensemble cache,
    {name}/ensemble of that config, and eft_config that config itself.

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


def check_reference(config, theta_ref):
    """
    Stop the run unless the nuisance networks were trained at the WC point of the EFT reference: theta_ref must equal
    the SM-inclusive WC part of c0, its first len(wcs) + 1 entries (c0 continues with the background strengths
    mu_bg, which the nuisance networks do not have).

    Args:
        config: nuisance validation config, completed by load_config
        theta_ref: the nuisance networks' reference point (SM-inclusive)
    """
    point = config['c0'][: len(config['wcs']) + 1]
    if len(theta_ref) != len(point) or not np.allclose(theta_ref, point, rtol=0, atol=1e-12):
        sys.exit(
            f'[error]: the nuisance networks were trained at theta_ref = {list(theta_ref)}, the EFT ensemble at the WC '
            f'point {point} of c0 = {config["c0"]}; both must share the reference point'
        )


def load_nuisance_ensemble(config, features, refit=False):
    """
    The nuisance cache {name}/nuisance of config, on every event of config['data'] (signal and background): read when
    it matches config (EFT ensemble, data, networks and their model.pt), otherwise built and written, but only when
    refit is set. The networks' reference point theta_ref must be the WC point of c0 (check_reference).

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
        check_reference(config, cache.theta_ref)
        return cache
    nuisances, networks = nuisance_networks(config['seeds'])
    theta_ref = reference_point(nuisances, networks)
    check_reference(config, theta_ref)
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
    return NuisanceCache(path)


def event_terms(coefs, n_wcs, background_processes):
    """
    Term of every event of the data, the process its weight belongs to: the background process p for an event whose
    column of p (after T(theta) in expansion) is non-zero, otherwise 'signal'.

    Args:
        coefs: [N, n_q + K] structure constants of the data, the columns of T(theta, mu)
        n_wcs: number of WCs n
        background_processes: names of the K background processes, in the order of their columns
    Returns:
        [N] numpy array of str, 'signal' or the background process of each event
    """
    n_q = (n_wcs + 1) * (n_wcs + 2) // 2
    if coefs.shape[1] != n_q + len(background_processes):
        raise ValueError(
            f'{n_wcs} wcs and {len(background_processes)} background processes give '
            f'{n_q + len(background_processes)} columns but coefs has {coefs.shape[1]}'
        )
    terms = np.full(coefs.shape[0], 'signal', dtype=object)
    for p, process in enumerate(background_processes):
        rows = (coefs[:, n_q + p] != 0).numpy()
        if (terms[rows] != 'signal').any():
            raise ValueError(f'events of {process} have non-zero columns of another background process too')
        terms[rows] = process
    return terms.astype(str)


def closure_binning(features, feature_config, w0, terms, nbins=20, names=None, min_neff=30):
    """
    Merged equal-count bins K_b of the network inputs, shared by every term: the bins of the EFT bias test
    (quantile_binning, then merged until every event together holds at least min_neff effective events per bin), merged
    further until the events of every term also hold at least min_neff effective events in every bin
    (metrics.merge_sparse_bins with every event as the first sample and one sample per term).

    Args:
        features: [N, n_features] raw network inputs of every event
        feature_config: contents of the features yml
        w0: [N] event weights the tests sum (w^s(theta_ref, 0))
        terms: [N] term of every event (event_terms)
        nbins: quantile bins per feature before merging
        names: features to bin, default every feature in feature_config
        min_neff: minimum effective number of events per merged bin, of every event together and of each term
    Returns:
        binning dict (as quantile_binning), its index over every event
    """
    w0 = np.asarray(w0, dtype=np.float64)
    terms = np.asarray(terms)
    binning = quantile_binning(features, feature_config, nbins, names)
    groups = [terms == term for term in np.unique(terms)]
    edges, index, feature = [], [], []
    offset = 0
    for f, name in enumerate(binning['names']):
        x = features[:, feature_config[name]['loc']].double().numpy()
        others = [(x[rows], w0[rows]) for rows in groups]
        e = merge_sparse_bins(x, w0, binning['edges'][f], min_neff, others)
        k = np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(e) - 2)
        edges.append(e)
        index.append(k + offset)
        feature += [f] * (len(e) - 1)
        offset += len(e) - 1
    return {
        'names': binning['names'],
        'edges': edges,
        'discrete': binning['discrete'],
        'feature': np.array(feature),
        'index': torch.from_numpy(np.stack(index)),
        'n_bins': offset,
    }


class ClosureTargets:
    """
    Target sample of every nuisance network, on the events of its term in config['data'] (the signal, or one
    background process). weight_shift: those events with w^s(theta_ref, nu) = w^s(theta_ref, 0) r(x, z; theta, nu)
    (syst_ratio). kinematic_shift: the events of the variation's file in the network's training data,
    <data>/<systematic><Up|Down>/<file>, with w^s(theta_ref, nu), binned by their own x' in the bins K_b and matched
    to the term's events by event_keys.
    """

    def __init__(self, config, cache, nominal, w0, binning, terms):
        """
        Args:
            config: nuisance validation config, completed by load_config (data, lumi, features, year_loc)
            cache: NuisanceCache (nuisances with method, systematic, term, data and eras)
            nominal: tensors of config['data'] (load_combined with features, ids, event_id, syst_ratio, w_ref)
            w0: [N] w^s(theta_ref, 0) of the events of config['data']
            binning: binning of the events of config['data'] (closure_binning)
            terms: [N] term of every event (event_terms)
        """
        self.config, self.cache, self.binning = config, cache, binning
        # the syst_ratio columns of config['data'] are listed in <root>/nominal/systematics.yml
        self.root = os.path.dirname(os.path.dirname(config['data']))
        self.file = os.path.basename(config['data'])
        self.year_loc = config.get('year_loc', 21)
        self.w0 = np.asarray(w0, dtype=np.float64)
        self.terms = np.asarray(terms)
        missing = sorted({nuisance['term'] for nuisance in cache.nuisances} - set(self.terms))
        if missing:
            raise ValueError(f'{config["data"]} has no events of the nuisance terms {missing}')
        columns = dict(zip(NUISANCE_COLUMNS['nominal'], nominal))
        self.syst_ratio = columns['syst_ratio']
        self.event_eras = columns['features'][:, self.year_loc].to(torch.int64).numpy()
        self.keys0 = event_keys(nominal, 'nominal', self.year_loc)

    def __call__(self, n, d):
        """
        Target of nuisance n, direction d (0 = Down, 1 = Up), as nuisance_sums reads it: rows0, i0, w1, index1, i1.
        """
        nuisance = self.cache.nuisances[n]
        rows0 = self.rows0(n)
        if nuisance['method'] == 'weight_shift':
            events = np.arange(len(self.w0))
            w1 = self.w0 * self.true_ratio(n, d)
            return {'rows0': rows0, 'i0': events, 'w1': w1, 'index1': self.binning['index'], 'i1': events}
        shifted = load_combined(self._path(n, d), 'variation', ('features', 'ids', 'event_id', 'w_ref'))
        term = np.flatnonzero(self.terms == nuisance['term'])
        i0_term, i1 = matched_events(self.keys0[term], event_keys(shifted, 'variation', self.year_loc))
        # the events of other terms are never matched: index -1, which no shifted event has
        i0 = np.full(len(self.w0), -1, dtype=np.int64)
        i0[term] = i0_term.numpy()
        return {
            'rows0': rows0,
            'i0': i0,
            'w1': sample_weight(shifted, 'variation', self.config).double().numpy(),
            'index1': bin_events(self.binning, shifted[0], self.config['features']),
            'i1': i1.numpy(),
        }

    def _path(self, n, d):
        """File of the shifted events of kinematic nuisance n, direction d, in the training data of its networks."""
        nuisance = self.cache.nuisances[n]
        return f'{nuisance["data"]}/{nuisance["systematic"]}{DIRECTIONS[d].capitalize()}/{self.file}'

    def rows0(self, n):
        """[N] bool, the events nuisance n is tested on: the events of its term, in its eras."""
        return (self.terms == self.cache.nuisances[n]['term']) & self.era_rows(n)

    def era_rows(self, n):
        """[N] bool, the events in the eras of nuisance n, of every term."""
        eras = self.cache.nuisances[n].get('eras')
        return np.ones(len(self.w0), dtype=bool) if eras is None else np.isin(self.event_eras, eras)

    def true_ratio(self, n, d):
        """
        [N] r(x, z; theta, nu) of every event for weight_shift nuisance n, direction d: syst_ratio on the events of its
        term, 1 on the events of the other terms, which the nuisance does not shift.
        """
        nuisance = self.cache.nuisances[n]
        k, j = syst_ratio_column(self.root, nuisance['systematic'], DIRECTIONS[d])
        r = self.syst_ratio[:, k, j].double().numpy()
        return np.where(self.terms == nuisance['term'], r, 1.0)

    def pair(self, n):
        """
        Targets of nuisance n for Down and Up (as __call__), with one event index i over the events of its term and the
        Down and Up samples, so an event in several of them is one unit in matched_variance.
        """
        down, up = self(n, 0), self(n, 1)
        if self.cache.nuisances[n]['method'] == 'weight_shift':
            return down, up
        term = np.flatnonzero(self.terms == self.cache.nuisances[n]['term'])
        keys = [self.keys0[term]] + [
            event_keys(
                load_combined(self._path(n, d), 'variation', ('features', 'ids', 'event_id')),
                'variation',
                self.year_loc,
            )
            for d in range(len(DIRECTIONS))
        ]
        _, events = np.unique(np.concatenate(keys), axis=0, return_inverse=True)
        events = events.reshape(-1)
        n0, n_down = len(term), len(keys[1])
        i0 = np.full(len(self.w0), -1, dtype=np.int64)
        i0[term] = events[:n0]
        down['i0'] = up['i0'] = i0
        down['i1'], up['i1'] = events[n0 : n0 + n_down], events[n0 + n_down :]
        return down, up


class ThetaTargets:
    """
    Targets of ClosureTargets moved to the WC point theta, for weight_shift nuisances: each weight scaled by
    rho_i(theta) = s_q(z_i) . T(theta, mu) / s_q(z_i) . T(theta_ref, mu) of its event.
    """

    def __init__(self, targets, rho, rows=None, every_term=False):
        """
        Args:
            targets: ClosureTargets
            rho: [N] rho_i(theta) of every event (factorisation_weights)
            rows: [N] bool, events to keep, e.g. those the EFT ensemble keeps; default every event
            every_term: test on the events of every term in the nuisance's eras, not only on those of its own term,
                for a model in which the other terms add to the same bins
        """
        self.targets, self.rho, self.rows, self.every_term = targets, rho, rows, every_term

    def __call__(self, n, d):
        t = dict(self.targets(n, d))
        # weight_shift: the target events are the events of the data (i1 = i0), so rho applies event by event
        t['w1'] = t['w1'] * self.rho
        if self.every_term:
            t['rows0'] = self.targets.era_rows(n)
        if self.rows is not None:
            t['rows0'] = t['rows0'] & self.rows
        return t


def factorisation_weights(coefs, theta_ref, points, n_backgrounds=0):
    """
    rho_i(theta) = s_q(z_i) . T(theta, mu) / s_q(z_i) . T(theta_ref, mu) per event for every WC point theta, at the
    nominal background strengths mu_p = 1: the change of the signal weight from theta_ref to theta on the signal
    events, 1 on the background events.

    Args:
        coefs: [N, n_q + K] structure constants s_q(z) of the events, the columns of T(theta, mu)
        theta_ref: the networks' reference point (SM-inclusive)
        points: {label: theta} WC points (SM-inclusive)
        n_backgrounds: number K of background processes, the columns after T(theta)
    Returns:
        {label: [N] rho(theta), float64}
    """
    coefs = coefs.to(torch.float64)
    n_wcs = len(theta_ref) - 1

    def weight(theta):
        return coefs @ expansion(list(theta) + [1.0] * n_backgrounds, n_wcs, dtype=torch.float64)

    reference = weight(theta_ref)
    return {label: (weight(theta) / reference).numpy() for label, theta in points.items()}


def eft_terms(eft, theta):
    """
    The terms of the learned EFT ratio at theta and the nominal background strengths mu_p = 1 (expansion),
    r_hat(x; theta, mu) = T(theta)[mask] <Gamma_sig>_e + sum_p mu_p <Gamma_p>_e, on every event of the EFT cache's
    data, nan where the ensemble drops the event.

    Args:
        eft: EnsembleCache
        theta: WC point (SM-inclusive)
    Returns:
        {'signal': [N] T(theta) Gamma_sig, p: [N] mu_p Gamma_p for every background process p}, float64
    """
    n_backgrounds = len(eft.background_processes)
    point = expansion(list(theta) + [1.0] * n_backgrounds, eft.n_wcs, dtype=torch.float64).numpy()
    n_theta = len(point) - n_backgrounds
    if not eft.mask[n_theta:].all():
        raise ValueError(f'{eft.path} drops a background column; the background term of its ratio is not separable')
    # rows of Gamma: the kept columns of T(theta) first, then one row per background process
    n_signal = int(eft.mask[:n_theta].sum())
    parts = {'signal': point[:n_theta][eft.mask[:n_theta]] @ eft.Gamma[:n_signal]}
    for p, process in enumerate(eft.background_processes):
        parts[process] = point[n_theta + p] * np.asarray(eft.Gamma[n_signal + p], dtype=np.float64)
    terms = {}
    for term, part in parts.items():
        terms[term] = np.full(eft.finite.shape, np.nan)
        terms[term][eft.finite] = part
    return terms


def eft_member_ratios(eft, theta, keep):
    """
    [S, N] learned EFT ratio r_hat_e(x; theta, mu) = T(theta, mu)[mask] Gamma_e of every member e of the EFT cache, at
    the nominal background strengths mu_p = 1 (expansion), on every event of its data, 0 outside keep.

    Args:
        eft: EnsembleCache
        theta: WC point (SM-inclusive)
        keep: [N] bool, the events every member keeps (eft.finite)
    """
    point = list(theta) + [1.0] * len(eft.background_processes)
    t = expansion(point, eft.n_wcs, dtype=torch.float64).numpy()[eft.mask]
    ratios = np.zeros((len(eft.seeds), keep.size))
    for i, e in enumerate(eft.seeds):
        member = np.full(keep.size, np.nan)
        member[eft.seed_finite[e]] = t @ eft.seed_Gamma[e]
        ratios[i, keep] = member[keep]
    return ratios


def eft_closure(binning, w0, rho, w_hat0, keep, members):
    """
    Closure of the learned EFT ratio alone at theta (nu = 0), hat w_b = sum_i w^s_i(c0, 0) r_hat(x_i; theta) against
    w_b = sum_i w^s_i(theta, 0), with Var_MC = sum_i (hat w_i - w_i)^2 of the member mean and Var_seeds from the
    scatter of the members' hat w_b; rms_bias is relative to lambda_K_b(theta, 0) of every event, as in the nuisance
    tests.

    Args:
        binning: binning of the events (closure_binning)
        w0: [N] w^s(theta_ref, 0) per event
        rho: [N] rho_i(theta) per event (factorisation_weights)
        w_hat0: [N] w^s(c0, 0) r_hat(x; theta) per event, 0 where the EFT ensemble drops the event
        keep: [N] bool, the events the EFT ensemble keeps
        members: [S, N] w^s(c0, 0) r_hat_e(x; theta) per member e (eft_member_ratios), 0 outside keep
    Returns:
        (sums, component_bias result with one row); sums has bias, mc_var, sm_yield (lambda_K_b(theta, 0) of every
        event), nominal and truth (both w_b on the kept events), shift_var (0: the target is the nominal yield) and
        learned_shift_var (sum_i (hat w_i - w_i)^2 on the kept events)
    """
    index, n_bins = binning['index'], binning['n_bins']
    learned, true = w_hat0[keep], (w0 * rho)[keep]
    truth = binned_sums(true[None], index[:, keep], n_bins)
    sums = {
        'bias': (binned_sums(members[:, keep], index[:, keep], n_bins) - truth)[:, None],
        'mc_var': binned_sums(((learned - true) ** 2)[None], index[:, keep], n_bins),
        'sm_yield': binned_sums((w0 * rho)[None], index, n_bins)[0],
        'nominal': truth[0],
        'truth': truth[0],
    }
    sums['shift_var'] = np.zeros(n_bins)
    sums['learned_shift_var'] = sums['mc_var'][0]
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
    # paired rows, one per event in the same order: the rows of an event are already together
    if len(i0) == len(i1) and np.array_equal(i0, i1) and np.all(np.diff(i0) > 0) and torch.equal(index0, index1):
        dw = np.asarray(w_hat, np.float64) - np.asarray(w, np.float64)
        return binned_sums((dw**2)[None], index0, n_bins)[0]
    events = np.tile(np.concatenate([i0, i1]).astype(np.int64), n_features)
    bins = torch.cat([index0, index1], dim=1).reshape(-1).numpy().astype(np.int64)
    signed = np.tile(np.concatenate([np.asarray(w_hat, np.float64), -np.asarray(w, np.float64)]), n_features)
    # rows of one event in one bin are summed first; bins of different features never coincide
    codes, entry = np.unique(events * n_bins + bins, return_inverse=True)
    dw = np.bincount(entry, weights=signed)
    return np.bincount(codes % n_bins, weights=dw**2, minlength=n_bins)


def nuisance_sums(cache, binning, w0, targets, methods=None, terms=None, w_hat0=None, offset=None):
    """
    Binned sums for the closure of every nuisance network (nuisance n, direction d) in kinematic bins K_b.

    Args:
        cache: NuisanceCache
        binning: binning dict of the events of config['data'] (closure_binning)
        w0: [N] w^s(theta_ref, 0) per event, or w^s(theta, 0) at another WC point
        targets: callable targets(n, d) (ClosureTargets, ThetaTargets) returning a dict with
            rows0: [N] bool, the events the network is tested on (its term and eras)
            i0: [N] event index i of each event
            w1, index1, i1: per target row, w^s(theta_ref, nu), bins [F, M] and event index i (weight_shift: the
                events of config['data'], w0 syst_ratio; kinematic_shift: the shifted events, bin_events of their x')
        methods: methods of the nuisances to test (default every method); the others give None
        terms: terms of the nuisances to test (default every term); the others give None
        w_hat0: [N] base weight of the learned yield, hat w_b = sum_i w_hat0_i exp f_hat(x_i) (default w0; with the
            learned EFT ratio, w^s(c0, 0) times the term of the learned EFT ratio the nuisance belongs to)
        offset: [N] added to every event's learned weight, hat w_b = sum_i (w_hat0_i exp f_hat(x_i) + offset_i)
            (default 0; with the learned EFT ratio, the other terms times w^s(c0, 0), or w^s(theta, 0) - w_hat0, which
            removes the closure of the learned EFT ratio alone)
    Returns:
        list over (n, d), row k = 2 n + d, of dicts for validation.component_bias with
            bias: [S, 1, B] per member e, hat w_b - w_b with hat w_b = sum_i w^s_i(theta_ref, 0) exp f_hat_e(x_i)
            mc_var: [1, B] Var_MC with <f_hat>_e (matched_variance)
            sm_yield: [B] lambda_K_b(theta, 0) = sum_i w0_i over every event of the data, the yield of every term in
                every era at the point of the test (normalises rms_bias, the same for every network)
            nominal: [B] sum_i w0_i over the tested events, the nominal yield w_b(nu = 0) of the network's term in its
                eras
            truth: [B] target yield w_b
            shift_var: [B] MC-statistics variance of the true shift w_b - w_b(nu = 0) (matched_variance of the
                target against the nominal rows)
            learned_shift_var: [B] MC-statistics variance of the learned shift hat w_b - w_b(nu = 0) with
                <f_hat>_e, sum_i (hat w_i - w0_i)^2 over the tested events
            rate_mc_var, rate_s2: Var_MC and Var_seeds of the rate closure sum_b Delta_b over every tested event,
                from one bin holding them all (Var_MC with the terms of the events that move between bins) and from
                each member's total
        None for the nuisances not tested
    """
    w0 = np.asarray(w0, dtype=np.float64)
    w_hat0 = w0 if w_hat0 is None else np.asarray(w_hat0, dtype=np.float64)
    offset = np.zeros_like(w0) if offset is None else np.asarray(offset, dtype=np.float64)
    index, n_bins = binning['index'], binning['n_bins']
    total_yield = binned_sums(w0[None], index, n_bins)[0]
    # <f_hat>_e on every event (nan where some member's f_hat is not finite; those events are left out)
    mean = np.full(cache.f_hat.shape[:2] + cache.finite.shape, np.nan)
    mean[:, :, cache.finite] = cache.f_hat
    out = []
    for n, nuisance in enumerate(cache.nuisances):
        for d in range(len(DIRECTIONS)):
            if (methods is not None and nuisance['method'] not in methods) or (
                terms is not None and nuisance['term'] not in terms
            ):
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
            # the rate closure in one bin holding every tested event: its Var_MC keeps the terms of the events that
            # move between bins, and Var_seeds comes from each member's total (one feature counts every event once)
            one0 = torch.zeros((1, int(rows0.sum())), dtype=index.dtype)
            one1 = torch.zeros((1, int(keep1.sum())), dtype=index.dtype)
            rate_mc_var = matched_variance(w_hat, one0, t['i0'][rows0], t['w1'][keep1], one1, t['i1'][keep1], 1)[0]
            totals = np.stack(bias)[:, binning['feature'] == 0].sum(1)
            rate_s2 = totals.var(ddof=1) / len(totals) if len(totals) > 1 else 0.0
            out.append(
                {
                    'bias': np.stack(bias)[:, None],
                    'mc_var': mc_var[None],
                    'sm_yield': total_yield,
                    'nominal': binned_sums(w0[None, rows0], index[:, rows0], n_bins)[0],
                    'truth': truth,
                    'shift_var': matched_variance(
                        w0[rows0],
                        index[:, rows0],
                        t['i0'][rows0],
                        t['w1'][keep1],
                        t['index1'][:, keep1],
                        t['i1'][keep1],
                        n_bins,
                    ),
                    'learned_shift_var': binned_sums(((w_hat - w0[rows0]) ** 2)[None], index[:, rows0], n_bins)[0],
                    'rate_mc_var': float(rate_mc_var),
                    'rate_s2': float(rate_s2),
                }
            )
    return out


def pair_sums(cache, binning, w0, targets):
    """
    Binned sums for the closure of the parts of the interpolation in nu, the symmetric part
    f_hat^sym = (f_hat^Up + f_hat^Down) / 2 and the slope f_hat^slope = (f_hat^Up - f_hat^Down) / 2, as
    Delta^sym_b = (Delta^Up_b + Delta^Down_b) / 2 and Delta^slope_b = (Delta^Up_b - Delta^Down_b) / 2 in kinematic
    bins K_b.

    Args:
        cache, binning, w0: as nuisance_sums
        targets: ClosureTargets (its pair(n): Down and Up targets on one event index)
    Returns:
        list over nuisances n of {'fit_symmetry': sums, 'fit_slope': sums}, each a dict for validation.component_bias
        as nuisance_sums (sm_yield the yield of every event, nominal that of the tested events); the shift of the
        symmetric part is measured from the nominal yield, (w^Up_b + w^Down_b) / 2 - w_b(nu = 0), the slope is a shift
        itself, (w^Up_b - w^Down_b) / 2, and shift_var counts an event in the nominal, Up and Down samples once
    """
    w0 = np.asarray(w0, dtype=np.float64)
    index, n_bins = binning['index'], binning['n_bins']
    total_yield = binned_sums(w0[None], index, n_bins)[0]
    mean = np.full(cache.f_hat.shape[:2] + cache.finite.shape, np.nan)
    mean[:, :, cache.finite] = cache.f_hat
    out = []
    for n in range(len(cache.nuisances)):
        pair = targets.pair(n)
        rows0 = pair[0]['rows0'] & cache.finite
        # a shifted event whose nominal event is left out is left out too; events only in the shifted samples stay
        keep1 = [~np.isin(t['i1'], t['i0'][~rows0]) for t in pair]
        truth = [binned_sums(t['w1'][None, k], t['index1'][:, k], n_bins)[0] for t, k in zip(pair, keep1)]
        nominal = binned_sums(w0[None, rows0], index[:, rows0], n_bins)[0]

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
            # the nominal yield the shift is measured from: w0 for the symmetric part, none for the slope
            base = w0[rows0] if sign == 1 else np.zeros(int(rows0.sum()))
            index1 = torch.cat([pair[1]['index1'][:, keep1[1]], pair[0]['index1'][:, keep1[0]]], dim=1)
            i1 = np.concatenate([pair[1]['i1'][keep1[1]], pair[0]['i1'][keep1[0]]])
            sums[part] = {
                'bias': np.stack(bias)[:, None],
                'mc_var': matched_variance(w_hat, index[:, rows0], pair[0]['i0'][rows0], w, index1, i1, n_bins)[None],
                'sm_yield': total_yield,
                'nominal': nominal,
                'truth': (truth[1] + sign * truth[0]) / 2,
                'shift_var': matched_variance(base, index[:, rows0], pair[0]['i0'][rows0], w, index1, i1, n_bins),
                'learned_shift_var': binned_sums(((w_hat - base) ** 2)[None], index[:, rows0], n_bins)[0],
            }
        out.append(sums)
    return out


def interpolation_yield(cache, w0, targets, nu):
    """
    Yield of every nuisance on the events it is tested on against nu: <exp f(nu)> weighted by w^s(theta_ref, 0), with
    f(nu) the interpolation of <f_hat>_e (nuisanceCache.interpolate).

    Args:
        cache: NuisanceCache
        w0: [N] w^s(theta_ref, 0) per event
        targets: ClosureTargets (its rows0(n): the events the nuisance is tested on)
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
        (lambda_K_b(theta, 0) of every event), nominal (the nominal yield of the tested events), shift_var and
        learned_shift_var (the MC-statistics variances of the true and the learned shift), and, for the tests whose
        sums have them (nuisance_sums), [rows] arrays rate_mc_var and rate_s2 (Var_MC and Var_seeds of the rate
        closure)
    """
    keys = ('D', 's2', 'mc_var', 'truth', 'sm_yield', 'nominal', 'shift_var', 'learned_shift_var')
    out = {key: np.full((len(sums), n_bins), np.nan) for key in keys}
    if any(s is not None and 'rate_mc_var' in s for s in sums):
        for key in ('rate_mc_var', 'rate_s2'):
            out[key] = np.array([np.nan if s is None else s[key] for s in sums])
    for row, (s, bias) in enumerate(zip(sums, biases)):
        if s is None:
            continue
        out['D'][row], out['s2'][row] = bias['D'][0], bias['s2'][0]
        out['mc_var'][row], out['truth'][row] = s['mc_var'][0], s['truth']
        out['sm_yield'][row], out['nominal'][row] = s['sm_yield'], s['nominal']
        out['shift_var'][row], out['learned_shift_var'][row] = s['shift_var'], s['learned_shift_var']
    return out


def write_sums(name, cache, binning, tests, nu, yields):
    """
    Write nuisance_sums.npz to name, the per-bin arrays behind nuisance_diagnostics.yml. Every key is listed in
    docs/validation.md:
        nuisances, methods: [n_n] names and methods; network tests have row k = 2 n + d
        binning/names, binning/feature, binning/discrete, binning/edges/<feature>: the binning K_b
        <test>/D, s2, mc_var, truth, sm_yield, nominal, shift_var, learned_shift_var: [rows, B] (test_arrays) for the
            tests closure, fit_symmetry, fit_slope, factorisation/<label>, combined/<label>, eft_only/<label>,
            eft_matched/<label>
        <test>/rate_mc_var, rate_s2: [rows] Var_MC and Var_seeds of the rate closure (test_arrays), for the tests
            closure, factorisation/<label>, combined/<label>
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


def closure_entry(bias, features, z_threshold, skip=()):
    """
    Summary of one closure test (a component_bias result with one row): flagged, max_Z and its feature, rms_bias, Z.
    The features in skip are not tested: their Z is nan, and they do not count in flagged, max_Z, worst_feature and
    rms_bias.
    """
    tested = np.array([f not in skip for f in features])
    z = np.where(tested, bias['Z'][0], np.nan)
    rms_bias = np.where(tested, bias['rms_bias'][0], np.nan)
    return {
        'flagged': bool(np.nanmax(z) > z_threshold),
        'max_Z': float(np.nanmax(z)),
        'worst_feature': features[int(np.nanargmax(z))],
        'rms_bias': float(np.nanmax(rms_bias)),
        'Z': {f: float(v) for f, v in zip(features, z)},
    }


def member_misclosure(sums, binning, z_threshold, skip=()):
    """
    Closure of each ensemble member of one network-direction, pooled over the bins K_b of every tested network input,
    each relative to lambda_K_b(theta, 0) (sums['sm_yield'], the yield of every event at the point of the test):
        misclosure: m_a = sqrt(<m_e^2>_e), with m_e = sqrt(mean_b max(bias_e,b^2 - Var_MC,b, 0) / lambda_K_b^2) the
            noise-subtracted rms misclosure of member e (the definition of network_misclosure in the EFT validation)
        member_misclosure: m_e of every member e
        mean_misclosure: the same for the ensemble mean, sqrt(mean_b max(Delta_b^2 - Var_MC,b, 0) / lambda_K_b^2);
            m_a / mean_misclosure is about sqrt(n_e) when the members' misclosures are independent training noise and
            about 1 when the members share one bias
        shift_size: sqrt(mean_b (w_b - w_b(nu = 0))^2 / lambda_K_b^2), the rms true shift on the same scale
        members_flagged: number of members whose own closure Z exceeds z_threshold in some tested input
    Var_MC of each member is approximated by Var_MC of the ensemble mean. The features in skip are not tested.

    Args:
        sums: one nuisance_sums dict (bias [S, 1, B], mc_var, sm_yield, truth, nominal)
        binning: the binning of the sums
        z_threshold: a member is flagged when its Z exceeds this in some tested input
        skip: names of the network inputs not tested
    Returns:
        dict with misclosure, member_misclosure, mean_misclosure, shift_size, members_flagged
    """
    tested = [i for i, f in enumerate(binning['names']) if f not in skip]
    b = np.isin(binning['feature'], tested) & (sums['sm_yield'] > 0)
    lam2 = sums['sm_yield'][b] ** 2
    bias, mc_var = sums['bias'][:, 0, b], sums['mc_var'][0, b]
    members = np.sqrt(np.mean(np.maximum(bias**2 - mc_var, 0) / lam2, axis=1))
    mean = np.sqrt(np.mean(np.maximum(bias.mean(0) ** 2 - mc_var, 0) / lam2))
    shift = np.sqrt(np.mean((sums['truth'][b] - sums['nominal'][b]) ** 2 / lam2))
    flagged = 0
    for e in range(len(sums['bias'])):
        own = {'bias': sums['bias'][e : e + 1], 'mc_var': sums['mc_var'], 'sm_yield': sums['sm_yield']}
        flagged += bool(np.nanmax(component_bias(own, binning)['Z'][0, tested]) > z_threshold)
    return {
        'misclosure': float(np.sqrt(np.mean(members**2))),
        'member_misclosure': [float(m) for m in members],
        'mean_misclosure': float(mean),
        'shift_size': float(shift),
        'members_flagged': flagged,
    }


def write_results(
    name,
    cache,
    binning,
    closure,
    closure_sums,
    parts,
    factorisation,
    combined,
    eft_only,
    eft_matched,
    z_threshold,
    era_feature=None,
):
    """
    Write nuisance_diagnostics.yml to name: the nuisance cache used, the bins of every network input, the network-
    directions whose closure is flagged (largest Z first), and per nuisance its method, term and eras; per direction the
    closure of its network averaged over members and the misclosure of each member alone; the closure of the symmetric
    part and the slope of the interpolation; the closure at other WC points (factorisation); and the
    closure with the learned EFT ratio (combined), each in every network input. The era input is not tested for a
    nuisance of one era, whose events fill a single bin of it. Every rms_bias is relative to lambda_K_b(theta, 0), the
    yield of every event at the point of the test, so the rms_bias of different tests and nuisances are comparable. The
    rms_bias_change of a combined test is measured against the learned EFT ratio alone on the same events (baseline
    eft_only), so it is what the nuisance adds to the learned EFT ratio; the test stores the rms_bias of its baseline as
    baseline_rms_bias. Every key is listed in docs/validation.md.

    Args:
        name: output directory
        cache: NuisanceCache
        binning: the binning of the closure
        closure: component_bias result of every network, row k = 2 n + d
        closure_sums: nuisance_sums of every network, row k = 2 n + d, for the closure of each member alone
            (member_misclosure)
        parts: per nuisance n, {'fit_symmetry': component_bias result, 'fit_slope': component_bias result}
        factorisation: {label: component_bias result of every network at that WC point, row k = 2 n + d, None for
            the networks not tested}
        combined: as factorisation, the nuisance ratio times the term of the learned EFT ratio it belongs to, plus
            the other terms
        eft_only: {label: component_bias result of the learned EFT ratio alone at that point}
        eft_matched: {label: component_bias result of the learned EFT ratio alone on the events of each combined test,
            row k = 2 n + d, None for the networks without one}
        z_threshold: a test is flagged when its Z exceeds this in some feature
        era_feature: name of the network input holding the era index (year_loc), None if it is not an input
    Returns:
        the contents of nuisance_diagnostics.yml
    """
    features = binning['names']
    nuisances = {}
    for n, nuisance in enumerate(cache.nuisances):
        eras = nuisance.get('eras')
        skip = (era_feature,) if era_feature in features and eras is not None and len(eras) == 1 else ()
        entry = {'method': nuisance['method'], 'term': nuisance['term'], 'eras': eras}
        for d, direction in enumerate(DIRECTIONS):
            row = len(DIRECTIONS) * n + d
            entry[direction] = closure_entry(closure[row], features, z_threshold, skip)
            entry[direction].update(member_misclosure(closure_sums[row], binning, z_threshold, skip))
        for part in ('fit_symmetry', 'fit_slope'):
            entry[part] = closure_entry(parts[n][part], features, z_threshold, skip)
        entry['factorisation'] = {}
        for label, results in factorisation.items():
            if results[len(DIRECTIONS) * n] is None:
                entry['factorisation'][label] = None
                continue
            entry['factorisation'][label] = {}
            for d, direction in enumerate(DIRECTIONS):
                test = closure_entry(results[len(DIRECTIONS) * n + d], features, z_threshold, skip)
                test['rms_bias_change'] = test['rms_bias'] - entry[direction]['rms_bias']
                entry['factorisation'][label][direction] = test
        entry['combined'] = {}
        for label, results in combined.items():
            if results[len(DIRECTIONS) * n] is None:
                entry['combined'][label] = None
                continue
            entry['combined'][label] = {}
            for d, direction in enumerate(DIRECTIONS):
                row = len(DIRECTIONS) * n + d
                test = closure_entry(results[row], features, z_threshold, skip)
                reference = closure_entry(eft_matched[label][row], features, z_threshold, skip)
                test['baseline'], test['baseline_rms_bias'] = 'eft_only', reference['rms_bias']
                test['rms_bias_change'] = test['rms_bias'] - reference['rms_bias']
                entry['combined'][label][direction] = test
        nuisances[nuisance['name']] = entry
    # the network-directions whose closure is flagged, largest Z first, named as their closure plots
    flagged = [
        (entry[direction]['max_Z'], f'{n}_{direction}')
        for n, entry in nuisances.items()
        for direction in DIRECTIONS
        if entry[direction]['flagged']
    ]
    results = {
        'nuisance': {
            'path': cache.path,
            'created': cache.manifest['created'],
            'seeds': list(cache.seeds),
            'theta_ref': list(cache.theta_ref),
            'ensemble': cache.manifest['ensemble'],
        },
        'binning': {f: int((binning['feature'] == i).sum()) for i, f in enumerate(features)},
        'flagged': [label for _, label in sorted(flagged, key=lambda item: -item[0])],
        'factorisation_points': list(factorisation),
        'eft_only': {label: closure_entry(bias, features, z_threshold) for label, bias in eft_only.items()},
        'nuisances': nuisances,
    }
    with open(f'{name}/nuisance_diagnostics.yml', 'w') as f:
        yaml.safe_dump(results, f, sort_keys=False)
    return results


# default thresholds of the possible problems (config problems)
PROBLEM_THRESHOLDS = {
    'not_learned': 0.8,
    'closure_size': 1.0,
    'yield_n_sigma': 3.0,
    'yield_fraction': 0.2,
    'factorisation': 1.0,
    'combined_excess': 0.5,
}


def find_problems(results, closure, closure_sums, binning, thresholds):
    """
    Possible problems of the nuisance model, each a criterion on the size of a misclosure rather than on its
    significance (the significant misclosures are in results['flagged']):
        not_learned: m_rms / delta w_rms (mean_misclosure / shift_size) >= not_learned, the error of the
            ensemble average a sizeable part of the shift it learns;
        closure_size: closure rms_bias / the rms_bias of the learned EFT ratio alone at theta_SM >= closure_size (not
            tested without the factorisation point sm);
        yield_mismatch: the rate closure at nu = +-1, sum_b Delta_b / w_b(nu = 0) over every tested event, in units of
            its noise sqrt(Var_MC + Var_seeds), >= yield_n_sigma, counted only when the rate closure is also larger
            than yield_fraction times the true rate shift;
        factorisation: factorisation rms_bias / the rms_bias of the learned EFT ratio alone at that point
            >= factorisation;
        combined_excess: combined rms_bias_change / its baseline, the learned EFT ratio alone on the same events,
            >= combined_excess.
    The severity of a problem is its value / threshold (>= 1).

    Args:
        results: output of write_results
        closure: component_bias result of every network, row k = 2 n + d
        closure_sums: nuisance_sums of every network, row k = 2 n + d (nominal, truth, rate_mc_var, rate_s2)
        binning: the binning of the closure
        thresholds: {criterion: threshold}, the keys of PROBLEM_THRESHOLDS
    Returns:
        list of {nuisance, test, criterion, value, threshold, severity}, largest severity first; test is the
        direction or <test>/<point>/<direction>
    """
    # every tested event is in exactly one bin of each feature
    first = binning['feature'] == 0
    eft_sm = results['eft_only'].get('sm', {}).get('rms_bias')
    problems = []

    def add(nuisance, test, criterion, value, threshold):
        if threshold > 0 and value >= threshold:
            problems.append(
                {
                    'nuisance': nuisance,
                    'test': test,
                    'criterion': criterion,
                    'value': float(value),
                    'threshold': float(threshold),
                    'severity': float(value / threshold),
                }
            )

    for n, (name, entry) in enumerate(results['nuisances'].items()):
        for d, direction in enumerate(DIRECTIONS):
            test = entry[direction]
            if test['shift_size'] > 0:
                ratio = test['mean_misclosure'] / test['shift_size']
                add(name, direction, 'not_learned', ratio, thresholds['not_learned'])
            if eft_sm:
                add(name, direction, 'closure_size', test['rms_bias'] / eft_sm, thresholds['closure_size'])
            row = len(DIRECTIONS) * n + d
            sums = closure_sums[row]
            nominal = sums['nominal'][first].sum()
            rate = closure[row]['D'][0, first].sum() / nominal
            noise = np.sqrt(sums['rate_mc_var'] + sums['rate_s2']) / nominal
            shift = abs(sums['truth'][first].sum() / nominal - 1)
            if noise > 0 and abs(rate) > thresholds['yield_fraction'] * shift:
                add(name, direction, 'yield_mismatch', abs(rate) / noise, thresholds['yield_n_sigma'])
        for label, tests in entry['factorisation'].items():
            reference = results['eft_only'][label]['rms_bias']
            for direction, test in (tests or {}).items():
                if reference > 0:
                    add(
                        name,
                        f'factorisation/{label}/{direction}',
                        'factorisation',
                        test['rms_bias'] / reference,
                        thresholds['factorisation'],
                    )
        for label, tests in entry['combined'].items():
            for direction, test in (tests or {}).items():
                if test['baseline_rms_bias'] > 0:
                    add(
                        name,
                        f'combined/{label}/{direction}',
                        'combined_excess',
                        test['rms_bias_change'] / test['baseline_rms_bias'],
                        thresholds['combined_excess'],
                    )
    return sorted(problems, key=lambda problem: -problem['severity'])


def write_problems(name, problems, thresholds):
    """
    Write nuisance_problems.yml to name: the thresholds used and the possible problems of find_problems, largest
    severity first. Every key is listed in docs/validation.md.

    Args:
        name: output directory
        problems: output of find_problems
        thresholds: the thresholds find_problems used
    """
    with open(f'{name}/nuisance_problems.yml', 'w') as f:
        yaml.safe_dump({'thresholds': thresholds, 'problems': problems}, f, sort_keys=False)


def plot_results(name, feature_config=None, z_threshold=5.0, value_names=None):
    """
    Plots of the validation from nuisance_diagnostics.yml and nuisance_sums.npz under name, written to name/plots
    (replaced): Z of every network and of the parts of the interpolation in every network input (z_heatmap_full,
    z_heatmap_parts_full; the compact versions show the flagged tests and the next 10, at most COMPACT_COLUMNS,
    sorted by their largest Z; hatched cells are not tested), Delta_rms against Z_max of the networks and of each
    part (size_vs_significance_{networks,slope,symmetry}), the misclosure of the ensemble average of every network,
    with the range of its members, against the size of its true shift (network_misclosure), the closure of every
    network in its worst feature (closure/{flagged,passed}) and of every part (closure_parts/{flagged,passed}), the
    yield against nu, one figure per nuisance (interpolation_yield/<nuisance>), the learned and true yield
    changes at nu = +-1 with their rate closure (yield_summary_<method>), and per factorisation point the rms bias of
    the factorisation test against the closure at theta_ref, with the possible problems and the threshold they were
    found with (factorisation_<label>), and of the combined test against the learned EFT ratio alone on the same
    events, with the possible problems and the threshold of each direction (eft_combined_<label>). Nuisances are
    ordered by method, then by name. Every file is listed in docs/validation.md.

    Args:
        name: output directory of the validation
        feature_config: contents of the features yml, for the TeX labels of the features
        z_threshold: flagging threshold of the bias test (config bias.z_threshold), for the outlined cells and the
            lines at the threshold
        value_names: {feature name: {value: name}} of the features with named values (validation.era_names), for the
            tick labels of the closure plots, optional
    """
    with open(f'{name}/nuisance_diagnostics.yml') as f:
        diag = yaml.safe_load(f)
    sums = np.load(f'{name}/nuisance_sums.npz')
    entries = diag['nuisances']
    features = [str(f) for f in sums['binning/names']]
    binning = {
        'names': features,
        'feature': sums['binning/feature'],
        'edges': [sums[f'binning/edges/{f}'] for f in features],
        'discrete': sums['binning/discrete'],
    }
    nuisances, methods = [str(n) for n in sums['nuisances']], [str(m) for m in sums['methods']]
    # by method (weight_shift first), then by name; rows 2 n + d of the network tests follow n
    order = sorted(range(len(nuisances)), key=lambda n: (list(METHOD_NAMES).index(methods[n]), nuisances[n].lower()))
    names = [nuisances[n] for n in order]
    plots = f'{name}/plots'
    shutil.rmtree(plots, ignore_errors=True)
    for sub in (
        'closure/flagged',
        'closure/passed',
        'closure_parts/flagged',
        'closure_parts/passed',
        'interpolation_yield',
    ):
        os.makedirs(f'{plots}/{sub}', mode=0o755)

    def test(key):
        keys = ('D', 's2', 'mc_var', 'truth', 'sm_yield', 'nominal', 'shift_var', 'learned_shift_var')
        return {k: sums[f'{key}/{k}'] for k in keys}

    # Z of every network (Down beside Up) and of the parts, grouped by method / part: full in the shared order,
    # compact the flagged tests and the next 10, sorted by their largest Z within each group
    heatmaps = {
        'z_heatmap': (
            np.array([[entries[n][d]['Z'][f] for f in features] for n in names for d in DIRECTIONS]),
            [f'{n} {DIRECTION_TEX[d]}' for n in names for d in DIRECTIONS],
            [METHOD_NAMES[entries[n]['method']] for n in names for d in DIRECTIONS],
            list(METHOD_NAMES.values()),
        ),
        'z_heatmap_parts': (
            np.array([[entries[n][p]['Z'][f] for f in features] for p in PARTS for n in names]),
            names * len(PARTS),
            [PARTS[p] for p in PARTS for n in names],
            list(PARTS.values()),
        ),
    }
    for stem, (Z, labels, kinds, kind_order) in heatmaps.items():
        max_z = np.nanmax(Z, axis=1)
        # the flagged tests and the next 10, at most COMPACT_COLUMNS, largest Z first
        n_compact = min(int((max_z > z_threshold).sum()) + 10, COMPACT_COLUMNS)
        compact = [int(k) for k in np.argsort(-max_z)[:n_compact]]
        for version, columns, sort in (('full', None, False), ('compact', compact, True)):
            plot_z_heatmap(
                Z,
                labels,
                kinds,
                features,
                f'{plots}/{stem}_{version}.png',
                z_threshold,
                components=columns,
                feature_config=feature_config,
                kind_order=kind_order,
                tick_labels=labels,
                sort=sort,
            )

    # rms bias against Z of the networks and of each part, one plot per test on common axes
    def size(key):
        return {n: (entries[n][key]['max_Z'], entries[n][key]['rms_bias']) for n in names}

    tests = {'networks': ('networks', {d: size(d) for d in DIRECTIONS})}
    tests.update({p.removeprefix('fit_'): (PARTS[p], {'part': size(p)}) for p in PARTS})
    reference = None
    if 'sm' in diag['eft_only']:
        reference = (r'EFT ensemble at $\theta_{\mathrm{SM}}$', diag['eft_only']['sm']['rms_bias'])
    floor, z_floor = 1e-5, 0.1
    points = np.array([p for _, t in tests.values() for values in t.values() for p in values.values()], dtype=float)
    z = np.maximum(points[:, 0], z_floor)
    rms = np.maximum(np.append(points[:, 1], [] if reference is None else [reference[1]]), floor)
    limits = (
        (min(np.nanmin(z), z_threshold) / 2, max(np.nanmax(z), z_threshold) * 2),
        (np.nanmin(rms) / 2, np.nanmax(rms) * 2),
    )
    method_of = {n: entries[n]['method'] for n in names}
    for stem, (title, t) in tests.items():
        plot_size_significance(
            t,
            title,
            method_of,
            z_threshold,
            f'{plots}/size_vs_significance_{stem}.png',
            limits,
            reference=reference,
            floor=floor,
            z_floor=z_floor,
        )

    # misclosure of the ensemble average of every network, with its members' range, against the size of the true shift
    plot_nuisance_misclosure(
        {(n, direction): entries[n][direction] for n in names for direction in DIRECTIONS},
        {n: entries[n]['method'] for n in names},
        f'{plots}/network_misclosure.png',
    )

    # closure of every network and of every part in their worst feature, as the learned against the true shift
    closure = test('closure')
    for n in order:
        entry = entries[nuisances[n]]
        for d, direction in enumerate(DIRECTIONS):
            result = entry[direction]
            group = 'flagged' if result['flagged'] else 'passed'
            sign = DIRECTION_SIGN[direction]
            plot_nuisance_closure(
                closure,
                len(DIRECTIONS) * n + d,
                features.index(result['worst_feature']),
                result['max_Z'],
                binning,
                f'{plots}/closure/{group}/{nuisances[n]}_{direction}.png',
                f'{nuisances[n]} {DIRECTION_TEX[direction]}',
                (rf'$w_b^{{{sign}}} - w_b^{{0}}$', rf'$\hat{{w}}_b^{{{sign}}} - w_b^{{0}}$'),
                feature_config=feature_config,
                value_names=value_names,
            )
    for p in PARTS:
        part = test(p)
        for n in order:
            result = entries[nuisances[n]][p]
            group = 'flagged' if result['flagged'] else 'passed'
            plot_nuisance_closure(
                part,
                n,
                features.index(result['worst_feature']),
                result['max_Z'],
                binning,
                f'{plots}/closure_parts/{group}/{nuisances[n]}_{p}.png',
                f'{nuisances[n]} {PARTS[p]}',
                PART_LABELS[p],
                subtract_nominal=p != 'fit_slope',
                feature_config=feature_config,
                value_names=value_names,
            )

    # yield against nu, with the true yields at nu = -1, +1 from the closure of the networks (one feature
    # counts every tested event once)
    first = binning['feature'] == 0
    nominal = closure['nominal'][:, first].sum(1)
    truth = (closure['truth'][:, first].sum(1) / nominal).reshape(-1, len(DIRECTIONS))
    truth_err = (np.sqrt(sums['closure/rate_mc_var']) / nominal).reshape(-1, len(DIRECTIONS))
    plot_interpolation_yield(
        sums['interpolation/nu'],
        sums['interpolation/yield'][order],
        truth[order],
        truth_err[order],
        names,
        [methods[n] for n in order],
        f'{plots}/interpolation_yield',
    )

    # learned and true kappa at nu = -1, +1 with the noise of the rate closure, as find_problems, the directions
    # reported as yield_mismatch, and the threshold they were found with
    rate = (closure['D'][:, first].sum(1) / nominal).reshape(-1, len(DIRECTIONS))
    rate_noise = (np.sqrt(sums['closure/rate_mc_var'] + sums['closure/rate_s2']) / nominal).reshape(-1, len(DIRECTIONS))
    flagged = np.zeros((len(nuisances), len(DIRECTIONS)), dtype=bool)
    reported = {'thresholds': {}, 'problems': []}
    if os.path.exists(f'{name}/nuisance_problems.yml'):
        with open(f'{name}/nuisance_problems.yml') as f:
            reported = yaml.safe_load(f)
    yield_n_sigma = reported['thresholds'].get('yield_n_sigma', PROBLEM_THRESHOLDS['yield_n_sigma'])
    for p in reported['problems']:
        if p['criterion'] == 'yield_mismatch':
            flagged[nuisances.index(p['nuisance']), DIRECTIONS.index(p['test'])] = True
    for m in METHOD_NAMES:
        inside = [i for i, n in enumerate(order) if methods[n] == m]
        if not inside:
            continue
        rows = [order[i] for i in inside]
        plot_yield_summary(
            truth[rows] + rate[rows],
            truth[rows],
            rate_noise[rows],
            flagged[rows],
            [names[i] for i in inside],
            'k',
            f'{METHOD_NAMES[m]}s',
            f'{plots}/yield_summary_{m}.png',
            n_sigma=yield_n_sigma,
        )

    # rms bias of the factorisation test against the closure at theta_ref, with its possible problems and their
    # threshold, and of the tests with the learned EFT ratio against their baseline, per factorisation point
    for label in diag['factorisation_points']:
        point = r'\theta_{\mathrm{SM}}' if label == 'sm' else label

        def rms(nuisances, key):
            return np.array(
                [
                    [(entries[n] if key is None else entries[n][key][label])[d]['rms_bias'] for d in DIRECTIONS]
                    for n in nuisances
                ]
            )

        def baseline(nuisances, key):
            return np.array([[entries[n][key][label][d]['baseline_rms_bias'] for d in DIRECTIONS] for n in nuisances])

        def reported_as(nuisances, test, criterion):
            # the directions of test at this point reported as possible problems of criterion
            found = {(p['nuisance'], p['test']) for p in reported['problems'] if p['criterion'] == criterion}
            return np.array([[(n, f'{test}/{label}/{d}') in found for d in DIRECTIONS] for n in nuisances])

        tested = [n for n in names if entries[n]['factorisation'][label] is not None]
        if tested:
            threshold = reported['thresholds'].get('factorisation', PROBLEM_THRESHOLDS['factorisation'])
            eft_label = rf'$\hat{{r}}(x; {point})$'
            plot_rms_comparison(
                tested,
                rms(tested, None),
                rms(tested, 'factorisation'),
                reported_as(tested, 'factorisation', 'factorisation'),
                (r'$\hat{r}(x; \theta_{\mathrm{ref}}, \nu)$', rf'$\hat{{r}}(x; {point}, \nu)$'),
                f'{plots}/factorisation_{label}.png',
                reference=(
                    eft_label if threshold == 1 else f'{threshold:g} × {eft_label}',
                    threshold * diag['eft_only'][label]['rms_bias'],
                ),
            )
        checked = [n for n in names if entries[n]['combined'][label] is not None]
        if not checked:
            continue
        threshold = reported['thresholds'].get('combined_excess', PROBLEM_THRESHOLDS['combined_excess'])
        before = baseline(checked, 'combined')
        plot_rms_comparison(
            checked,
            before,
            rms(checked, 'combined'),
            reported_as(checked, 'combined', 'combined_excess'),
            (rf'$\hat{{r}}(x; {point})$', rf'$\hat{{r}}(x; {point}, \nu)$'),
            f'{plots}/eft_combined_{label}.png',
            limits=(1 + threshold) * before,
            limit_label='threshold',
        )


def main(config_path, refit=False):
    """
    Run the validation of the nuisance ensemble (docs/validation.md) on the events of config['data'], the signal and
    background events of the EFT ensemble, and write nuisance_diagnostics.yml, nuisance_problems.yml, nuisance_sums.npz and plots/
    (plot_results) under config['name']. Each test is a closure in merged equal-count bins of every network input:
        closure: every nuisance network averaged over members, on the events of its term;
        fit_symmetry, fit_slope: the symmetric part and the slope of the interpolation in nu, with its yield
            against nu;
        factorisation: the signal weight_shift networks, trained at theta_ref, at other WC points theta;
        combined: the learned model at those points, the nuisance ratio times the term of the learned EFT ratio it
            belongs to plus the other terms (for a signal nuisance w^s(c0, 0) [r_hat(x; nu) T(theta) Gamma_sig +
            mu_bg Gamma_bg]), on the events of every term in the nuisance's eras;
        eft_only: the learned EFT ratio alone at those points, on every event and on the events of each combined test
            (eft_matched, the baseline of the combined test).
    The networks' reference point theta_ref is the WC point of c0 (check_reference), so
    w^s(theta_ref, 0) = w^s(c0, 0) on every event.

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
    eft = EnsembleCache(config['eft_cache'])
    if not eft.matches(config['eft_config']):
        sys.exit(f'[error]: {config["eft_cache"]} is stale; rerun the EFT validation with --refit')

    nominal = load_combined(config['data'], 'nominal', ('features', 'ids', 'event_id', 'syst_ratio', 'w_ref'))
    features = nominal[0]
    cache = load_nuisance_ensemble(config, features, refit)
    print(f'[info]: checks at theta_ref = {cache.theta_ref} on {config["data"]}')
    w0 = sample_weight(nominal, 'nominal', config).double().numpy()
    coefs = load_combined(config['data'], 'nominal', ('coefs',))[1]
    terms = event_terms(coefs, eft.n_wcs, eft.background_processes)

    # closure of every network, averaged over members, in the bins of the EFT bias test (merged further where a term
    # is sparse) of every network input
    cfg = config.get('bias', {})
    binning = closure_binning(
        features, config['features'], w0, terms, cfg.get('n_bins', 20), cfg.get('features'), cfg.get('min_neff', 30)
    )
    targets = ClosureTargets(config, cache, nominal, w0, binning, terms)
    closure_sums = nuisance_sums(cache, binning, w0, targets)
    closure = [component_bias(s, binning) for s in closure_sums]
    tests = {'closure': (closure_sums, closure)}

    # closure of the symmetric part and the slope of the interpolation
    pairs = pair_sums(cache, binning, w0, targets)
    parts = [{part: component_bias(s, binning) for part, s in pair.items()} for pair in pairs]
    for part in ('fit_symmetry', 'fit_slope'):
        tests[part] = ([pair[part] for pair in pairs], [p[part] for p in parts])
    nu = np.linspace(-2, 2, 41)
    yields = interpolation_yield(cache, w0, targets, nu)

    # factorisation: closure at other WC points theta with the networks trained at theta_ref, in the bins of
    # theta_ref, for the signal weight_shift networks; kinematic_shift is not tested, its variation files have no
    # structure constants, and rho = 1 on the background events, whose closure at theta is the closure at theta_ref
    points = config.get('factorisation', {}).get('points', {'sm': [1.0] + [0.0] * len(config['wcs'])})
    weights = factorisation_weights(coefs, cache.theta_ref, points, len(eft.background_processes))
    del coefs
    factorisation = {}
    for label, rho in weights.items():
        sums = nuisance_sums(
            cache, binning, w0 * rho, ThetaTargets(targets, rho), methods=('weight_shift',), terms=('signal',)
        )
        factorisation[label] = [None if s is None else component_bias(s, binning) for s in sums]
        tests[f'factorisation/{label}'] = (sums, factorisation[label])

    # the learned model of the scans at the factorisation points (weight_shift only), r_hat(x; nu) at nu_n = +-1 times
    # the term of the learned EFT ratio the nuisance belongs to, plus the other terms, on the events of every term;
    # the closure of the learned EFT ratio alone at nu = 0 for comparison
    combined, eft_only, eft_matched = {}, {}, {}
    for label, rho in weights.items():
        parts_theta = eft_terms(eft, points[label])
        keep = np.all([np.isfinite(part) for part in parts_theta.values()], axis=0)
        learned = {term: w0 * np.nan_to_num(part) for term, part in parts_theta.items()}
        del parts_theta
        total = sum(learned.values())
        theta_targets = ThetaTargets(targets, rho, keep, every_term=True)
        combined_sums = [None] * (len(DIRECTIONS) * len(cache.nuisances))
        for term, w_hat0 in learned.items():
            # the nuisance multiplies the term it belongs to; the other terms enter unchanged
            sums = nuisance_sums(
                cache,
                binning,
                w0 * rho,
                theta_targets,
                methods=('weight_shift',),
                terms=(term,),
                w_hat0=w_hat0,
                offset=total - w_hat0,
            )
            combined_sums = [a if b is None else b for a, b in zip(combined_sums, sums)]
        combined[label] = [None if s is None else component_bias(s, binning) for s in combined_sums]
        tests[f'combined/{label}'] = (combined_sums, combined[label])
        members = w0[None] * eft_member_ratios(eft, points[label], keep)
        sums, eft_only[label] = eft_closure(binning, w0, rho, total, keep, members)
        tests[f'eft_only/{label}'] = ([sums], [eft_only[label]])
        # the learned EFT ratio alone on the events of each combined test (every term, in the nuisance's eras), the
        # baseline of the combined test; one closure per set of eras
        by_eras = {None: (sums, eft_only[label])}
        matched_sums = [None] * len(combined_sums)
        eft_matched[label] = [None] * len(combined_sums)
        for n, nuisance in enumerate(cache.nuisances):
            if combined_sums[len(DIRECTIONS) * n] is None:
                continue
            eras = nuisance.get('eras')
            key = None if eras is None else tuple(sorted(eras))
            if key not in by_eras:
                by_eras[key] = eft_closure(binning, w0, rho, total, keep & targets.era_rows(n), members)
            for d in range(len(DIRECTIONS)):
                matched_sums[len(DIRECTIONS) * n + d], eft_matched[label][len(DIRECTIONS) * n + d] = by_eras[key]
        tests[f'eft_matched/{label}'] = (matched_sums, eft_matched[label])

    z_threshold = cfg.get('z_threshold', 5.0)
    year_loc = config.get('year_loc', 21)
    era_feature = next((f for f, p in config['features'].items() if p['loc'] == year_loc), None)
    results = write_results(
        name,
        cache,
        binning,
        closure,
        closure_sums,
        parts,
        factorisation,
        combined,
        eft_only,
        eft_matched,
        z_threshold,
        era_feature,
    )
    write_sums(name, cache, binning, tests, nu, yields)

    # possible problems of the nuisance model, by the size of its misclosures
    thresholds = {**PROBLEM_THRESHOLDS, **config.get('problems', {})}
    problems = find_problems(results, closure, closure_sums, binning, thresholds)
    write_problems(name, problems, thresholds)
    print(f'[info]: {len(problems)} possible problems; see {name}/nuisance_problems.yml')
    plot_results(
        name,
        config['features'],
        config.get('bias', {}).get('z_threshold', 5.0),
        era_names(config),
    )
    n_flagged = len(results['flagged'])
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
        plot_results(
            config['name'],
            config['features'],
            config.get('bias', {}).get('z_threshold', 5.0),
            era_names(config),
        )
    else:
        main(args.config, args.refit)
