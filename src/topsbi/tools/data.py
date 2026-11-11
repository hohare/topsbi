import numpy as np
import os, torch, yaml


def find_zero_terms(
    coefs: torch.tensor,
    wcs: list,
    lumi_per_event: torch.tensor,
    features: torch.tensor,
    n_bins: int = 20,
    tol: float = 1e-6,
    background_processes: list = (),
):
    """
    WC pairs whose structure constants s_q(z) are negligible in every region of phase space.

    Every feature column is cut into n_bins equal-count quantile bins K_b (fewer for discrete features). The size of
    a term in a bin is sum_{i in K_b} L |s_q(z_i)| / sum_{i in K_b} L |s_SM(z_i)|, and a term is dropped if its
    largest size over all bins of all features is below tol. Absolute values stop positive and negative
    contributions from cancelling; the per-bin ratio keeps a term that is large only in a small region (a tail),
    which an inclusive ratio dilutes. The largest bin ratio is never below the inclusive ratio, so this drops a
    subset of the terms an inclusive criterion drops.

    Only the signal enters: with background_processes the last K columns (one per process, expansion) are left out,
    and so are the background events, whose T(theta) columns are all zero.

    Args:
        coefs: structure constants s_q(z) [n_events, n_q + K]
        wcs: WC names in the order of coefs
        lumi_per_event: L of each event's era
        features: event features [n_events, n_features]; every column is binned
        n_bins: quantile bins per feature
        tol: threshold on the largest bin ratio
        background_processes: names of the background processes, in the order of their columns
    Returns:
        list of [wc_i, wc_j] pairs (names as in zero_terms.yml)
    """
    names = ['sm'] + list(wcs)
    n_q = len(names) * (len(names) + 1) // 2
    if coefs.shape[1] != n_q + len(background_processes):
        raise ValueError(
            f'wcs and background_processes give {n_q + len(background_processes)} columns but coefs has {coefs.shape[1]}'
        )
    coefs = coefs[:, :n_q]
    signal = coefs.ne(0).any(1)
    coefs, lumi_per_event, features = coefs[signal], lumi_per_event[signal], features[signal]
    w = lumi_per_event.to(torch.float64)[:, None] * coefs.to(torch.float64).abs()
    largest = torch.zeros(w.shape[1], dtype=torch.float64)
    for column in features.to(torch.float64).T.numpy():
        edges = np.unique(np.quantile(column, np.linspace(0, 1, n_bins + 1)))
        bins = torch.from_numpy(np.clip(np.searchsorted(edges, column, side='right') - 1, 0, max(len(edges) - 2, 0)))
        sums = torch.zeros(max(len(edges) - 1, 1), w.shape[1], dtype=torch.float64).index_add_(0, bins, w)
        largest = torch.maximum(largest, (sums / sums[:, :1]).max(0).values)
    rows, cols = torch.tril_indices(len(names), len(names))
    return [[names[i], names[j]] for i, j, r in zip(rows.tolist(), cols.tolist(), largest.tolist()) if r < tol]


def load_zero_terms(config: dict):
    """WC pairs listed in the file at config['zero_terms']; none if the key is absent."""
    if 'zero_terms' not in config:
        return []
    with open(config['zero_terms']) as f:
        return yaml.safe_load(f)['zero_terms']


def zero_term_mask(wcs: list, zero_terms: list, background_processes: list = ()):
    """
    Mask over the columns of T(theta, mu) (expansion) that is False for the listed (structurally zero) WC pairs.

    The columns of T(theta) follow get_lower_tri; the one column per background process after them is always kept,
    as zero_terms names WC pairs only.
    Args:
        wcs: WC names in the order of the point
        zero_terms: [wc_i, wc_j] pairs ('sm' for the SM entry), e.g. from load_zero_terms
        background_processes: names of the background processes, in the order of their columns
    Returns:
        boolean tensor of length (n + 1)(n + 2) / 2 + K
    """
    names = ['sm'] + list(wcs)
    mask = torch.ones(len(names) * (len(names) + 1) // 2 + len(background_processes), dtype=torch.bool)
    for a, b in zero_terms:
        if a not in names or b not in names:
            raise ValueError(f'zero term {[a, b]} names something not in {names}: only WC pairs can be zero terms')
        i, j = sorted((names.index(a), names.index(b)), reverse=True)
        mask[i * (i + 1) // 2 + j] = False
    return mask


def kept_columns(config: dict, n_cols: int):
    """
    Columns of the expansion T(theta, mu) to keep: all of them unless config lists zero_terms.

    With background_processes the data hold one column per process after T(theta); the width of the data is then
    always checked against wcs and background_processes.
    Args:
        config: dictionary with 'device' and optional 'zero_terms', 'background_processes' (with 'wcs' if either
            is given)
        n_cols: number of columns of the data (coefs)
    Returns:
        boolean tensor [n_cols] on config['device'], False for the zero terms
    """
    zero_terms = load_zero_terms(config)
    background_processes = config.get('background_processes', [])
    if not zero_terms and not background_processes:
        return torch.ones(n_cols, dtype=torch.bool, device=config['device'])
    keep = zero_term_mask(config['wcs'], zero_terms, background_processes)
    if keep.numel() != n_cols:
        raise ValueError(f'wcs and background_processes give {keep.numel()} columns but the data have {n_cols}')
    return keep.to(config['device'])


def prepare_weights(
    data: torch.utils.data.TensorDataset,
    config: dict,
):
    """
    Features and the two weights of an EFT training.

    Args:
        data: TensorDataset (or a Subset of one) whose first two tensors are (features x, coefs); further tensors
            (ids, and in the combined format event_id, syst_ratio, w_ref) are ignored
        config: training config; 'c0' / 'c1' are theta_0 / theta_1, 'lumi' the per-era luminosities
    Returns:
        features, w0, w1: x, w^s(theta_0) and w^s(theta_1) (get_weights)
    """
    if config['method'] != 'eft':
        raise ValueError(f"prepare_weights serves the method 'eft', not {config['method']!r}")
    year_loc = config.get('year_loc', 21)
    features, coefs = data[:][:2]
    features = features.to(torch.float32)
    w0, w1 = get_weights(coefs, config, features[:, year_loc])
    return features, w0, w1


def get_lower_tri(
    theta: list,
    dtype: type = torch.float32,
    off_diag: float = 1.0,
):
    """
    Quadratic expansion T(theta) of an SM-inclusive WC point theta = (1, theta_1, ..., theta_n): the lower triangle,
    in (i, j <= i) order, of the outer product theta theta^T, so that w(z; theta) = s_q(z) . T(theta).

    Args:
        theta: SM-inclusive WC point (first entry 1)
        dtype: dtype of the result
        off_diag: factor to multiply off-diagonal elements by
    Returns:
        T(theta), length (n + 1)(n + 2) / 2
    """
    theta = torch.as_tensor(theta, dtype=dtype)
    outer = torch.outer(theta, theta)
    n = outer.shape[0]
    rows, cols = torch.tril_indices(n, n)
    off_rows, off_cols = torch.tril_indices(n, n, offset=-1)
    outer[off_rows, off_cols] *= off_diag
    return outer[rows, cols]


def expansion(
    point: list,
    n_wcs: int,
    dtype: type = torch.float32,
):
    """
    Expansion T(theta, mu) = [T(theta), mu_1, ..., mu_K] of a point (1, theta_1, ..., theta_n, mu_1, ..., mu_K), so
    that w(z; theta, mu) = s_q(z) . T(theta, mu).

    The first n + 1 entries are the SM-inclusive WC point theta, expanded quadratically by get_lower_tri. The K
    entries after them are the strengths mu_p of the background processes (config 'background_processes'), which
    enter linearly, one column each, after T(theta). Without background processes T(theta, mu) = T(theta).

    Args:
        point: SM-inclusive WC point followed by the background strengths
        n_wcs: number of WCs n
        dtype: dtype of the result
    Returns:
        T(theta, mu), length (n + 1)(n + 2) / 2 + K
    """
    point = torch.as_tensor(point, dtype=dtype)
    if point.ndim != 1 or len(point) < n_wcs + 1:
        raise ValueError(f'a point needs the SM entry and {n_wcs} WCs, got shape {tuple(point.shape)}')
    return torch.cat([get_lower_tri(point[: n_wcs + 1], dtype=dtype), point[n_wcs + 1 :]])


def lumi_per_event(years: torch.tensor, config: dict):
    """
    Integrated luminosity of each event's era, from config['lumi'] indexed by the year_int feature.

    Args:
        years: era index of each event (year_int feature)
        config: dictionary with 'lumi' (list of per-era luminosities)
    Returns:
        float64 tensor [N] of luminosities, on the device of years
    """
    if 'lumi' not in config:
        raise KeyError("'lumi' is required: a list of luminosities indexed by the year_int feature")
    return torch.tensor(config['lumi'], dtype=torch.float64, device=years.device)[years.long()]


# tensor order of combined nominal / variation samples for nuisance training; w_ref is each event's weight at the
# reference WC point chosen when the sample was combined (the same point for nominal and every variation)
NUISANCE_COLUMNS = {
    'nominal': ('features', 'coefs', 'ids', 'event_id', 'syst_ratio', 'w_ref'),
    'variation': ('features', 'w_ref', 'ids', 'event_id'),
}


def sample_weight(tensors: list, layout: str, config: dict):
    """
    Event weight of a combined nominal or variation sample: w^s(theta_ref) = L_era w_ref x 1000, where w_ref is the
    stored weight at the reference point theta_ref (theta_SM or theta_g, chosen when the sample was combined).

    Args:
        tensors: tensors of a combined train.p / validation.p (NUISANCE_COLUMNS order)
        layout: 'nominal' or 'variation'
        config: dictionary with 'lumi' and 'device' (optional 'year_loc', default 21)
    Returns:
        float32 tensor [N] of weights
    """
    columns = dict(zip(NUISANCE_COLUMNS[layout], tensors))
    years = columns['features'][:, config.get('year_loc', 21)]
    w_ref = columns['w_ref'][:, 0].to(torch.float64)
    return (lumi_per_event(years, config) * w_ref * 1000).to(torch.float32)


# offset that keeps the train/test salt distinct from the salt (0) used for train/validation when combining
SPLIT_SALT_OFFSET = 1 << 32


def splitmix64(x: np.ndarray):
    """
    splitmix64 finalizer on a uint64 array: every input bit affects every output bit; arithmetic wraps modulo 2**64.

    The step adds the golden-ratio increment and applies the mix64 finalizer of SplitMix (G. L. Steele Jr., D. Lea and
    C. H. Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014, doi:10.1145/2660193.2660195), whose
    shifts and multipliers are D. Stafford's Mix13 ("Better Bit Mixing - Improving on MurmurHash3's 64-bit
    Finalizer", 2011), as in S. Vigna's reference splitmix64.c (https://prng.di.unimi.it/splitmix64.c). The same
    function makes the train/validation split in ttbarEFT analysis/combine_tensors.py (split_fraction), so
    event_split keys the train/test split the same way with a different salt.
    """
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def event_keys(tensors: list, layout: str, year_loc: int = 21):
    """
    Identity of each event: (era, mtt slice, every event_id column).

    The same event has the same key in nominal and in every variation. The era and mtt slice are included because
    (run, lumi block, event) can repeat between the private samples.

    Args:
        tensors: tensors of a combined train.p / validation.p (NUISANCE_COLUMNS order)
        layout: 'nominal' or 'variation'
        year_loc: feature column of the era index
    Returns:
        int64 numpy array [N, 2 + number of event_id columns]
    """
    columns = dict(zip(NUISANCE_COLUMNS[layout], tensors))
    return np.column_stack(
        [
            columns['features'][:, year_loc].numpy().astype(np.int64),
            columns['ids'][:, 2].numpy().astype(np.int64),
            columns['event_id'].numpy().astype(np.int64),
        ]
    )


def matched_events(keys0: np.ndarray, keys1: np.ndarray):
    """
    Event index i of each nominal and shifted row, over the set of distinct events.

    An event in both samples gets the same index in both, so two-sample metrics can treat each MC event as one
    independent unit. Rows with the same key in one sample share an index.

    Args:
        keys0: event_keys of the nominal rows
        keys1: event_keys of the shifted rows
    Returns:
        (i0, i1): int64 tensors [N0], [N1] with values in [0, number of distinct events)
    """
    if keys0.shape[1] != keys1.shape[1]:
        raise ValueError(f'event keys have {keys0.shape[1]} and {keys1.shape[1]} columns')
    _, events = np.unique(np.concatenate([keys0, keys1]), axis=0, return_inverse=True)
    events = torch.from_numpy(events.reshape(-1))
    return events[: len(keys0)], events[len(keys0) :]


def event_split(tensors: list, layout: str, seed: int, test_fraction: float = 0.25, year_loc: int = 21):
    """
    Test-split mask (early stopping, LR scheduler) from a hash of each event's identity.

    The key is the event_keys row (era, mtt slice, event_id), so an event is on the same side in nominal and in every
    variation, and in every nuisance training with the same seed. The seed changes the partition, as in
    sample_bootstrap; the salt is offset so it never reproduces the train/validation split made when combining.

    Args:
        tensors: tensors of a combined train.p (NUISANCE_COLUMNS order)
        layout: 'nominal' or 'variation'
        seed: training seed
        test_fraction: fraction of events in the test split
        year_loc: feature column of the era index
    Returns:
        boolean tensor [N], True for test events
    """
    key = np.full(len(tensors[0]), SPLIT_SALT_OFFSET + seed, dtype=np.uint64)
    for field in event_keys(tensors, layout, year_loc).T:
        key = splitmix64(key ^ field.astype(np.uint64))
    u = (key >> np.uint64(11)).astype(np.float64) / 2.0**53
    return torch.from_numpy(u < test_fraction)


def syst_ratio_column(root: str, systematic, direction: str):
    """
    Index of a weight systematic and direction in the nominal syst_ratio tensor.

    Args:
        root: combined data directory (holds nominal/systematics.yml)
        systematic: name as in systematics.yml, e.g. 'renorm', or a list of names trained as one systematic (per-era
            systematics, each acting on its own era's events only, e.g. ['btagSFbc_2017', 'btagSFbc_2018'])
        direction: 'up' or 'down' (case-insensitive)
    Returns:
        (k, d): syst_ratio[:, k, d] is r(x, z; theta, nu) = w^s(theta, nu) / w^s(theta, 0) of that systematic; for a
        list, k is the list of its columns and r is the product of syst_ratio[:, k, d] over them
    """
    with open(f'{root}/nominal/systematics.yml') as f:
        spec = yaml.safe_load(f)
    names = spec['syst_ratio']
    directions = [d.lower() for d in spec.get('directions', ['Up', 'Down'])]
    wanted = [systematic] if isinstance(systematic, str) else list(systematic)
    unknown = [name for name in wanted if name not in names]
    if unknown:
        raise ValueError(f'unknown weight systematic {unknown}; available: {names}')
    if direction.lower() not in directions:
        raise ValueError(f'unknown direction {direction!r}; available: {directions}')
    d = directions.index(direction.lower())
    if isinstance(systematic, str):
        return names.index(systematic), d
    return [names.index(name) for name in wanted], d


def load_combined(path: str, layout: str, names: tuple):
    """
    Selected columns of a combined train.p / validation.p, read through a memory map.

    Columns that are not requested (e.g. the 153 float64 coefs) are never read from disk.

    Args:
        path: file to load
        layout: 'nominal' or 'variation' (NUISANCE_COLUMNS)
        names: columns to load
    Returns:
        list in NUISANCE_COLUMNS order, None for columns not loaded (accepted by sample_weight and event_split)
    """
    tensors = torch.load(path, weights_only=False, mmap=True).tensors
    if len(tensors) != len(NUISANCE_COLUMNS[layout]):
        raise ValueError(f'{path}: {len(tensors)} tensors, expected {NUISANCE_COLUMNS[layout]}')
    return [t.clone() if name in names else None for name, t in zip(NUISANCE_COLUMNS[layout], tensors)]


def weight_shift_sample(tensors: list, config: dict, k, d: int):
    """
    Paired sample of a weight systematic: the nominal events with w^s(theta, 0) and w^s(theta, nu).

    Args:
        tensors: nominal tensors (load_combined)
        config: dictionary with 'lumi' and 'device'
        k: syst_ratio column of the systematic (syst_ratio_column), or a list of columns of per-era systematics
            trained as one: their ratio is the product, and each event may be shifted by one of them at most
        d: direction index, 0 up / 1 down
    Returns:
        features x, w0 = w^s(theta, 0) (sample_weight), w1 = w^s(theta, nu) = w0 r(x, z; theta, nu)
    Raises:
        ValueError: for a list of columns, if an event has a ratio != 1 in more than one of them
    """
    columns = dict(zip(NUISANCE_COLUMNS['nominal'], tensors))
    w0 = sample_weight(tensors, 'nominal', config)
    ratio = columns['syst_ratio'][:, k, d]
    if isinstance(k, list):
        shared = int(((ratio != 1).sum(dim=1) > 1).sum())
        if shared:
            raise ValueError(f'{shared} events are shifted by more than one of the merged systematics {k}')
        ratio = ratio.prod(dim=1)
    return columns['features'], w0, w0 * ratio


def kinematic_shift_sample(nominal: list, shifted: list, config: dict):
    """
    Two-sample data of a kinematic systematic.

    Args:
        nominal: nominal tensors (load_combined)
        shifted: variation tensors (load_combined)
        config: dictionary with 'lumi' and 'device'
    Returns:
        {'rows': (features, weights, labels) for Model.labelled_loss, label 0 nominal / 1 shifted,
         'nominal': (x, w0, i0), 'shifted': (x_shift, w1, i1) for plots and metrics, with w0 = w^s(theta, 0),
         w1 = w^s(theta, nu) and i0 / i1 the event index i (matched_events), the same for an event in both samples}
    """
    x = dict(zip(NUISANCE_COLUMNS['nominal'], nominal))['features']
    x_shift = dict(zip(NUISANCE_COLUMNS['variation'], shifted))['features']
    w0, w1 = sample_weight(nominal, 'nominal', config), sample_weight(shifted, 'variation', config)
    i0, i1 = matched_events(event_keys(nominal, 'nominal'), event_keys(shifted, 'variation'))
    labels = torch.cat([torch.zeros_like(w0), torch.ones_like(w1)])
    return {
        'rows': (torch.cat([x, x_shift]), torch.cat([w0, w1]), labels),
        'nominal': (x, w0, i0),
        'shifted': (x_shift, w1, i1),
    }


def background_sample(signal: list, background: list, config: dict):
    """
    Two-sample data of a background training: the signal at its reference point theta_ref (label 0; theta_g for a
    sample combined at the generation point) and the sample of one background process p (label 1), so the learned
    ratio is r_hat(x; p) = f_p(x) / f_S(x; theta_ref).

    The two samples are different processes, so no event is in both: the event index i runs over the signal rows
    first and the background rows after them.
    Args:
        signal: nominal tensors of the signal sample combined at theta_ref (load_combined)
        background: nominal tensors of the background sample (load_combined)
        config: dictionary with 'lumi' and 'device'
    Returns:
        {'rows': (features, weights, labels) for Model.labelled_loss, label 0 signal / 1 background,
         'nominal': (x, w0, i0) of the signal and 'shifted': (x_p, w1, i1) of the background for plots and metrics,
         with w0 = w^s(theta_ref), w1 = L_era w_ref x 1000 of the background (negative weights kept) and i0 / i1
         disjoint event indices}
    """
    x = dict(zip(NUISANCE_COLUMNS['nominal'], signal))['features']
    x_p = dict(zip(NUISANCE_COLUMNS['nominal'], background))['features']
    w0, w1 = sample_weight(signal, 'nominal', config), sample_weight(background, 'nominal', config)
    i0 = torch.arange(len(w0))
    i1 = torch.arange(len(w0), len(w0) + len(w1))
    labels = torch.cat([torch.zeros_like(w0), torch.ones_like(w1)])
    return {
        'rows': (torch.cat([x, x_p]), torch.cat([w0, w1]), labels),
        'nominal': (x, w0, i0),
        'shifted': (x_p, w1, i1),
    }


def _select(tensors: list, mask: torch.tensor):
    """rows of every loaded column (None columns stay None)"""
    return [None if t is None else t[mask] for t in tensors]


def _variation_eras(shifted: list, year_loc: int):
    """era indices (year_int feature) present in a variation sample"""
    features = dict(zip(NUISANCE_COLUMNS['variation'], shifted))['features']
    return torch.unique(features[:, year_loc].to(torch.int64))


def _keep_eras(nominal: list, eras: torch.tensor, year_loc: int):
    """
    Nominal rows of the given eras. A kinematic variation produced only in some eras (JER per year) leaves the other
    eras unchanged, r(x, z; theta, nu) = 1 there, so they are not part of its training.
    """
    features = dict(zip(NUISANCE_COLUMNS['nominal'], nominal))['features']
    return _select(nominal, torch.isin(features[:, year_loc].to(torch.int64), eras))


def _load_variations(root: str, variations: list, file: str, need: tuple, year_loc: int):
    """
    Variation tensors of one file, concatenated over the variation directories, and the era indices they cover.
    Several directories (per-era variations trained as one systematic) must cover disjoint eras.
    """
    parts, eras = [], torch.zeros(0, dtype=torch.int64)
    for variation in variations:
        shifted = load_combined(f'{root}/{variation}/{file}', 'variation', need)
        variation_eras = _variation_eras(shifted, year_loc)
        overlap = variation_eras[torch.isin(variation_eras, eras)]
        if len(overlap):
            raise ValueError(f'{variation}: eras {overlap.tolist()} in {file} are also in another merged variation')
        parts.append(shifted)
        eras = torch.cat([eras, variation_eras])
    merged = [None if parts[0][i] is None else torch.cat([part[i] for part in parts]) for i in range(len(parts[0]))]
    return merged, torch.unique(eras)


def _check_same_sample_setup(root: str, variation: str):
    """
    Nominal and the variation must share the reference point and the train/validation split salt.

    Only checked when both directories have a sample.yml; samples made without one are accepted.
    """
    specs = []
    for name in ('nominal', variation):
        path = f'{root}/{name}/sample.yml'
        if not os.path.exists(path):
            return
        with open(path) as f:
            specs.append(yaml.safe_load(f))
    for key in ('reference', 'salt_split'):
        if specs[0].get(key) != specs[1].get(key):
            raise ValueError(f'{key} differs: nominal {specs[0].get(key)!r}, {variation} {specs[1].get(key)!r}')


def load_nuisance_data(config: dict):
    """
    Train / test / holdout data of a nuisance training (weight_shift or kinematic_shift).

    train.p is split into train and test (early stopping) by event_split with the training seed, so an event is on
    the same side in nominal and the variation; validation.p is the holdout. The variation of kinematic_shift is the
    directory <systematic><Up|Down>; its nominal sample is restricted to the eras present in the variation (a
    variation produced only for some eras, e.g. JER per year, leaves the others unchanged). A list of systematics
    (per-era systematics on disjoint eras) is trained as one: weight_shift takes the product of their ratios,
    kinematic_shift the union of their variation directories.

    Args:
        config: dictionary with 'method', 'data', 'systematic' (a name or a list of names), 'direction', 'seed',
            'lumi', 'device'
    Returns:
        {'train', 'test', 'holdout'}: (features, w0, w1) for weight_shift, kinematic_shift_sample dicts otherwise
    """
    root, method, seed = config['data'], config['method'], config['seed']
    direction = config['direction'].lower()
    systematics = [config['systematic']] if isinstance(config['systematic'], str) else list(config['systematic'])
    variations = None
    if method == 'weight_shift':
        k, d = syst_ratio_column(root, config['systematic'], direction)
        need = ('features', 'ids', 'event_id', 'syst_ratio', 'w_ref')
    elif method == 'kinematic_shift':
        variations = [f'{name}{direction.capitalize()}' for name in systematics]
        for variation in variations:
            if not os.path.isdir(f'{root}/{variation}'):
                raise FileNotFoundError(f'no variation {variation!r} in {root}')
            _check_same_sample_setup(root, variation)
        need = ('features', 'ids', 'event_id', 'w_ref')
    else:
        raise ValueError(f'{method!r} is not a nuisance method')

    def build(nominal, shifted):
        if method == 'weight_shift':
            return tuple(t.to(config['device']) for t in weight_shift_sample(nominal, config, k, d))
        sample = kinematic_shift_sample(nominal, shifted, config)
        return {key: tuple(t.to(config['device']) for t in value) for key, value in sample.items()}

    year_loc = config.get('year_loc', 21)
    eras = None
    out = {}
    for split_name, file in (('train', 'train.p'), ('holdout', 'validation.p')):
        nominal = load_combined(f'{root}/nominal/{file}', 'nominal', need)
        shifted = None
        if variations is not None:
            shifted, split_eras = _load_variations(root, variations, file, need, year_loc)
            if eras is None:
                eras = split_eras
                print(f"[INFO] {' + '.join(variations)}: eras {eras.tolist()}; nominal rows of other eras are left out")
            elif not torch.equal(split_eras, eras):
                raise ValueError(f'{variations}: eras {split_eras.tolist()} in {file} but {eras.tolist()} in train.p')
            nominal = _keep_eras(nominal, eras, year_loc)
        if split_name == 'holdout':
            out['holdout'] = build(nominal, shifted)
            continue
        # split on the CPU (event_split hashes numpy arrays), then move to the device
        test_nominal = event_split(nominal, 'nominal', seed)
        test_shifted = None if shifted is None else event_split(shifted, 'variation', seed)
        if not test_nominal.any() or (test_shifted is not None and not test_shifted.any()):
            raise ValueError(f'empty test split in {root}: too few events for early stopping')
        if shifted is None:
            out['train'] = build(_select(nominal, ~test_nominal), None)
            out['test'] = build(_select(nominal, test_nominal), None)
        else:
            out['train'] = build(_select(nominal, ~test_nominal), _select(shifted, ~test_shifted))
            out['test'] = build(_select(nominal, test_nominal), _select(shifted, test_shifted))
    return out


def load_background_data(config: dict):
    """
    Train / test / holdout data of a background training (method 'background').

    The signal sample is the nominal directory of config['data'], combined at its reference point theta_ref (theta_g
    for a sample combined at the generation point); the background sample is the nominal directory of
    config['background'], a sample whose sample.yml has eft: false. Each train.p is split into train and test (early
    stopping) by event_split with the training seed, so the signal test events are those of the EFT and nuisance
    trainings with the same seed; validation.p of both samples is the holdout.

    Args:
        config: dictionary with 'data', 'background', 'seed', 'lumi', 'device' (optional 'year_loc', default 21)
    Returns:
        {'train', 'test', 'holdout'}: background_sample dicts
    """
    specs = {}
    for key in ('data', 'background'):
        path = f'{config[key]}/nominal/sample.yml'
        if not os.path.exists(path):
            raise FileNotFoundError(f"config {key!r}: no combined sample in {config[key]} (missing {path})")
        with open(path) as f:
            specs[key] = yaml.safe_load(f)
    if specs['background'].get('eft', True):
        raise ValueError(f"config 'background': {config['background']} is not a background sample (sample.yml eft)")
    if specs['data'].get('eft') is False:
        raise ValueError(f"config 'data': {config['data']} is a background sample, expected the signal")
    need = ('features', 'ids', 'event_id', 'w_ref')
    year_loc = config.get('year_loc', 21)

    def build(signal, background):
        sample = background_sample(signal, background, config)
        return {key: tuple(t.to(config['device']) for t in value) for key, value in sample.items()}

    # split on the CPU (event_split hashes numpy arrays), then move to the device
    signal = load_combined(f"{config['data']}/nominal/train.p", 'nominal', need)
    background = load_combined(f"{config['background']}/nominal/train.p", 'nominal', need)
    test_signal = event_split(signal, 'nominal', config['seed'], year_loc=year_loc)
    test_background = event_split(background, 'nominal', config['seed'], year_loc=year_loc)
    if not test_signal.any() or not test_background.any():
        raise ValueError(f"empty test split in {config['data']} or {config['background']}: too few events")
    holdout = [load_combined(f'{config[key]}/nominal/validation.p', 'nominal', need) for key in ('data', 'background')]
    return {
        'train': build(_select(signal, ~test_signal), _select(background, ~test_background)),
        'test': build(_select(signal, test_signal), _select(background, test_background)),
        'holdout': build(*holdout),
    }


def background_reference(config: dict):
    """
    Process and reference point of a background training, from the sample.yml of its two samples.

    Args:
        config: dictionary with 'data' (signal sample) and 'background' (background sample)
    Returns:
        {'process': the background process p, 'reference': name of theta_ref ('gen' or 'sm'), 'reference_point':
         theta_ref (SM-inclusive) the signal was combined at}
    """
    with open(f"{config['data']}/nominal/sample.yml") as f:
        signal = yaml.safe_load(f)
    with open(f"{config['background']}/nominal/sample.yml") as f:
        background = yaml.safe_load(f)
    return {
        'process': background['process'],
        'reference': signal['reference'],
        'reference_point': [float(c) for c in signal['reference_point']],
    }


def is_combined(data: str):
    """True for a directory of the combined format (the combiner writes its sample.yml next to train.p)"""
    return os.path.exists(f'{data}/sample.yml')


def load_eft_combined(config: dict):
    """
    Train / test / holdout of an EFT training on the nominal directory of a combined sample (config['data']).

    train.p is split by event_split, so a seed has the same test events as the nuisance trainings with that seed and
    the split does not depend on the row order; the holdout is config['holdout'], by default <data>/validation.p.
    Only the features, coefs and event-key columns are read (load_combined).
    Args:
        config: EFT training config ('data', 'seed', 'c0', 'c1', 'lumi', 'device', optional 'holdout', 'year_loc')
    Returns:
        {'train', 'test', 'holdout'}: (x, w^s(theta_0), w^s(theta_1)) each (prepare_weights)
    """

    def weights(tensors):
        return prepare_weights([None if t is None else t.to(config['device']) for t in tensors], config)

    train = load_combined(f"{config['data']}/train.p", 'nominal', ('features', 'coefs', 'ids', 'event_id'))
    test = event_split(train, 'nominal', config['seed'], year_loc=config.get('year_loc', 21))
    if not test.any():
        raise ValueError(f"empty test split in {config['data']}: too few events for early stopping")
    holdout = load_combined(config.get('holdout') or f"{config['data']}/validation.p", 'nominal', ('features', 'coefs'))
    return {'train': weights(_select(train, ~test)), 'test': weights(_select(train, test)), 'holdout': weights(holdout)}


def get_weights(coefs: torch.tensor, config: dict, years: torch.tensor):
    """
    Stitched event weights w^s(theta_0) and w^s(theta_1).

    The stored structure constants s_q(z) (coefs) already carry xsec/nSumOfWeights and the detector scale factors,
    but not the luminosity, so w^s(theta) = L_era s_q(z) . T(theta) x 1000 (pb to fb), computed in float64. With
    background_processes, T(theta, mu) replaces T(theta): c0 and c1 end with the background strengths mu_p and coefs with the background columns
    (expansion).

    Args:
        coefs: structure constants s_q(z) per event (rows) and term of T(theta, mu) (columns)
        config: dictionary with 'c0' (theta_0), 'c1' (theta_1), the per-era luminosities 'lumi' and optional
            'background_processes'
        years: era index (year_int feature) of each event, used to look up the luminosity
    Returns:
        w0: w^s(theta_0) per event
        w1: w^s(theta_1) per event
    """
    if len(config['c0']) != len(config['c1']):
        raise ValueError(f"c0 has {len(config['c0'])} entries but c1 has {len(config['c1'])}")
    n_wcs = len(config['c0']) - 1 - len(config.get('background_processes', []))
    keep = kept_columns(config, coefs.shape[1])
    coefs = coefs[:, keep].to(torch.float64)
    lumi = lumi_per_event(years, config)
    T0 = expansion(config['c0'], n_wcs, dtype=torch.float64).to(config['device'])[keep]
    T1 = expansion(config['c1'], n_wcs, dtype=torch.float64).to(config['device'])[keep]
    w0 = lumi * (coefs @ T0) * 1000
    w1 = lumi * (coefs @ T1) * 1000
    return (w0).to(torch.float32), (w1).to(torch.float32)


def get_feature_normalization(features: torch.tensor):
    """
    Get mean and standard deviation to normalize a set of features.

    A feature that is constant in the sample (e.g. the era when only one era is present) gets std 1 instead of 0,
    so standardising it gives 0 rather than 0/0.
    Args:
        features: torch tensor whose rows are each event and whose columns are each features
    Returns:
        feature mean
        feature standard deviation
    """
    std = features.std(0)
    return features.mean(0), torch.where(std > 0, std, torch.ones_like(std))


def sample_bootstrap(sample: torch.utils.data.TensorDataset, seed: int):
    """
    Random 75/25 split of a dataset into training and test sets, reproducible with seed.
    """
    train, test = torch.utils.data.random_split(sample, [0.75, 0.25], generator=torch.Generator().manual_seed(seed))
    return train, test
