"""
Validation functions on a small synthetic ensemble with a known truth, written through the real cache code (no
trained networks): two WCs, three members of twelve training points plus SM, "networks" that return the true ratio
with 1% noise per event, optionally with a bias injected into one network.
"""

import pickle
import types

import numpy as np
import pytest
import torch
import yaml

import topsbi.validation as val
from topsbi.model.net import Net
from topsbi.tools.build_ratio import NetworkRatio, fit_gamma, generalized_inverse
from topsbi.tools.data import expansion, get_lower_tri
from topsbi.tools.ensembleCache import (
    EnsembleCache,
    SeedCollector,
    binned_sums,
    component_labels,
    merge_sparse,
    model_hashes,
    network_hash,
    quantile_binning,
    write_cache,
)
from topsbi.tools.plots import component_kinds, component_tex, network_kinds, network_tex, unit_tex, wc_tex

WCS = ['cA', 'cB']
N_EVENTS = 3000
MEMBERS = [0, 1, 2]
FEATURES = {'x0': {'loc': 0}, 'x1': {'loc': 1}, 'x2': {'loc': 2}}
POINTS = {
    'cA/p_one': (1, 0),
    'cA/n_one': (-1, 0),
    'cA/p_two': (2, 0),
    'cA/n_two': (-2, 0),
    'cB/p_one': (0, 1),
    'cB/n_one': (0, -1),
    'cB/p_two': (0, 2),
    'cB/n_two': (0, -2),
    'cA_cB/p_one_p_one': (1, 1),
    'cA_cB/p_one_n_one': (1, -1),
    'cA_cB/n_one_p_one': (-1, 1),
    'cA_cB/n_one_n_one': (-1, -1),
}
CROSS = [name for name in POINTS if name.startswith('cA_cB/')]
BAD = 'cA/p_two'  # the network a bias is injected into
NOISE = 0.01  # relative noise of every network's ratio, per event


def expanded(theta):
    """T(theta) of an SM-inclusive point as a float64 numpy row."""
    return get_lower_tri(list(map(float, theta)), dtype=torch.float64).numpy()


def truth():
    """Features, per-event component weights (L s_q x 1000 in validation) and w^s(theta_0) of the toy sample."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (N_EVENTS, 3))
    x[:, 2] = rng.integers(0, 3, N_EVENTS)  # a discrete input
    # w(theta) = (1, theta) A (1, theta)^T per event, A positive definite, so every weight is positive
    v = np.stack([np.ones(N_EVENTS), 0.5 * x[:, 0] + 0.2, 0.4 * x[:, 1] - 0.3], axis=1)
    a = v[:, :, None] * v[:, None, :] + 0.05 * np.eye(3)
    rows, cols = np.tril_indices(3)
    s = a[:, rows, cols] * np.where(rows == cols, 1.0, 2.0)
    weights = s * rng.uniform(0.5, 1.5, N_EVENTS)[:, None]  # unequal event weights, as in MC
    w0 = weights @ expanded([1, 0, 0])
    return x, weights, w0


def fake_fit(Theta, R, background_rows=None):
    """
    What RatioFit hands to the per-seed callback, for given Theta and R (all events finite); with background_rows,
    the fit whose background column (the last) comes from those rows alone.
    """
    Theta, R = torch.as_tensor(Theta, dtype=torch.float32), torch.as_tensor(R, dtype=torch.float32)
    finite = ~torch.isinf(R).any(0)
    if background_rows is None:
        solution, residuals, rank, _ = torch.linalg.lstsq(Theta, R[:, finite], driver='gelsd')
        Gamma = solution[: Theta.shape[1]].clone()
    else:
        background_rows = torch.as_tensor(background_rows, dtype=torch.bool)
        Gamma = fit_gamma(Theta, R[:, finite], background_rows, 1)
        residuals = ((Theta @ Gamma - R[:, finite]) ** 2).sum(0)
        rank = torch.linalg.matrix_rank(Theta)
    return types.SimpleNamespace(
        Theta=Theta,
        R=R,
        finite=finite,
        Gamma=Gamma,
        residuals=residuals,
        rank=rank,
        background_rows=background_rows,
    )


def build_toy(root, inject=None, members=MEMBERS):
    """
    Write the toy ensemble cache under root and return what the validation functions need.

    Args:
        root: directory for the fake training configs and the cache
        inject: optional (size, members): network BAD returns r (1 + size) for events with x0 > 0.5 in those members
        members: the ensemble members (seeds)
    """
    x, weights, w0 = truth()
    names = list(POINTS) + ['sm']
    Theta = np.stack([expanded((1, *POINTS[n])) for n in POINTS] + [expanded((1, 0, 0))])
    gamma = weights / w0[:, None]  # true per-event coefficients, so hat w = w0 gamma reproduces weights exactly
    R_true = Theta @ gamma.T
    seeds = {}
    for e in members:
        paths = []
        for name in names:
            directory = root / f'seed_{e}' / name
            directory.mkdir(parents=True)
            (directory / 'model.pt').write_bytes(f'{name} {e}'.encode())
            (directory / 'training.yml').write_text(yaml.safe_dump({'name': str(directory)}))
            paths.append(str(directory / 'training.yml'))
        seeds[e] = {'networks': paths[:-1], 'sm': paths[-1]}
    config = {'data': 'toy', 'c0': [1.0, 0.0, 0.0], 'wcs': WCS, 'lumi': [1.0], 'seeds': seeds}

    path = str(root / 'ensemble')
    collector = SeedCollector(seeds, path)
    Gamma, rank = 0, {}
    for e in members:
        rng = np.random.default_rng(100 + e)
        R = R_true * (1 + NOISE * rng.standard_normal(R_true.shape))
        if inject is not None and e in inject[1]:
            R[names.index(BAD)] *= 1 + inject[0] * (x[:, 0] > 0.5)
        fit = fake_fit(Theta, R)
        collector(e, fit)
        Gamma, rank[e] = Gamma + fit.Gamma / len(members), int(fit.rank)
    ensemble = types.SimpleNamespace(
        Gamma=Gamma,
        mask=torch.ones(Theta.shape[1], dtype=torch.bool),
        finite=torch.ones(N_EVENTS, dtype=torch.bool),
        background_processes=[],
    )
    ensemble.rank = rank
    write_cache(path, config, ensemble)

    features = torch.from_numpy(x).float()
    w0, weights = torch.from_numpy(w0), torch.from_numpy(weights)
    binning = merge_sparse(quantile_binning(features, FEATURES, 10), features, FEATURES, w0, 30)
    cache = EnsembleCache(path)
    sums = val.component_sums(cache, binning, w0, weights)
    return types.SimpleNamespace(
        cache=cache,
        config=config,
        Theta=Theta,
        names=names,
        R_true=R_true,
        x=x,
        w0=w0,
        weights=weights,
        binning=binning,
        sums=sums,
        bias=val.component_bias(sums, binning),
    )


@pytest.fixture(scope='module')
def null(tmp_path_factory):
    return build_toy(tmp_path_factory.mktemp('null'))


@pytest.fixture(scope='module')
def biased(tmp_path_factory):
    """Bias of 5% in network BAD of every member."""
    return build_toy(tmp_path_factory.mktemp('biased'), inject=(0.05, MEMBERS))


@pytest.fixture(scope='module')
def bad_copies(tmp_path_factory):
    """
    Ten members, bias of 10% in network BAD of four of them. A bad copy in few members mostly raises Var_seeds rather
    than the ensemble-mean bias (with one bad copy among three members, Delta^2 = Var_seeds exactly, so it never
    flags); four of ten still flag, and are fewer than seed_fraction (0.5) of the members, so the point is not dropped.
    """
    return build_toy(tmp_path_factory.mktemp('bad_copies'), inject=(0.1, [0, 1, 2, 3]), members=list(range(10)))


def flagged(toy, threshold=5.0):
    max_z = np.nanmax(toy.bias['Z'], axis=1)
    return [int(k) for k in np.argsort(-max_z) if max_z[k] > threshold]


def equal(a, b):
    """Bit-identical nested results (dicts, lists, arrays, tensors, scalars)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(u, v) for u, v in zip(a, b))
    if isinstance(a, (np.ndarray, torch.Tensor)):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def test_cache_round_trip(null, tmp_path):
    cache = null.cache
    assert cache.seeds == MEMBERS
    assert cache.mask.all() and cache.finite.all()
    assert cache.manifest['rank'] == {e: 6 for e in MEMBERS}
    for e in MEMBERS:
        network = cache.network(e)
        assert network['networks'] == null.names
        assert np.allclose(network['Theta'], null.Theta)
        assert cache.seed_R(e).shape == (len(null.names), N_EVENTS)
    assert np.allclose(cache.Gamma, np.mean([cache.seed_Gamma[e] for e in MEMBERS], axis=0), rtol=1e-6)
    assert cache.matches(null.config)
    assert not cache.matches({**null.config, 'c0': [1.0, 0.5, 0.0]})
    # a retrained network makes the cache stale
    model = yaml.safe_load(open(null.config['seeds'][0]['sm']))['name'] + '/model.pt'
    original = open(model, 'rb').read()
    try:
        open(model, 'wb').write(b'retrained')
        assert not cache.matches(null.config)
    finally:
        open(model, 'wb').write(original)


def test_bias_test_without_bias(null):
    Z = null.bias['Z']
    assert np.isfinite(Z).sum() > 0.9 * Z.size
    assert np.nanmax(Z) < 5
    assert abs(np.nanmean(Z)) < 2
    assert flagged(null) == []


def test_bias_test_finds_injected_bias(biased):
    # Gamma is linear in R, so the injected bias propagates through Theta^+ into every component
    shift = biased.R_true[biased.names.index(BAD)] * 0.05 * (biased.x[:, 0] > 0.5)
    gamma_shift = np.linalg.pinv(biased.Theta)[:, biased.names.index(BAD), None] * shift
    expected = binned_sums(gamma_shift * biased.w0.numpy(), biased.binning['index'], biased.binning['n_bins'])
    q = int(np.argmax(np.abs(expected).max(1)))
    assert q in flagged(biased)
    D = biased.bias['D'][q]
    assert np.abs(D - expected[q]).max() < 0.2 * np.abs(expected[q]).max()
    assert np.nanmax(biased.bias['rms_bias'][q]) > 0


def test_attribution(biased):
    components = flagged(biased)
    networks = val.network_sums(biased.cache, biased.binning, biased.w0, biased.weights)
    shares, total = val.attribute(networks, components, biased.sums['sm_yield'])
    D = biased.bias['D'][components]
    assert np.abs(total - D).max() / np.abs(D).max() < 1e-3
    assert np.allclose(sum(shares.values()), 1)
    top = max(shares, key=lambda n: abs(shares[n][0]))
    assert top == BAD


def test_suspects_drop_a_bad_point(biased):
    networks = val.network_sums(biased.cache, biased.binning, biased.w0, biased.weights)
    shares, _ = val.attribute(networks, flagged(biased), biased.sums['sm_yield'])
    z = val.network_z(networks, biased.sums['sm_yield'], biased.binning)
    assert val.suspects(shares, z, networks) == [BAD]


def test_suspects_few_bad_copies(bad_copies):
    toy = bad_copies
    networks = val.network_sums(toy.cache, toy.binning, toy.w0, toy.weights)
    shares, _ = val.attribute(networks, flagged(toy), toy.sums['sm_yield'])
    z = val.network_z(networks, toy.sums['sm_yield'], toy.binning)
    assert flagged(toy)
    assert val.suspects(shares, z, networks) == []


def test_drop_shortcut_equals_exact_refit(biased):
    groups = {'bad': [BAD], 'cross': CROSS[:2]}
    shortcut = val.drop_bias_sums(biased.cache, groups, biased.binning, biased.w0, biased.weights, biased.sums)
    scale = np.abs(biased.sums['truth']).max()
    for g, members in groups.items():
        refit = val.refit_without(biased.cache, members)
        exact = val.component_sums(refit, biased.binning, biased.w0, biased.weights)['bias']
        assert np.abs(shortcut[g]['bias'] - exact).max() < 1e-5 * scale


def test_drop_tests(biased):
    groups = {'bad': [BAD], 'harmless': ['cB/p_one'], 'every cross point': CROSS}
    sums = val.drop_bias_sums(biased.cache, groups, biased.binning, biased.w0, biased.weights, biased.sums)
    after = {g: None if s is None else val.component_bias(s, biased.binning) for g, s in sums.items()}
    blamed = {g: flagged(biased) for g in groups}
    result = val.test_drops(biased.bias, after, blamed)
    assert result['bad']['accepted'] and result['bad']['gain'] >= 2
    assert not result['harmless']['accepted'] and result['harmless']['gain'] < 2
    # without every cross point the interference term is undetermined: not testable
    assert sums['every cross point'] is None
    assert not result['every cross point']['accepted'] and isinstance(result['every cross point']['harmed'], str)


@pytest.mark.slow
def test_workers_give_identical_results(biased):
    # one spawn pool (network_sums): seed_map's parallel path gives the serial result bit for bit, in seed order
    args = (biased.cache, biased.binning, biased.w0, biased.weights)
    assert equal(val.network_sums(*args, workers=1), val.network_sums(*args, workers=2))
    # the other per-seed functions go through the same seed_map; in a spawn pool they and their arguments are
    # pickled, so the pickled call must give the serial result
    groups = {'bad': [BAD]}
    w0 = torch.as_tensor(biased.w0, dtype=torch.float64)
    weights = torch.as_tensor(biased.weights, dtype=torch.float64)
    calls = [
        (val._refit_seed, (biased.cache.path, [BAD])),
        (val._drop_bias_seed, (biased.cache.path, groups, biased.binning, w0, weights, 50_000)),
    ]
    seed = biased.cache.seeds[0]
    for function, call_args in calls:
        copy, copied_args = pickle.loads(pickle.dumps((function, call_args)))
        assert equal(function(seed, *call_args), copy(seed, *copied_args))


def test_design_uncertainty(null):
    labels, _ = component_labels(WCS, np.ones(6, dtype=bool))
    no_cross = np.array([n not in CROSS for n in null.names])
    unconstrained = val.unconstrained_components(null.Theta[no_cross])
    assert [labels[k] for k in np.flatnonzero(unconstrained)] == ['cB*cA']
    assert not val.unconstrained_components(null.Theta).any()
    # propagated covariance: H^-1 Theta^T diag(m^2) Theta H^-1 for a full-rank design
    m = np.linspace(0.01, 0.02, len(null.Theta))
    _, sigma0, _, _ = val.search_points(null.Theta, m, [], np.eye(6))
    H = np.linalg.inv(null.Theta.T @ null.Theta)
    cov = H @ null.Theta.T @ np.diag(m**2) @ null.Theta @ H
    assert np.allclose(sigma0, np.sqrt(np.diag(cov)))


def test_point_search(null):
    labels, _ = component_labels(WCS, np.ones(6, dtype=bool))
    misclosure = {n: 0.01 for n in null.names}
    m = np.full(len(null.names), 0.01)

    def yield_ratio(theta):
        return 1 + theta[1] ** 2 + theta[2] ** 2

    def units(max_yield):
        return val.candidate_units(WCS, null.Theta, null.names, misclosure, np.ones(6, bool), yield_ratio, max_yield)

    # the existing (+-1, +-1) cross points are not offered again; the cap removes points above it
    offered = units(9.0)
    assert offered and all(u['kind'] == 'cross' and len(u['points']) == 4 for u in offered)
    assert all(max(abs(p['theta'][1]), abs(p['theta'][2])) == 2 for u in offered for p in u['points'])
    assert all(u['max_yield_ratio'] <= 6 for u in units(6.5))

    # the search improves the interference term most with the largest lever arm
    q = labels.index('cB*cA')
    G = np.zeros((6, 6))
    G[q, q] = 1
    chosen, sigma0, sigma1, history = val.search_points(null.Theta, m, offered, G, n_units=1)
    assert len(chosen) == 1 and history[1] < history[0] and sigma1[q] < sigma0[q]
    assert all(abs(p['theta'][1]) == 2 and abs(p['theta'][2]) == 2 for p in chosen[0]['points'])


def test_scan_moments():
    a, b = 0.5, 2.0
    G = val.scan_moments(['cA'], {'cA': a}, np.ones(3, bool))  # columns 1, theta, theta^2
    assert np.allclose(G, [[1, 0, a**2 / 3], [0, a**2 / 3, 0], [a**2 / 3, 0, a**4 / 5]])
    G = val.scan_moments(['cA'], {'cA': a}, np.ones(3, bool), shape='gaussian')
    assert np.isclose(G[1, 1], a**2) and np.isclose(G[2, 2], 3 * a**4)
    # two WCs, columns 1, cA, cA^2, cB, cB*cA, cB^2: varied together (box) or one at a time (1d)
    box = val.scan_moments(WCS, {'cA': a, 'cB': b}, np.ones(6, bool))
    one_d = val.scan_moments(WCS, {'cA': a, 'cB': b}, np.ones(6, bool), mode='1d')
    assert np.isclose(box[2, 5], a**2 / 3 * b**2 / 3) and np.isclose(one_d[2, 5], 0)
    assert np.isclose(one_d[2, 2], a**4 / 5 / 2)


def test_binning():
    x, _, _ = truth()
    features = torch.from_numpy(x).float()
    binning = quantile_binning(features, FEATURES, 10)
    counts = np.bincount(binning['index'][0].numpy(), minlength=binning['n_bins'])[binning['feature'] == 0]
    assert len(counts) == 10 and counts.max() - counts.min() <= 1
    assert binning['discrete'] == [False, False, True] and (binning['feature'] == 2).sum() == 3

    # heavy-tailed weights: merged bins reach the effective number of events, every event stays in one bin
    weights = np.exp(2 * np.random.default_rng(1).standard_normal(N_EVENTS))
    merged = merge_sparse(binning, features, FEATURES, weights, 30)
    for f in range(3):
        index = merged['index'][f].numpy()
        assert (merged['feature'][index] == f).all()
        w = np.bincount(index, weights=weights, minlength=merged['n_bins'])
        w2 = np.bincount(index, weights=weights**2, minlength=merged['n_bins'])
        own = merged['feature'] == f
        assert (w[own] ** 2 / w2[own] >= 30).all()


# ---------------------------------------------------------------------------------------------------------------------
# background caches (--backgrounds): the toy EFT ensemble on two background event sets, and real (untrained) background
# networks with a linear output, loaded through NetworkRatio

BACKGROUNDS = {'tW': 400, 'Others': 600}  # process -> number of holdout events
BG_C0 = [1.0, 0.5, -0.5]  # reference point theta_g of the toy (c0 and every background network's reference_point)
BG_NETWORK = [
    {'type': 'Linear', 'out': 8, 'activation': 'LeakyReLU'},
    {'type': 'Linear', 'in': 8, 'out': 1, 'activation': 'None'},
]


def toy_gamma(x):
    """True per-event coefficients of the toy at features x (the weight formula of truth(), divided by the SM weight)."""
    n = len(x)
    v = np.stack([np.ones(n), 0.5 * x[:, 0] + 0.2, 0.4 * x[:, 1] - 0.3], axis=1)
    a = v[:, :, None] * v[:, None, :] + 0.05 * np.eye(3)
    rows, cols = np.tril_indices(3)
    s = a[:, rows, cols] * np.where(rows == cols, 1.0, 2.0)
    return s / (s @ expanded([1, 0, 0]))[:, None]


def toy_features(n, seed):
    """Features of a toy event set: x0, x1 uniform, x2 an era index 0-2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, 3))
    x[:, 2] = rng.integers(0, 3, n)
    return x


def toy_seeds(root, members=MEMBERS):
    """Fake training configs and model.pt files of the toy ensemble's networks, as config['seeds']."""
    names = list(POINTS) + ['sm']
    seeds = {}
    for e in members:
        paths = []
        for name in names:
            directory = root / f'seed_{e}' / name
            directory.mkdir(parents=True)
            (directory / 'model.pt').write_bytes(f'{name} {e}'.encode())
            (directory / 'training.yml').write_text(yaml.safe_dump({'name': str(directory)}))
            paths.append(str(directory / 'training.yml'))
        seeds[e] = {'networks': paths[:-1], 'sm': paths[-1]}
    return seeds


def toy_cache(path, config, x, mask=None):
    """
    Toy ensemble cache for features x (true ratios with NOISE per member), written through SeedCollector and
    write_cache with config's data and seeds; mask replaces the kept columns written to mask_cache.npy.
    """
    names = list(POINTS) + ['sm']
    Theta = np.stack([expanded((1, *POINTS[n])) for n in names[:-1]] + [expanded((1, 0, 0))])
    R_true = Theta @ toy_gamma(x).T
    collector = SeedCollector(config['seeds'], path)
    Gamma, rank = 0, {}
    for e in config['seeds']:
        rng = np.random.default_rng(200 + e)
        fit = fake_fit(Theta, R_true * (1 + NOISE * rng.standard_normal(R_true.shape)))
        collector(e, fit)
        Gamma, rank[e] = Gamma + fit.Gamma / len(config['seeds']), int(fit.rank)
    ensemble = types.SimpleNamespace(
        Gamma=Gamma,
        mask=torch.ones(Theta.shape[1], dtype=torch.bool) if mask is None else mask,
        finite=torch.ones(len(x), dtype=torch.bool),
        rank=rank,
        background_processes=[],
    )
    write_cache(path, config, ensemble)


def background_dataset(x, seed, negative=0.0):
    """Combined-format holdout of a background (features, coefs, ids, event_id, syst_ratio, w_ref)."""
    n = len(x)
    rng = np.random.default_rng(seed)
    w_ref = rng.uniform(0.5, 1.5, n) * np.where(rng.uniform(size=n) < negative, -1.0, 1.0)
    return torch.utils.data.TensorDataset(
        torch.from_numpy(x).float(),
        torch.zeros(n, 6, dtype=torch.float64),
        torch.zeros(n, 3),
        torch.arange(n),
        torch.zeros(n, 1),
        torch.from_numpy(w_ref)[:, None].float(),
    )


def background_network(directory, process, seed):
    """An untrained background network (linear output) with its training.yml and complete/performance.yml."""
    (directory / 'complete').mkdir(parents=True)
    torch.manual_seed(1000 + seed + (0 if process == 'tW' else 100))
    net = Net(3, 'cpu', BG_NETWORK)
    net.set_feature_normalization(torch.tensor([0.5, 0.5, 1.0]), torch.tensor([0.3, 0.3, 0.8]))
    torch.save(net.state_dict(), directory / 'model.pt')
    (directory / 'training.yml').write_text(yaml.safe_dump({'name': str(directory), 'network': BG_NETWORK}))
    (directory / 'complete' / 'performance.yml').write_text(
        yaml.safe_dump({'process': process, 'reference_point': BG_C0})
    )
    return str(directory / 'training.yml')


@pytest.fixture(scope='module')
def backgrounds(tmp_path_factory):
    """
    Validation config with backgrounds and background_networks, the signal and background ensemble caches written
    from the toy, and the features of every event set.
    """
    root = tmp_path_factory.mktemp('backgrounds')
    x = {'signal': toy_features(N_EVENTS, 0)}
    (root / 'signal').mkdir()
    torch.save(torch.utils.data.TensorDataset(torch.from_numpy(x['signal']).float()), root / 'signal' / 'validation.p')
    directories, networks = {}, {}
    for i, (p, n) in enumerate(BACKGROUNDS.items()):
        x[p] = toy_features(n, 10 + i)
        directory = root / 'backgrounds' / p
        directory.mkdir(parents=True)
        torch.save(background_dataset(x[p], 20 + i, negative=0.2 if p == 'Others' else 0.0), directory / 'validation.p')
        (directory / 'sample.yml').write_text(yaml.safe_dump({'eft': False, 'reference_point': BG_C0}))
        directories[p] = str(directory)
        networks[p] = {e: background_network(root / 'bg_networks' / p / f'seed_{e}', p, e) for e in MEMBERS}
    config = {
        'name': str(root / 'validation'),
        'data': str(root / 'signal' / 'validation.p'),
        'c0': BG_C0,
        'wcs': WCS,
        'lumi': [1.0, 2.0, 3.0],
        'year_loc': 2,
        'device': 'cpu',
        'features': FEATURES,
        'seeds': toy_seeds(root / 'eft_networks'),
        'backgrounds': directories,
        'background_networks': networks,
    }
    toy_cache(f'{config["name"]}/ensemble', config, x['signal'])
    for p, directory in directories.items():
        toy_cache(f'{config["name"]}/ensemble_{p}', {**config, 'data': f'{directory}/validation.p'}, x[p])
    return types.SimpleNamespace(config=config, x=x, root=root)


def background_sets(config):
    return {'signal': config['data'], **{p: f'{d}/validation.p' for p, d in config['backgrounds'].items()}}


def test_network_hash(backgrounds):
    seeds = backgrounds.config['seeds']
    hashes = model_hashes(seeds)
    for e in MEMBERS:
        assert hashes[e] == [network_hash(path) for path in seeds[e]['networks'] + [seeds[e]['sm']]]


def test_load_ensemble_data_override(backgrounds):
    config = backgrounds.config
    data = f'{config["backgrounds"]["tW"]}/validation.p'
    path = f'{config["name"]}/ensemble_tW'
    cache = val.load_ensemble(config, None, data=data, path=path)
    assert cache.manifest['data'] == data and cache.Gamma.shape[1] == BACKGROUNDS['tW']
    assert config['data'] != data  # the caller's config is unchanged
    with pytest.raises(SystemExit, match='other data'):
        val.load_ensemble(config, None, path=path)
    assert val.load_ensemble(config, None).manifest['data'] == config['data']


def test_check_backgrounds(backgrounds, tmp_path):
    config = backgrounds.config
    assert val.check_backgrounds(config) == (list(BACKGROUNDS), MEMBERS)
    with pytest.raises(SystemExit, match='background_networks'):
        val.check_backgrounds({k: v for k, v in config.items() if k != 'background_networks'})
    with pytest.raises(SystemExit, match='names'):
        val.check_backgrounds({**config, 'backgrounds': {'tW': config['backgrounds']['tW']}})
    networks = {**config['background_networks'], 'Others': {0: config['background_networks']['Others'][0]}}
    with pytest.raises(SystemExit, match='seeds'):
        val.check_backgrounds({**config, 'background_networks': networks})
    swapped = {'tW': config['background_networks']['Others'], 'Others': config['background_networks']['tW']}
    with pytest.raises(SystemExit, match="has process 'Others'"):
        val.check_backgrounds({**config, 'background_networks': swapped})
    with pytest.raises(SystemExit, match='reference_point'):
        val.check_backgrounds({**config, 'c0': [1.0, 0.0, 0.0]})
    (tmp_path / 'sample.yml').write_text(yaml.safe_dump({'eft': True}))
    with pytest.raises(SystemExit, match='not a background sample') as error:
        val.check_backgrounds({**config, 'backgrounds': {**config['backgrounds'], 'tW': str(tmp_path)}})
    assert 'no validation.p' in str(error.value)  # every failure is listed


def test_background_ensembles(backgrounds, tmp_path, monkeypatch):
    config = backgrounds.config
    signal = EnsembleCache(f'{config["name"]}/ensemble')
    caches = val.background_ensembles(config, signal)
    assert list(caches) == list(BACKGROUNDS)
    for p, cache in caches.items():
        assert cache.manifest['data'] == f'{config["backgrounds"][p]}/validation.p'
        assert cache.manifest['model_hash'] == signal.manifest['model_hash']
        assert np.array_equal(cache.mask, signal.mask) and cache.finite.shape == (BACKGROUNDS[p],)

    # a stale or missing cache is fitted (load_ensemble with refit) on the background's own features
    calls = []

    def fake_load(config, features, refit=False, data=None, path=None):
        calls.append((refit, None if features is None else tuple(features.shape), data, path))
        return EnsembleCache(path)

    monkeypatch.setattr(val, 'load_ensemble', fake_load)
    val.background_ensembles(config, signal)
    assert all(refit is False and shape is None for refit, shape, _, _ in calls)
    calls.clear()
    val.background_ensembles(config, signal, refit=True)
    assert [shape for _, shape, _, _ in calls] == [(n, 3) for n in BACKGROUNDS.values()]
    assert all(refit for refit, _, _, _ in calls)
    monkeypatch.undo()

    # a background cache that keeps other columns of T(theta) stops the run
    other = {**config, 'name': str(tmp_path / 'validation')}
    mask = torch.ones(6, dtype=torch.bool)
    mask[-1] = False
    for p, directory in config['backgrounds'].items():
        toy_cache(f'{other["name"]}/ensemble_{p}', {**other, 'data': f'{directory}/validation.p'}, backgrounds.x[p], mask)
    with pytest.raises(SystemExit, match='other columns'):
        val.background_ensembles(other, signal)


def test_background_ratios(backgrounds, monkeypatch):
    config = backgrounds.config
    sets = background_sets(config)
    out = val.background_ratios(config, sets, refit=True)
    path = f'{config["name"]}/background_ratios'
    for name, n in [('signal', N_EVENTS), *BACKGROUNDS.items()]:
        assert out[name]['r_hat'].shape == (len(BACKGROUNDS), len(MEMBERS), n)
        assert out[name]['r_hat'].dtype == np.float32 and out[name]['finite'].all()
        assert list(out[name]['processes']) == list(BACKGROUNDS) and list(out[name]['seeds']) == MEMBERS
        features = torch.from_numpy(backgrounds.x[name]).float()
        for i, q in enumerate(BACKGROUNDS):
            for j, e in enumerate(MEMBERS):
                expected = torch.exp(NetworkRatio(config['background_networks'][q][e], 3).log_ratio(features).double())
                assert np.allclose(out[name]['r_hat'][i, j], expected.numpy(), rtol=1e-6)
        with np.load(f'{path}/{name}.npz') as f:
            assert np.array_equal(f['r_hat'], out[name]['r_hat'])
    manifest = yaml.safe_load(open(f'{path}/manifest.yml'))
    assert list(manifest) == ['networks', 'model_hash', 'reference_point', 'data', 'created']
    assert manifest['reference_point'] == BG_C0 and manifest['data'] == sets
    assert manifest['model_hash']['tW'][0] == network_hash(config['background_networks']['tW'][0])

    # a matching cache is read, not evaluated again
    def no_networks(*args, **kwargs):
        raise AssertionError('networks evaluated for a matching cache')

    monkeypatch.setattr(val, 'NetworkRatio', no_networks)
    again = val.background_ratios(config, sets)
    assert np.array_equal(again['Others']['r_hat'], out['Others']['r_hat'])
    monkeypatch.undo()
    stored = val.ratio_manifest(config, sets)
    assert val.ratios_match(path, stored)
    assert not val.ratios_match(path, {**stored, 'reference_point': [1.0, 0.0, 0.0]})
    assert not val.ratios_match(path, {**stored, 'data': {'signal': config['data']}})
    hashes = {**stored['model_hash'], 'tW': {**stored['model_hash']['tW'], 0: 'retrained'}}
    assert not val.ratios_match(path, {**stored, 'model_hash': hashes})


def test_background_diagnostics(backgrounds):
    config = backgrounds.config
    signal = EnsembleCache(f'{config["name"]}/ensemble')
    caches = val.background_ensembles(config, signal)
    ratios = val.background_ratios(config, background_sets(config))
    diagnostics = val.background_diagnostics(config, caches, ratios)
    assert list(diagnostics['backgrounds']) == list(BACKGROUNDS) and diagnostics['seeds'] == MEMBERS
    for p, entry in diagnostics['backgrounds'].items():
        assert entry['events'] == BACKGROUNDS[p]
        assert entry['dropped'] == {'ensemble': 0.0, 'ratios': 0.0, 'either': 0.0}
        for key in ('theta_sm', 'theta_g'):
            assert entry['nonpositive'][key]['mean'] == 0.0 and list(entry['nonpositive'][key]['members']) == MEMBERS
        tensors = torch.load(f'{config["backgrounds"][p]}/validation.p', weights_only=False).tensors
        w = (torch.tensor(config['lumi'])[tensors[0][:, 2].long()] * tensors[5][:, 0] * 1000).double().numpy()
        r_sm = expanded([1, 0, 0]) @ np.asarray(caches[p].Gamma, dtype=np.float64)
        assert np.isclose(entry['yield']['sum_w'], w.sum(), rtol=1e-5)
        assert np.isclose(entry['yield']['ratio'], np.sum(w * r_sm) / w.sum(), rtol=1e-5)
        assert np.isclose(entry['yield']['ratio'], 1.0, rtol=0.01)  # the toy's r_hat(theta_SM) = 1 up to NOISE
    for q in BACKGROUNDS:
        assert list(diagnostics['seed_spread'][q]) == ['signal', q]
        r = ratios[q]['r_hat'][list(BACKGROUNDS).index(q)].astype(np.float64)
        assert np.isclose(diagnostics['seed_spread'][q][q]['median'], np.median(r.std(axis=0) / r.mean(axis=0)))


def test_run_backgrounds(backgrounds, tmp_path):
    config = backgrounds.config
    path = tmp_path / 'validation.yml'
    path.write_text(yaml.safe_dump(config))
    val.main(str(path), backgrounds=True)
    diagnostics = yaml.safe_load(open(f'{config["name"]}/background_diagnostics.yml'))
    assert list(diagnostics) == ['backgrounds', 'seed_spread', 'seeds']
    assert not (backgrounds.root / 'validation' / 'plots').exists()  # the bias test did not run
    with pytest.raises(SystemExit, match="'signal'"):
        val.run_backgrounds({**config, 'backgrounds': {'signal': config['backgrounds']['tW']}})


# background columns after the expansion, T(theta, mu) = [T(theta), mu_bg] (one process, bg)
BG = ['bg']
MASK_BG = np.ones(7, dtype=bool)


def test_component_labels_background():
    labels, kinds = component_labels(WCS, MASK_BG, BG)
    assert labels == ['sm', 'cA', 'cA^2', 'cB', 'cB*cA', 'cB^2', 'mu_bg']
    assert kinds[-1] == 'background' and component_kinds(labels) == kinds
    blind = MASK_BG.copy()
    blind[-1] = False
    assert component_labels(WCS, blind, BG)[0] == labels[:-1]
    with pytest.raises(ValueError):
        component_labels(WCS, MASK_BG)
    assert component_tex('mu_bg') == r'$\mu_{\rm bg}$'


def test_network_and_unit_tex_background():
    points = {
        'cA/p_one': [1.0, 1.0, 0.0, 1.0],
        'cA_cB/p_one_n_one': [1.0, 1.0, -1.0, 1.0],
        'bg/mu_0': [1.0, 0.5, -0.5, 0.0],
        'bg/mu_2': [1.0, 0.5, -0.5, 2.0],
        'sm': [1.0, 0.0, 0.0, 1.0],
    }
    Theta = np.stack([expansion(p, 2, dtype=torch.float64).numpy() for p in points.values()])
    a, b = wc_tex('cA'), wc_tex('cB')
    expected = {
        'cA/p_one': f'${a} = 1.00$',
        'cA_cB/p_one_n_one': f'${a}, {b} = 1.00, -1.00$',
        'bg/mu_0': r'$\theta_g, \mu_{\rm bg} = 0.00$',
        'bg/mu_2': r'$\theta_g, \mu_{\rm bg} = 2.00$',
        'sm': 'sm',
    }
    assert network_tex(Theta, list(points), WCS, MASK_BG, BG, [1.0, 0.5, -0.5, 1.0]) == expected
    # without backgrounds or reference the labels are the WC values, as before
    plain = network_tex(Theta[:, :6], list(points), WCS, np.ones(6, dtype=bool))
    assert plain['cA/p_one'] == expected['cA/p_one'] and plain['bg/mu_0'] == f'${a}, {b} = 0.50, -0.50$'
    # an SM reference never labels a point theta_g
    assert network_tex(Theta, list(points), WCS, MASK_BG, BG, [1.0, 0.0, 0.0, 1.0])['bg/mu_0'] == (
        f'${a}, {b}, \\mu_{{\\rm bg}} = 0.50, -0.50, 0.00$'
    )
    assert unit_tex({'points': [{'theta': [1.0, 2.0, 0.0, 1.0]}]}, WCS) == f'${a} = 2.00$'
    # a point without signal reads 'b only'; the SM at another strength keeps 'sm' as its head; labels shared by
    # networks of different groups get the group, unique ones do not
    more = {
        'bg/only': [0.0, 0.0, 0.0, 1.0],
        'nominal/sm': [1.0, 0.0, 0.0, 0.0],
        'other:sm': [1.0, 0.0, 0.0, 1.0],
    }
    rows = np.stack([expansion(p, 2, dtype=torch.float64).numpy() for p in more.values()])
    Theta_more, names = np.vstack([Theta, rows]), list(points) + list(more)
    reference = [1.0, 0.5, -0.5, 1.0]
    plain = network_tex(Theta_more, names, WCS, MASK_BG, BG, reference)
    assert plain['bg/only'] == '$b$ only'
    assert plain['nominal/sm'] == r'${\rm sm}, \mu_{\rm bg} = 0.00$'
    assert plain['sm'] == plain['other:sm'] == 'sm'
    groups = ['first'] * len(points) + ['second'] * len(more)
    grouped = network_tex(Theta_more, names, WCS, MASK_BG, BG, reference, groups)
    assert grouped['sm'] == 'sm (first)' and grouped['other:sm'] == 'sm (second)'
    unique = [n for n in names if n not in ('sm', 'other:sm')]
    assert [grouped[n] for n in unique] == [plain[n] for n in unique]
    # a strength every WC point of a group shares is the group's and is left out, so the same point in two groups
    # reads the same, with the group; theta_g points keep their strength
    signal = {'signal:cA/p_one': [1.0, 1.0, 0.0, 0.0], 'signal:cA_cB/p_one_n_one': [1.0, 1.0, -1.0, 0.0]}
    rows = np.stack([expansion(p, 2, dtype=torch.float64).numpy() for p in signal.values()])
    both = network_tex(
        np.vstack([Theta, rows]),
        list(points) + list(signal),
        WCS,
        MASK_BG,
        BG,
        reference,
        ['mixed'] * len(points) + ['signal'] * len(signal),
    )
    assert both['signal:cA/p_one'] == f'${a} = 1.00$ (signal)' and both['cA/p_one'] == f'${a} = 1.00$ (mixed)'
    assert both['bg/mu_0'] == expected['bg/mu_0']
    # without groups the two strengths are not shared, so the mu_bg = 0 points keep theirs
    ungrouped = network_tex(np.vstack([Theta, rows]), list(points) + list(signal), WCS, MASK_BG, BG, reference)
    assert ungrouped['signal:cA/p_one'] == f'${a}, \\mu_{{\\rm bg}} = 1.00, 0.00$'
    # kinds from the rows of Theta: no signal is background, no WC is sm (under any group prefix or strength)
    assert network_kinds(Theta_more, names, WCS, MASK_BG) == {
        'cA/p_one': 'scan',
        'cA_cB/p_one_n_one': 'cross',
        'bg/mu_0': 'scan',
        'bg/mu_2': 'scan',
        'sm': 'sm',
        'bg/only': 'background',
        'nominal/sm': 'sm',
        'other:sm': 'sm',
    }


def test_scan_moments_background():
    bounds = {'cA': 0.5, 'cB': 2.0}
    plain = val.scan_moments(WCS, bounds, np.ones(6, bool))
    G = val.scan_moments(WCS, bounds, MASK_BG, n_background=1)
    # mu_bg = 1 is a constant, like the SM entry
    assert np.allclose(G[:6, :6], plain) and np.allclose(G[6, :6], plain[0]) and np.isclose(G[6, 6], 1)
    blind = MASK_BG.copy()
    blind[-1] = False
    assert np.allclose(val.scan_moments(WCS, bounds, blind, n_background=1), plain)
    weights = val.objective_weights({'G': 'scan', 'bounds': bounds}, WCS, MASK_BG, [], None, None, 1)
    assert np.allclose(weights, G)


def test_candidate_units_background(null):
    misclosure = {n: 0.01 for n in null.names}

    def yield_ratio(theta):
        return 1 + theta[1] ** 2 + theta[2] ** 2

    plain = val.candidate_units(WCS, null.Theta, null.names, misclosure, np.ones(6, bool), yield_ratio, 9.0)
    Theta = np.hstack([null.Theta, np.ones((len(null.Theta), 1))])  # every network at mu_bg = 1
    units = val.candidate_units(WCS, Theta, null.names, misclosure, MASK_BG, yield_ratio, 9.0, BG)
    cross = [u for u in units if u['kind'] == 'cross']
    assert len(cross) == len(units) == len(plain)
    for u, v in zip(cross, plain):
        assert [p['theta'] for p in u['points']] == [p['theta'] + [1.0] for p in v['points']]
        assert np.allclose(u['rows'], np.hstack([v['rows'], np.ones((len(v['rows']), 1))]))


def background_column_toy(root, moved, background_fit=False):
    """
    Validation config and cache of a toy with one background column: the toy signal of truth() plus a background
    b(x) on the same events; members fitted at the toy points (mu_bg = 1), the SM point and, when moved, at
    mu_bg = 0 and 2, against theta_0 = (SM, mu_bg = 1). With background_fit, the networks come in two groups:
    'mixed' (the toy points and the SM point at mu_bg = 1) and 'signal' (the toy points at mu_bg = 0 and a
    background-only point), the SM network is the signal group's 'bg/sm', and the background column comes from the
    signal group and the SM network alone (config background_fit).
    """
    x, weights, _ = truth()
    b = 0.3 * weights[:, 0] * (1 + x[:, 0])
    coefs = np.hstack([weights, b[:, None]]) / 1000
    c0 = [1.0, 0.0, 0.0, 1.0]
    if background_fit:
        groups = {
            'mixed': {**{n: (1.0, *POINTS[n], 1.0) for n in POINTS}, 'sm': tuple(c0)},
            'signal': {**{n: (1.0, *POINTS[n], 0.0) for n in POINTS}, 'bg/only': (0.0, 0.0, 0.0, 1.0)},
        }
        sm_slot = ('signal', 'bg/sm')
    else:
        points = {n: (1.0, *POINTS[n], 1.0) for n in POINTS}
        if moved:
            points.update({'bg/mu_0': (1.0, 0.0, 0.0, 0.0), 'bg/mu_2': (1.0, 0.0, 0.0, 2.0)})
        groups = {None: points}
        sm_slot = (None, 'sm')
    rows = [(g, n, p) for g, points in groups.items() for n, p in points.items()] + [(*sm_slot, tuple(c0))]
    Theta = np.stack([expansion(list(p), 2, dtype=torch.float64).numpy() for _, _, p in rows])
    mask = np.ones(7, dtype=bool)
    mask[-1] = moved or background_fit
    full = np.hstack([weights, b[:, None]])
    R_true = Theta @ (full / (full @ expansion(c0, 2, dtype=torch.float64).numpy())[:, None]).T
    Theta = Theta[:, mask]
    root.mkdir(parents=True, exist_ok=True)
    data = root / 'data.p'
    torch.save(torch.utils.data.TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(coefs)), data)
    seeds = {}
    for e in MEMBERS:
        paths = []
        for g, name, _ in rows:
            directory = root / f'seed_{e}' / (name if g is None else f'{g}/{name}')
            directory.mkdir(parents=True)
            (directory / 'model.pt').write_bytes(f'{g} {name} {e}'.encode())
            (directory / 'training.yml').write_text(yaml.safe_dump({'name': str(directory)}))
            paths.append(str(directory / 'training.yml'))
        if background_fit:
            networks = {g: [p for (h, _, _), p in zip(rows[:-1], paths[:-1]) if h == g] for g in groups}
        else:
            networks = paths[:-1]
        seeds[e] = {'networks': networks, 'sm': paths[-1]}
    config = {
        'name': str(root / 'validation'),
        'data': str(data),
        'c0': c0,
        'wcs': WCS,
        'lumi': [1.0, 1.0, 1.0],
        'year_loc': 2,
        'device': 'cpu',
        'features': {name: {**spec, 'label': name} for name, spec in FEATURES.items()},
        'seeds': seeds,
        'bias': {'n_bins': 10},
    }
    background_rows = None
    if background_fit:
        config['background_fit'] = 'signal'
        background_rows = np.array([g == 'signal' for g, _, _ in rows[:-1]] + [True])
    path = f'{config["name"]}/ensemble'
    collector = SeedCollector(seeds, path)
    Gamma, rank = 0, {}
    for e in MEMBERS:
        rng = np.random.default_rng(300 + e)
        fit = fake_fit(Theta, R_true * (1 + NOISE * rng.standard_normal(R_true.shape)), background_rows)
        collector(e, fit)
        Gamma, rank[e] = Gamma + fit.Gamma / len(MEMBERS), int(fit.rank)
    ensemble = types.SimpleNamespace(
        Gamma=Gamma,
        mask=torch.from_numpy(mask),
        finite=torch.ones(N_EVENTS, dtype=torch.bool),
        rank=rank,
        background_processes=BG,
    )
    write_cache(path, config, ensemble)
    return config


def test_cache_background_column(tmp_path):
    config = background_column_toy(tmp_path, moved=True)
    cache = EnsembleCache(f'{config["name"]}/ensemble')
    assert cache.background_processes == BG and cache.n_wcs == 2
    point = [1.0, 0.5, -0.5, 0.7]
    assert np.allclose(cache(point), expansion(point, 2, dtype=torch.float64).numpy() @ cache.Gamma)
    blind = EnsembleCache(f'{background_column_toy(tmp_path / "blind", moved=False)["name"]}/ensemble')
    assert np.allclose(blind(point[:3] + [1.0]), expansion(point[:3], 2, dtype=torch.float64).numpy() @ blind.Gamma)
    with pytest.raises(ValueError, match='no network moved'):
        blind(point)
    # a cache written before background_processes was recorded reads as having none
    manifest = yaml.safe_load(open(f'{config["name"]}/ensemble/ensemble.yml'))
    del manifest['background_processes']
    with open(f'{config["name"]}/ensemble/ensemble.yml', 'w') as f:
        yaml.safe_dump(manifest, f)
    assert EnsembleCache(f'{config["name"]}/ensemble').background_processes == []


@pytest.mark.parametrize(
    'moved, background_fit', [(True, False), (False, False), (True, True)], ids=['moved', 'blind', 'background_fit']
)
def test_main_background_column(tmp_path, moved, background_fit):
    # the whole validation on a background column; in the blind fit the SM column holds b, and closes only
    # because its true weights include b too; with background_fit, the two-stage fit of two groups of networks
    config = background_column_toy(tmp_path, moved, background_fit)
    path = tmp_path / 'validation.yml'
    path.write_text(yaml.safe_dump(config))
    val.main(str(path))
    diagnostics = yaml.safe_load(open(f'{config["name"]}/ensemble_diagnostics.yml'))
    assert ('mu_bg' in diagnostics['components']) == moved
    if moved:
        assert diagnostics['components']['mu_bg']['kind'] == 'background'
    assert diagnostics['flagged'] == []
    proposed = yaml.safe_load(open(f'{config["name"]}/proposed_points.yml'))
    assert proposed['background_processes'] == BG
    assert all(len(p['theta']) == 4 and p['theta'][3] == 1.0 for p in proposed['points'])


def test_background_fit_drops_and_covariance(tmp_path):
    config = background_column_toy(tmp_path, moved=True, background_fit=True)
    cache = EnsembleCache(f'{config["name"]}/ensemble')
    features, coefs = torch.load(config['data'], weights_only=False).tensors
    weights = coefs.double() * 1000
    w0 = weights @ expansion(config['c0'], 2, dtype=torch.float64)
    binning = merge_sparse(quantile_binning(features, FEATURES, 10), features, FEATURES, w0, 30)
    sums = val.component_sums(cache, binning, w0, weights)
    # the downdated normal equations of both stages equal an exact refit, for a network outside the background rows
    # and one inside them
    groups = {'outside': ['mixed:cA/p_two'], 'inside': ['signal:cA/p_two']}
    shortcut = val.drop_bias_sums(cache, groups, binning, w0, weights, sums)
    scale = np.abs(sums['truth']).max()
    for g, members in groups.items():
        exact = val.component_sums(val.refit_without(cache, members), binning, w0, weights)['bias']
        assert np.abs(shortcut[g]['bias'] - exact).max() < 1e-5 * scale

    # Cov(gamma_hat) = Theta^- M^2 Theta^-T, from H and S; a new point joins the fit of the other columns, so the
    # additive update of H and S gives the covariance of the larger design (batched like the point search)
    network = cache.network(cache.seeds[0])
    Theta, rows = network['Theta'].astype(np.float64), network['background_rows']
    m = np.random.default_rng(0).uniform(0.5, 2.0, len(Theta))
    H, S = Theta.T @ Theta, Theta.T @ (m[:, None] ** 2 * Theta)
    fixed = val.fixed_background(Theta, m, rows, 1)
    inverse = val.seed_inverse(cache, network)
    assert np.allclose(val.design_covariance(H, S, fixed), inverse @ np.diag(m**2) @ inverse.T, rtol=1e-6, atol=1e-12)
    u = expansion([1.0, 0.5, 1.5, 1.0], 2, dtype=torch.float64).numpy()[None]
    larger = generalized_inverse(torch.from_numpy(np.vstack([Theta, u])), np.r_[rows, False], 1).numpy()
    expected = larger @ np.diag(np.r_[m, 1.0] ** 2) @ larger.T
    stacked = val.design_covariance(np.stack([H, H + u.T @ u]), np.stack([S, S + u.T @ u]), fixed)
    assert np.allclose(stacked[1], expected, rtol=1e-6, atol=1e-12)
    # least squares over every row: H^+ S H^+ = Theta^+ M^2 Theta^+T
    plain = np.linalg.pinv(Theta)
    assert np.allclose(val.design_covariance(H, S), plain @ np.diag(m**2) @ plain.T, rtol=1e-6, atol=1e-12)
