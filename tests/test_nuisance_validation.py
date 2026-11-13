"""
Nuisance validation on a small synthetic production built from the samples of conftest: a mixed event set (the
signal events of make_combined_sample followed by the events of make_background_sample, process bg) with its EFT
reference c0 = (theta_SM, mu_bg = 1); signal nuisance networks trained on the signal sample (ISR as a weight_shift,
JER_2017 as a kinematic_shift whose variation exists in era 2 only) and a background nuisance network trained on the
background sample (ISR as a weight_shift, named ISR_bg). Every network is untrained, two members per direction. The
nuisance cache is built and the validation run once per module, against a stand-in EFT cache whose Gamma gives the
exact background term and a signal term off by about 1% per event; the tests read its outputs.
"""

import os
import shutil
import types

import numpy as np
import pytest
import torch
import yaml
from conftest import N_FEATURES, N_WCS, WCS, make_background_sample, make_combined_sample

import topsbi.nuisance_validation as nv
from topsbi.model.net import Net
from topsbi.tools.build_ratio import NetworkRatio
from topsbi.tools.data import expansion

MEMBERS = [0, 1]
N_EVENTS = 1000  # train events per sample; the validation.p of each holds half
N_Q = (N_WCS + 1) * (N_WCS + 2) // 2
SM = [1.0] + [0.0] * N_WCS
C0 = SM + [1.0]  # the EFT reference: the reference_point of both samples, then mu_bg
POINTS = {'sm': SM, 'theta': [1.0, 0.3, -0.2] + [0.0] * (N_WCS - 2)}  # factorisation points
LUMI = [19.52, 16.81, 41.48, 59.83]
FEATURES = {
    'feat0': {'label': 'feature 0', 'loc': 0},
    'feat1': {'label': 'feature 1', 'loc': 1},
    'year': {'label': 'era', 'loc': 21},
}
# (term, systematic, method) of every network; the cache rows are sorted by name
NETWORKS = [('signal', 'ISR', 'weight_shift'), ('signal', 'JER_2017', 'kinematic_shift'), ('bg', 'ISR', 'weight_shift')]
ROWS = {'ISR': 0, 'ISR_bg': 2, 'JER_2017': 1}  # cache row name: index into NETWORKS
NETWORK = [
    {'type': 'Linear', 'out': 8, 'activation': 'LeakyReLU'},
    {'type': 'Linear', 'in': 8, 'out': 1, 'activation': 'None'},
]


def nuisance_network(directory, data, systematic, direction, method, seed):
    """An untrained nuisance network (linear output) with its training.yml and complete/performance.yml."""
    (directory / 'complete').mkdir(parents=True)
    torch.manual_seed(seed)
    net = Net(N_FEATURES, 'cpu', NETWORK)
    net.set_feature_normalization(torch.full((N_FEATURES,), 100.0), torch.full((N_FEATURES,), 58.0))
    torch.save(net.state_dict(), directory / 'model.pt')
    training = {
        'name': str(directory),
        'network': NETWORK,
        'data': str(data),
        'systematic': systematic,
        'direction': direction,
        'method': method,
    }
    (directory / 'training.yml').write_text(yaml.safe_dump(training))
    (directory / 'complete' / 'performance.yml').write_text(yaml.safe_dump({}))
    return str(directory / 'training.yml')


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    """
    The signal, background and mixed samples, the networks of NETWORKS for every member and the EFT validation config
    of the mixed sample (eft.yml).
    """
    root = tmp_path_factory.mktemp('nuisance_validation')
    data = {
        'signal': make_combined_sample(root / 'signal', N_EVENTS),
        'bg': make_background_sample(root / 'bg', N_EVENTS, process='bg'),
    }
    # the era-2 variation in both directions at the signal's reference point, and the background sample at it too
    shutil.copytree(data['signal'] / 'JER_2017Up', data['signal'] / 'JER_2017Down')
    for direction in ('Up', 'Down'):
        variation = {'variation': f'JER_2017{direction}', 'reference_point': SM}
        (data['signal'] / f'JER_2017{direction}' / 'sample.yml').write_text(yaml.safe_dump(variation))
    sample = yaml.safe_load((data['bg'] / 'nominal' / 'sample.yml').read_text())
    (data['bg'] / 'nominal' / 'sample.yml').write_text(yaml.safe_dump({**sample, 'reference_point': SM}))
    # the mixed sample: the signal events, then the background events, with one background column after T(theta)
    signal = list(torch.load(data['signal'] / 'nominal' / 'validation.p', weights_only=False).tensors)
    background = list(torch.load(data['bg'] / 'nominal' / 'validation.p', weights_only=False).tensors)
    signal[1] = torch.cat([signal[1], torch.zeros(len(signal[1]), 1, dtype=torch.float64)], dim=1)
    background[1] = torch.cat(
        [torch.zeros(len(background[1]), N_Q, dtype=torch.float64), background[5].double()], dim=1
    )
    mixed = root / 'mixed' / 'nominal'
    mixed.mkdir(parents=True)
    tensors = [torch.cat([a, b]) for a, b in zip(signal, background)]
    torch.save(torch.utils.data.TensorDataset(*tensors), mixed / 'validation.p')
    for name in ('sample.yml', 'systematics.yml'):
        shutil.copy(data['signal'] / 'nominal' / name, mixed / name)
    seeds = {}
    for e in MEMBERS:
        networks = []
        for k, (term, systematic, method) in enumerate(NETWORKS):
            for d, direction in enumerate(('down', 'up')):
                directory = root / 'networks' / f'seed_{e}' / f'{term}_{systematic}' / direction
                seed = 100 * e + 10 * k + d
                networks.append(nuisance_network(directory, data[term], systematic, direction, method, seed))
        seeds[e] = {'networks': networks}
    eft = {
        'name': str(root / 'eft'),
        'data': str(mixed / 'validation.p'),
        'c0': C0,
        'wcs': WCS,
        'lumi': LUMI,
        'features': FEATURES,
    }
    eft_path = root / 'eft.yml'
    eft_path.write_text(yaml.safe_dump(eft, sort_keys=False))
    return types.SimpleNamespace(
        root=root, eft=eft, eft_path=str(eft_path), seeds=seeds, data=data, tensors=tensors, n_signal=len(signal[0])
    )


def write_config(toy, directory, eft_path=None, **extra):
    """Nuisance validation config yml in directory (name directory/validation), against eft_path (default toy's)."""
    config = {
        'name': str(directory / 'validation'),
        'ensemble': eft_path or toy.eft_path,
        'seeds': toy.seeds,
        'device': 'cpu',
        'workers': 1,
        **extra,
    }
    path = directory / 'nuisance.yml'
    path.write_text(yaml.safe_dump(config))
    return str(path)


def member_mean(toy, features, row):
    """[2, N] member mean of the networks' f_hat (Down, Up) of cache row row on the events features, in float64."""
    k = ROWS[row]
    f_hat = [
        [
            NetworkRatio(toy.seeds[e]['networks'][2 * k + d], N_FEATURES).log_ratio(features).double().numpy()
            for d in (0, 1)
        ]
        for e in MEMBERS
    ]
    return np.mean(f_hat, axis=0)


def stand_in_eft(coefs, noise=0.0):
    """
    A stand-in for EnsembleCache: Gamma = s_q(z) / s(z) . T(c0) per event for every column of T(theta, mu), the exact
    ratio, with the signal columns of each event scaled by 1 + epsilon_i (epsilon_i normal with standard deviation
    noise), and two members scattered around it by 1%.
    """
    reference = coefs @ expansion(C0, N_WCS, dtype=torch.float64)
    gamma = (coefs / reference[:, None]).T.numpy()
    n = gamma.shape[1]
    gamma[:N_Q] *= 1 + noise * np.random.default_rng(7).standard_normal(n)
    return types.SimpleNamespace(
        path='stand-in',
        n_wcs=N_WCS,
        background_processes=['bg'],
        mask=np.ones(len(gamma), dtype=bool),
        Gamma=gamma,
        finite=np.ones(n, dtype=bool),
        seeds=MEMBERS,
        seed_Gamma={e: gamma * (1 + 0.01 * (2 * e - 1)) for e in MEMBERS},
        seed_finite={e: np.ones(n, dtype=bool) for e in MEMBERS},
        matches=lambda config: True,
    )


@pytest.fixture(scope='module')
def run(toy, tmp_path_factory):
    """
    The validation of toy, run once with a fresh nuisance cache against the stand-in EFT cache with a 1% signal term
    error: config and cache.
    """
    directory = tmp_path_factory.mktemp('run')
    path = write_config(toy, directory, factorisation={'points': POINTS})
    coefs = toy.tensors[1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nv, 'EnsembleCache', lambda path: stand_in_eft(coefs, noise=0.01))
        nv.main(path, refit=True)
    config = nv.load_config(path)
    return types.SimpleNamespace(path=path, config=config, cache=nv.NuisanceCache(f'{config["name"]}/nuisance'))


def moved_networks(toy, directory, e, k, data):
    """toy's seeds with both networks k (index into NETWORKS) of member e pointing to the data directory data."""
    seeds = {m: {'networks': list(s['networks'])} for m, s in toy.seeds.items()}
    for d in (0, 1):
        with open(seeds[e]['networks'][2 * k + d]) as f:
            training = yaml.safe_load(f)
        path = directory / f'{data.name}_{e}_{k}_{d}.yml'
        path.write_text(yaml.safe_dump({**training, 'data': str(data)}))
        seeds[e]['networks'][2 * k + d] = str(path)
    return seeds


def test_nuisance_networks_terms(toy, tmp_path):
    nuisances, networks = nv.nuisance_networks(toy.seeds)
    # the background network of the same systematic as a signal one is its own row, named <systematic>_<process>
    assert [nuisance['name'] for nuisance in nuisances] == list(ROWS)
    for nuisance in nuisances:
        term, systematic, method = NETWORKS[ROWS[nuisance['name']]]
        assert nuisance == {
            'name': nuisance['name'],
            'systematic': systematic,
            'method': method,
            'term': term,
            'data': str(toy.data[term]),
        }
    for e in MEMBERS:
        assert networks[e] == [toy.seeds[e]['networks'][2 * k : 2 * k + 2] for k in ROWS.values()]

    # a background sample without its process, and one network trained on two data directories, stop
    unnamed = tmp_path / 'unnamed'
    (unnamed / 'nominal').mkdir(parents=True)
    (unnamed / 'nominal' / 'sample.yml').write_text(yaml.safe_dump({'eft': False}))
    with pytest.raises(ValueError, match='eft: false and no process'):
        nv.nuisance_networks(moved_networks(toy, tmp_path, 0, ROWS['ISR_bg'], unnamed))
    other = tmp_path / 'other'
    (other / 'nominal').mkdir(parents=True)
    (other / 'nominal' / 'sample.yml').write_text(yaml.safe_dump({'eft': True}))
    with pytest.raises(ValueError, match='is trained on both'):
        nv.nuisance_networks(moved_networks(toy, tmp_path, 1, ROWS['ISR'], other))


def test_nuisance_cache(toy, run, tmp_path):
    cache = run.cache
    n = len(toy.tensors[0])
    assert cache.manifest['data'] == toy.eft['data'] and np.allclose(cache.theta_ref, SM)
    assert [nuisance['name'] for nuisance in cache.nuisances] == list(ROWS)
    assert [nuisance['eras'] for nuisance in cache.nuisances] == [None, None, [2]]
    assert cache.f_hat.shape == (len(ROWS), 2, n) and cache.finite.all()
    # every row on every event of the mixed sample, the background events included
    features = toy.tensors[0]
    for row in ('ISR', 'ISR_bg'):
        assert np.allclose(cache.f_hat[list(ROWS).index(row)], member_mean(toy, features, row), rtol=1e-5, atol=1e-6)
    # the kinematic_shift with a variation in era 2 only is 0 in the other eras
    era2 = (features[:, 21] == 2).numpy()
    jer = cache.f_hat[list(ROWS).index('JER_2017')]
    assert np.all(jer[:, ~era2] == 0)
    assert np.allclose(jer[:, era2], member_mean(toy, features[era2], 'JER_2017'), rtol=1e-5, atol=1e-6)

    # c0 at another WC point than the networks' theta_ref stops the run, on reading and before building
    other = {**run.config, 'c0': POINTS['theta'] + [1.0]}
    with pytest.raises(SystemExit, match='share the reference point'):
        nv.load_nuisance_ensemble(other, None)
    rebuilt = {**other, 'name': str(tmp_path / 'validation')}
    with pytest.raises(SystemExit, match='share the reference point'):
        nv.load_nuisance_ensemble(rebuilt, features, refit=True)
    assert not (tmp_path / 'validation').exists()


def test_event_terms_and_binning(toy):
    coefs, features = toy.tensors[1], toy.tensors[0]
    terms = nv.event_terms(coefs, N_WCS, ['bg'])
    assert list(terms[: toy.n_signal]) == ['signal'] * toy.n_signal
    assert list(terms[toy.n_signal :]) == ['bg'] * (len(terms) - toy.n_signal)
    # without background columns every event is signal
    assert set(nv.event_terms(coefs[:, :N_Q], N_WCS, [])) == {'signal'}
    with pytest.raises(ValueError, match='columns but coefs has'):
        nv.event_terms(coefs, N_WCS, [])
    with pytest.raises(ValueError, match='another background process'):
        nv.event_terms(torch.cat([coefs, coefs[:, N_Q:]], dim=1), N_WCS, ['bg', 'copy'])

    # the shared bins hold min_neff effective events of every event together and of every term in every feature
    w0 = nv.sample_weight(toy.tensors, 'nominal', {'lumi': LUMI}).double().numpy()
    binning = nv.closure_binning(features, FEATURES, w0, terms, nbins=20, min_neff=30)
    index = binning['index'].numpy()
    for rows in (np.ones(len(terms), dtype=bool), terms == 'signal', terms == 'bg'):
        for f in range(len(binning['names'])):
            sumw = np.bincount(index[f, rows], w0[rows], binning['n_bins'])
            sumw2 = np.bincount(index[f, rows], w0[rows] ** 2, binning['n_bins'])
            bins = binning['feature'] == f
            assert np.all(sumw[bins] ** 2 / sumw2[bins] >= 30)


def test_closure_targets(toy, run):
    config, cache = run.config, run.cache
    columns = ('features', 'ids', 'event_id', 'syst_ratio', 'w_ref')
    nominal = nv.load_combined(config['data'], 'nominal', columns)
    features = nominal[0]
    terms = nv.event_terms(toy.tensors[1], N_WCS, ['bg'])
    w0 = nv.sample_weight(nominal, 'nominal', config).double().numpy()
    binning = nv.closure_binning(features, FEATURES, w0, terms)
    targets = nv.ClosureTargets(config, cache, nominal, w0, binning, terms)
    row = {name: list(ROWS).index(name) for name in ROWS}
    signal, era2 = terms == 'signal', (features[:, 21] == 2).numpy()
    events = np.arange(len(w0))

    # each nuisance is tested on the events of its term, in its eras
    assert np.array_equal(targets.rows0(row['ISR']), signal)
    assert np.array_equal(targets.rows0(row['ISR_bg']), ~signal)
    assert np.array_equal(targets.rows0(row['JER_2017']), signal & era2)
    assert np.array_equal(targets.era_rows(row['JER_2017']), era2)

    # weight_shift: the true ratio on the events of the nuisance's term, 1 on the others
    for name in ('ISR', 'ISR_bg'):
        own = terms == NETWORKS[ROWS[name]][0]
        for d, direction in enumerate(nv.DIRECTIONS):
            k, j = nv.syst_ratio_column(str(toy.root / 'mixed'), 'ISR', direction)
            r = nominal[4][:, k, j].double().numpy()
            t = targets(row[name], d)
            assert np.allclose(t['w1'], w0 * np.where(own, r, 1.0))
            assert np.array_equal(t['i0'], events) and np.array_equal(t['i1'], events)
            assert torch.equal(t['index1'], binning['index'])

    # kinematic_shift: the variation of the networks' training data, matched to the events of its term only
    keys = nv.event_keys(nominal, 'nominal', 21)
    for d, direction in enumerate(nv.DIRECTIONS):
        path = f'{toy.data["signal"]}/JER_2017{direction.capitalize()}/validation.p'
        assert targets._path(row['JER_2017'], d) == path
        shifted = nv.load_combined(path, 'variation', ('features', 'ids', 'event_id', 'w_ref'))
        i0, i1 = nv.matched_events(keys[signal], nv.event_keys(shifted, 'variation', 21))
        t = targets(row['JER_2017'], d)
        assert np.all(t['i0'][~signal] == -1) and np.array_equal(t['i0'][signal], i0.numpy())
        assert np.array_equal(t['i1'], i1.numpy())
        assert np.allclose(t['w1'], nv.sample_weight(shifted, 'variation', config).double().numpy())
    down, up = targets.pair(row['JER_2017'])
    assert np.array_equal(down['i0'], up['i0']) and np.all(down['i0'][~signal] == -1)
    # the Down sample is a copy of Up, so both give every shifted event the index of its nominal event
    assert np.array_equal(down['i1'], up['i1'])

    # the shift of a part counts every event once: for weight_shift, (w_Up + w_Down) / 2 - w0 for the symmetric part
    # and (w_Up - w_Down) / 2 for the slope, per event
    parts = nv.pair_sums(cache, binning, w0, targets)[row['ISR']]
    rows0 = targets.rows0(row['ISR']) & cache.finite
    w_down, w_up = targets(row['ISR'], 0)['w1'], targets(row['ISR'], 1)['w1']
    index = binning['index'].numpy()
    for part, sign, base in (('fit_symmetry', 1, w0), ('fit_slope', -1, 0)):
        per_event = ((w_up + sign * w_down) / 2 - base)[rows0] ** 2
        expected = sum(np.bincount(index[f, rows0], per_event, binning['n_bins']) for f in range(index.shape[0]))
        assert np.allclose(parts[part]['shift_var'], expected)

    # paired rows give the same Var_MC as the general sum over events
    rng = np.random.default_rng(0)
    w_hat, w = rng.uniform(0.5, 1.5, len(events)), rng.uniform(0.5, 1.5, len(events))
    paired = nv.matched_variance(w_hat, binning['index'], events, w, binning['index'], events, binning['n_bins'])
    order = events[::-1].copy()
    general = nv.matched_variance(
        w_hat, binning['index'], events, w[order], binning['index'][:, order], order, binning['n_bins']
    )
    assert np.allclose(paired, general)


def test_eft_helpers(toy):
    coefs = toy.tensors[1]
    signal = np.arange(len(coefs)) < toy.n_signal

    # rho: the change of the signal weight on the signal events, 1 on the background events
    rho = nv.factorisation_weights(coefs, SM, POINTS, n_backgrounds=1)
    structure = coefs[signal][:, :N_Q]
    for label, theta in POINTS.items():
        expected = structure @ expansion(theta, N_WCS, dtype=torch.float64)
        expected = expected / (structure @ expansion(SM, N_WCS, dtype=torch.float64))
        assert np.allclose(rho[label][signal], expected.numpy()) and np.all(rho[label][~signal] == 1)

    # the terms of the learned EFT ratio: each 0 on the events of the other, and with the exact ratio w^s(c0) times
    # their sum is the weight at (theta, mu_bg = 1)
    eft = stand_in_eft(coefs)
    w_ref = toy.tensors[5][:, 0].double().numpy()
    for theta in POINTS.values():
        terms = nv.eft_terms(eft, theta)
        assert list(terms) == ['signal', 'bg']
        assert np.all(terms['signal'][~signal] == 0) and np.all(terms['bg'][signal] == 0)
        weight = coefs @ expansion(theta + [1.0], N_WCS, dtype=torch.float64)
        assert np.allclose(w_ref * (terms['signal'] + terms['bg']), weight.numpy())
    # an event the ensemble drops is nan in every term; a dropped background column cannot be separated
    kept = np.arange(len(coefs)) > 0
    terms = nv.eft_terms(types.SimpleNamespace(**{**vars(eft), 'finite': kept, 'Gamma': eft.Gamma[:, 1:]}), SM)
    assert all(np.isnan(term[0]) and np.isfinite(term[1:]).all() for term in terms.values())
    mask = np.r_[np.ones(N_Q, dtype=bool), False]
    with pytest.raises(ValueError, match='not separable'):
        nv.eft_terms(types.SimpleNamespace(**{**vars(eft), 'mask': mask, 'Gamma': eft.Gamma[:N_Q]}), SM)

    # member ratios on the events every member keeps, and the closure of the learned EFT ratio with their scatter
    rng = np.random.default_rng(3)
    n, n_wcs, n_bins = 200, 2, 4
    n_columns = (n_wcs + 1) * (n_wcs + 2) // 2 + 1
    theta = [1.0, 0.4, -0.3]
    finite = {e: np.ones(n, dtype=bool) for e in MEMBERS}
    finite[1][0] = False
    gammas = {e: rng.normal(1.0, 0.1, (n_columns, finite[e].sum())) for e in MEMBERS}
    members_eft = types.SimpleNamespace(
        seeds=MEMBERS,
        n_wcs=n_wcs,
        background_processes=['bg'],
        mask=np.ones(n_columns, dtype=bool),
        seed_finite=finite,
        seed_Gamma=gammas,
        finite=finite[0] & finite[1],
    )
    keep = members_eft.finite
    ratios = nv.eft_member_ratios(members_eft, theta, keep)
    t = expansion(theta + [1.0], n_wcs, dtype=torch.float64).numpy()
    assert ratios.shape == (len(MEMBERS), n) and np.all(ratios[:, ~keep] == 0)
    assert np.allclose(ratios[0, keep], t @ gammas[0][:, keep])
    # member 1's coefficients are stored on its own kept events
    assert np.allclose(ratios[1, keep], t @ gammas[1][:, keep[finite[1]]])

    bins = np.arange(n) % n_bins
    binning = {
        'index': torch.from_numpy(bins)[None],
        'n_bins': n_bins,
        'feature': np.zeros(n_bins, dtype=int),
        'names': ['x0'],
    }
    w0, rho = rng.uniform(0.5, 1.5, n), rng.uniform(0.8, 1.2, n)
    members = w0[None] * ratios
    w_hat0 = members.mean(0)
    sums, bias = nv.eft_closure(binning, w0, rho, w_hat0, keep, members)
    assert sums['bias'].shape == (len(MEMBERS), 1, n_bins)
    true = np.bincount(bins[keep], (w0 * rho)[keep], n_bins)
    per_member = np.stack([np.bincount(bins[keep], m[keep], n_bins) for m in members]) - true
    assert np.allclose(bias['D'][0], np.bincount(bins[keep], w_hat0[keep], n_bins) - true)
    assert np.allclose(bias['s2'][0], per_member.var(0, ddof=1) / len(MEMBERS))
    assert np.allclose(sums['mc_var'][0], np.bincount(bins[keep], (w_hat0 - w0 * rho)[keep] ** 2, n_bins))
    # the target of the learned EFT ratio alone is the nominal yield at theta: no true shift
    assert np.array_equal(sums['shift_var'], np.zeros(n_bins))
    assert np.allclose(sums['learned_shift_var'], sums['mc_var'][0])


def test_member_misclosure():
    # two inputs of two bins; members with the same bias (shared) and with opposite biases (independent noise)
    binning = {'names': ['a', 'year'], 'feature': np.array([0, 0, 1, 1])}
    base = {
        'mc_var': np.ones((1, 4)),
        'sm_yield': np.full(4, 100.0),
        'truth': np.array([110.0, 90.0, 100.0, 100.0]),
        'nominal': np.full(4, 100.0),
    }
    shared = dict(base, bias=np.tile([5.0, -5.0, 0.0, 0.0], (2, 1, 1)))
    result = nv.member_misclosure(shared, binning, 5.0)
    assert np.isclose(result['misclosure'], np.sqrt(48 / 4 / 1e4))
    assert np.isclose(result['mean_misclosure'], result['misclosure'])
    assert result['members_flagged'] == 2
    # an input that is not tested leaves the pooled bins
    result = nv.member_misclosure(shared, binning, 5.0, skip=('year',))
    assert np.isclose(result['misclosure'], np.sqrt(24 / 1e4))
    assert np.isclose(result['shift_size'], 0.1)
    opposite = dict(base, bias=np.array([[[5.0, -5.0, 0.0, 0.0]], [[-5.0, 5.0, 0.0, 0.0]]]))
    result = nv.member_misclosure(opposite, binning, 5.0, skip=('year',))
    assert np.allclose(result['member_misclosure'], np.sqrt(24 / 1e4))
    assert result['mean_misclosure'] == 0.0


def test_main_outputs(toy, run):
    name = run.config['name']
    with open(f'{name}/nuisance_diagnostics.yml') as f:
        diagnostics = yaml.safe_load(f)
    sums = np.load(f'{name}/nuisance_sums.npz')
    entries = diagnostics['nuisances']
    # the flagged network-directions, largest Z first, and the bins of every input
    flagged = [f'{row}_{d}' for row in ROWS for d in nv.DIRECTIONS if entries[row][d]['flagged']]
    assert sorted(diagnostics['flagged']) == sorted(flagged)
    z = [entries[label.rsplit('_', 1)[0]][label.rsplit('_', 1)[1]]['max_Z'] for label in diagnostics['flagged']]
    assert z == sorted(z, reverse=True)
    assert sum(diagnostics['binning'].values()) == sums['closure/D'].shape[1]
    assert diagnostics['nuisance']['seeds'] == run.cache.seeds
    # a nuisance of one era fills one bin of the era input: that input is not tested
    for d in nv.DIRECTIONS:
        assert np.isnan(entries['JER_2017'][d]['Z']['year']) and entries['JER_2017'][d]['worst_feature'] != 'year'
        assert not np.isnan(entries['ISR'][d]['Z']['year'])
    # the misclosure of each member alone: one value per member, flagged members counted
    for row in ROWS:
        for d in nv.DIRECTIONS:
            entry = entries[row][d]
            assert len(entry['member_misclosure']) == len(run.cache.seeds)
            assert np.isclose(entry['misclosure'], np.sqrt(np.mean(np.square(entry['member_misclosure']))))
            assert 0 <= entry['members_flagged'] <= len(run.cache.seeds)
    # the possible problems: the thresholds used, every problem at or above its threshold, worst first
    with open(f'{name}/nuisance_problems.yml') as f:
        problems = yaml.safe_load(f)
    assert problems['thresholds'] == nv.PROBLEM_THRESHOLDS
    severity = [p['severity'] for p in problems['problems']]
    assert severity == sorted(severity, reverse=True) and all(s >= 1 for s in severity)
    assert all(p['nuisance'] in entries for p in problems['problems'])
    assert list(entries) == list(ROWS) and diagnostics['factorisation_points'] == list(POINTS)
    assert {row: entries[row]['term'] for row in ROWS} == {row: NETWORKS[k][0] for row, k in ROWS.items()}

    # the factorisation test runs on the signal weight_shift networks, the combined test on every weight_shift network,
    # measured against the learned EFT ratio alone on its events (every event for these nuisances of every era)
    for label in POINTS:
        assert entries['ISR_bg']['factorisation'][label] is None and entries['JER_2017']['factorisation'][label] is None
        assert entries['JER_2017']['combined'][label] is None
        for row in ('ISR', 'ISR_bg'):
            for direction in nv.DIRECTIONS:
                test = entries[row]['combined'][label][direction]
                reference = diagnostics['eft_only'][label]
                assert test['baseline'] == 'eft_only'
                assert np.isclose(test['baseline_rms_bias'], reference['rms_bias'], equal_nan=True)
                assert np.isclose(test['rms_bias_change'], test['rms_bias'] - reference['rms_bias'], equal_nan=True)

    def same(a, b):
        """Whether two [rows, B] arrays of per-bin sums agree, rows not tested (nan) included."""
        scale = np.nanmax(np.abs(sums['closure/truth']))
        return np.allclose(a, b, rtol=1e-6, atol=1e-9 * scale, equal_nan=True)

    row = {name: [2 * list(ROWS).index(name) + d for d in range(len(nv.DIRECTIONS))] for name in ROWS}
    # every test at theta_ref normalises by the yield of every event, the sum of the nominal yields of the terms
    every_event = sums['closure/nominal'][row['ISR'][0]] + sums['closure/nominal'][row['ISR_bg'][0]]
    tests = ('closure', 'fit_symmetry', 'fit_slope', 'factorisation/sm', 'combined/sm', 'eft_only/sm', 'eft_matched/sm')
    for key in tests:
        lam = sums[f'{key}/sm_yield']
        assert same(lam[~np.isnan(lam).all(1)], np.broadcast_to(every_event, lam[~np.isnan(lam).all(1)].shape))
    # a network of one era is tested on fewer events than the normaliser holds
    assert np.all(sums['closure/nominal'][row['JER_2017']] <= every_event)
    # at theta_SM (rho = 1): the factorisation test of the signal network is its closure
    assert same(sums['factorisation/sm/D'][row['ISR']], sums['closure/D'][row['ISR']])

    for plot in (
        'z_heatmap_full',
        'z_heatmap_compact',
        'z_heatmap_parts_full',
        'z_heatmap_parts_compact',
        'network_misclosure',
        'size_vs_significance_networks',
        'size_vs_significance_slope',
        'size_vs_significance_symmetry',
        'factorisation_theta',
        'eft_combined_theta',
    ):
        assert os.path.exists(f'{name}/plots/{plot}.png')
    assert sorted(os.listdir(f'{name}/plots/interpolation_yield')) == sorted(f'{n}.png' for n in ROWS)
    # the rate noise: paired rows never move between bins, so their Var_MC is the sum over the bins of one feature
    first = sums['binning/feature'] == 0
    paired = np.repeat(sums['methods'] == 'weight_shift', len(nv.DIRECTIONS))
    assert np.allclose(sums['closure/rate_mc_var'][paired], sums['closure/mc_var'][paired][:, first].sum(1))
    assert np.all(sums['closure/rate_s2'] >= 0)
    for method in {entries[n]['method'] for n in ROWS}:
        assert os.path.exists(f'{name}/plots/yield_summary_{method}.png')
