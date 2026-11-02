"""
Background step of the nuisance validation (--backgrounds) on a small synthetic nuisance ensemble, written through
the real cache code: two members of two untrained nuisance networks per direction, a weight_shift in every era and a
kinematic_shift whose variation exists in era 2 only, evaluated on a signal event set and two background event sets.
"""

import types

import numpy as np
import pytest
import torch
import yaml

import topsbi.nuisance_validation as nv
from topsbi.model.net import Net
from topsbi.tools.build_ratio import NetworkRatio

MEMBERS = [0, 1]
C0 = [1.0, 0.5, -0.5]
WCS = ['cA', 'cB']
FEATURES = {'x0': {'loc': 0}, 'x1': {'loc': 1}, 'x2': {'loc': 2}}
NUISANCES = {'ISR': 'weight_shift', 'JER_X': 'kinematic_shift'}  # sorted by name: row n of f_hat
N_SIGNAL = 600
BACKGROUNDS = {'tW': 300, 'Others': 400}
NETWORK = [
    {'type': 'Linear', 'out': 8, 'activation': 'LeakyReLU'},
    {'type': 'Linear', 'in': 8, 'out': 1, 'activation': 'None'},
]


def toy_features(n, seed):
    """Features of a toy event set: x0, x1 uniform, x2 an era index 0-2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, 3))
    x[:, 2] = rng.integers(0, 3, n)
    return x


def nominal_dataset(x):
    """Combined nominal layout (features, coefs, ids, event_id, syst_ratio, w_ref) of the events x."""
    n = len(x)
    return torch.utils.data.TensorDataset(
        torch.from_numpy(x).float(),
        torch.zeros(n, 6, dtype=torch.float64),
        torch.zeros(n, 3),
        torch.stack([torch.arange(n)] * 4, dim=1),
        torch.ones(n, 1, 2),
        torch.ones(n, 1),
    )


def variation_dataset(x):
    """Combined variation layout (features, w_ref, ids, event_id) of the events x."""
    n = len(x)
    return torch.utils.data.TensorDataset(
        torch.from_numpy(x).float(), torch.ones(n, 1), torch.zeros(n, 3), torch.stack([torch.arange(n)] * 4, dim=1)
    )


def nuisance_network(directory, data, systematic, direction, seed):
    """An untrained nuisance network (linear output) with its training.yml and complete/performance.yml."""
    (directory / 'complete').mkdir(parents=True)
    torch.manual_seed(seed)
    net = Net(3, 'cpu', NETWORK)
    net.set_feature_normalization(torch.tensor([0.5, 0.5, 1.0]), torch.tensor([0.3, 0.3, 0.8]))
    torch.save(net.state_dict(), directory / 'model.pt')
    training = {
        'name': str(directory),
        'network': NETWORK,
        'data': str(data),
        'systematic': systematic,
        'direction': direction,
        'method': NUISANCES[systematic],
    }
    (directory / 'training.yml').write_text(yaml.safe_dump(training))
    (directory / 'complete' / 'performance.yml').write_text(yaml.safe_dump({}))
    return str(directory / 'training.yml')


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    """
    EFT validation config (eft.yml, with backgrounds) and the nuisance validation config's seeds, on a combined
    nominal sample with reference_point C0, a JER_X variation in era 2 only and two background holdouts.
    """
    root = tmp_path_factory.mktemp('nuisance_backgrounds')
    data = root / 'combined'
    x = {'signal': toy_features(N_SIGNAL, 0)}
    (data / 'nominal').mkdir(parents=True)
    torch.save(nominal_dataset(x['signal']), data / 'nominal' / 'validation.p')
    (data / 'nominal' / 'sample.yml').write_text(yaml.safe_dump({'reference_point': C0}))
    for direction in ('Up', 'Down'):
        (data / f'JER_X{direction}').mkdir()
        era2 = x['signal'][x['signal'][:, 2] == 2]
        torch.save(variation_dataset(era2), data / f'JER_X{direction}' / 'validation.p')
        (data / f'JER_X{direction}' / 'sample.yml').write_text(yaml.safe_dump({'reference_point': C0}))
    directories = {}
    for i, (p, n) in enumerate(BACKGROUNDS.items()):
        x[p] = toy_features(n, 10 + i)
        directory = root / 'backgrounds' / p
        directory.mkdir(parents=True)
        torch.save(nominal_dataset(x[p]), directory / 'validation.p')
        directories[p] = str(directory)
    seeds = {}
    for e in MEMBERS:
        networks = []
        for k, systematic in enumerate(NUISANCES):
            for d, direction in enumerate(('down', 'up')):
                directory = root / 'networks' / f'seed_{e}' / systematic / direction
                networks.append(nuisance_network(directory, data, systematic, direction, 100 * e + 10 * k + d))
        seeds[e] = {'networks': networks}
    eft = {
        'name': str(root / 'eft'),
        'data': str(data / 'nominal' / 'validation.p'),
        'c0': C0,
        'wcs': WCS,
        'lumi': [1.0, 2.0, 3.0],
        'year_loc': 2,
        'features': FEATURES,
        'backgrounds': directories,
    }
    eft_path = root / 'eft.yml'
    eft_path.write_text(yaml.safe_dump(eft, sort_keys=False))
    return types.SimpleNamespace(root=root, eft=eft, eft_path=str(eft_path), seeds=seeds, x=x)


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


def member_mean(toy, x, systematic):
    """[2, N] member mean of the networks' f_hat (Down, Up) of systematic on the events x, in float64."""
    k = list(NUISANCES).index(systematic)
    features = torch.from_numpy(x).float()
    f_hat = [
        [NetworkRatio(toy.seeds[e]['networks'][2 * k + d], 3).log_ratio(features).double().numpy() for d in (0, 1)]
        for e in MEMBERS
    ]
    return np.mean(f_hat, axis=0)


def test_load_config_backgrounds(toy, tmp_path):
    config = nv.load_config(write_config(toy, tmp_path))
    assert config['backgrounds'] == toy.eft['backgrounds']
    with pytest.raises(SystemExit, match='backgrounds is taken from'):
        nv.load_config(write_config(toy, tmp_path, backgrounds=toy.eft['backgrounds']))


def test_load_nuisance_ensemble_data_override(toy, tmp_path):
    config = nv.load_config(write_config(toy, tmp_path))
    data = f'{toy.eft["backgrounds"]["tW"]}/validation.p'
    path = str(tmp_path / 'nuisance_tW')
    features = torch.from_numpy(toy.x['tW']).float()
    cache = nv.load_nuisance_ensemble(config, features, refit=True, data=data, path=path)
    assert cache.manifest['data'] == data and cache.manifest['ensemble'] == toy.eft_path
    assert config['data'] == toy.eft['data']  # the caller's config is unchanged
    assert [n['eras'] for n in cache.nuisances] == [None, [2]]
    assert cache.f_hat.shape == (len(NUISANCES), 2, BACKGROUNDS['tW']) and cache.finite.all()
    assert np.allclose(cache.f_hat[0], member_mean(toy, toy.x['tW'], 'ISR'), rtol=1e-5, atol=1e-6)
    # the era rule: the kinematic_shift is 0 outside its eras
    era2 = toy.x['tW'][:, 2] == 2
    assert np.all(cache.f_hat[1][:, ~era2] == 0)
    assert np.allclose(cache.f_hat[1][:, era2], member_mean(toy, toy.x['tW'][era2], 'JER_X'), rtol=1e-5, atol=1e-6)
    # read back with the same data path; another data path or c0 stops the run
    assert nv.load_nuisance_ensemble(config, None, data=data, path=path).manifest['data'] == data
    with pytest.raises(SystemExit, match='another ensemble, data'):
        nv.load_nuisance_ensemble(config, None, path=path)
    with pytest.raises(SystemExit, match='theta_ref'):
        nv.load_nuisance_ensemble({**config, 'c0': [1.0, 0.0, 0.0]}, None, data=data, path=path)


def test_background_nuisances(toy, tmp_path, monkeypatch):
    config = nv.load_config(write_config(toy, tmp_path))
    caches = nv.background_nuisances(config)
    assert list(caches) == list(BACKGROUNDS)
    for p, cache in caches.items():
        assert cache.path == f'{config["name"]}/nuisance_{p}'
        assert cache.manifest['data'] == f'{toy.eft["backgrounds"][p]}/validation.p'
        assert cache.finite.shape == (BACKGROUNDS[p],)

    # a matching cache is read; refit builds every one from the background's own features
    calls = []

    def fake_load(config, features, refit=False, data=None, path=None):
        calls.append((refit, None if features is None else tuple(features.shape)))
        return nv.NuisanceCache(path)

    monkeypatch.setattr(nv, 'load_nuisance_ensemble', fake_load)
    nv.background_nuisances(config)
    assert calls == [(False, None)] * len(BACKGROUNDS)
    calls.clear()
    nv.background_nuisances(config, refit=True)
    assert calls == [(True, (n, 3)) for n in BACKGROUNDS.values()]


def test_run_backgrounds(toy, tmp_path):
    path = write_config(toy, tmp_path)
    with pytest.raises(SystemExit, match='no nuisance cache'):
        nv.main(path, backgrounds=True)
    nv.main(path, refit=True, backgrounds=True)
    name = tmp_path / 'validation'
    diagnostics = yaml.safe_load(open(name / 'background_diagnostics.yml'))
    assert list(diagnostics) == ['seeds', 'event_sets'] and diagnostics['seeds'] == MEMBERS
    assert list(diagnostics['event_sets']) == ['signal', *BACKGROUNDS]
    for set_name, n in [('signal', N_SIGNAL), *BACKGROUNDS.items()]:
        entry = diagnostics['event_sets'][set_name]
        assert entry['events'] == n and entry['dropped'] == 0.0
        assert entry['dropped_members'] == {e: 0.0 for e in MEMBERS}
    assert (name / 'nuisance' / 'nuisance.yml').exists()
    assert not (name / 'nuisance_diagnostics.yml').exists()  # the checks did not run
    # the caches now match, so the step runs again without refit
    nv.main(path, backgrounds=True)

    config = nv.load_config(path)
    with pytest.raises(SystemExit, match='needs the key backgrounds'):
        nv.run_backgrounds({k: v for k, v in config.items() if k != 'backgrounds'})
    with pytest.raises(SystemExit, match="'signal'"):
        nv.run_backgrounds({**config, 'backgrounds': {'signal': config['backgrounds']['tW']}})
    with pytest.raises(SystemExit, match='no validation.p'):
        nv.run_backgrounds({**config, 'backgrounds': {**config['backgrounds'], 'tW': str(tmp_path)}})
