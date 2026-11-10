import types

import numpy as np
import pytest
import torch
import yaml

import topsbi.tools.build_ratio as build
from topsbi.tools.data import expansion, get_lower_tri
from topsbi.tools.ensembleCache import network_rows

N_EVENTS = 6
# two WCs: get_lower_tri([1, a, b]) has 6 terms, 3 of which are kept
MASK = torch.tensor([True, True, False, True, False, False])
SEEDS = {
    # seed: (events kept by RatioFit, Gamma for the kept events)
    's1': (torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool), torch.arange(18.0).reshape(3, 6)),
    's2': (torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool), 10 + torch.arange(15.0).reshape(3, 5)),
}


class FakeRatioFit:
    """Stands in for RatioFit with known per-event coefficients Gamma."""

    masks = {}
    backgrounds = {}

    def __init__(self, config, features):
        self.finite, self.Gamma = SEEDS[config['sm']]
        self.mask = self.masks.get(config['sm'], MASK)
        self.n_wcs = 2
        self.background_processes = self.backgrounds.get(config['sm'], [])
        self.residuals = torch.rand(int(self.finite.sum()))
        self.rank = int(self.mask.sum())  # full column rank, as lstsq reports for a well-posed fit


@pytest.fixture
def ensemble_config(monkeypatch):
    monkeypatch.setattr(build, 'RatioFit', FakeRatioFit)
    FakeRatioFit.masks = {}
    FakeRatioFit.backgrounds = {}
    return {
        'device': 'cpu',
        'c0': [1.0, 0.0, 0.0],
        'features': {},
        'seeds': {seed: {'sm': seed, 'networks': []} for seed in SEEDS},
    }


def full_size(seed):
    keep, Gamma = SEEDS[seed]
    out = torch.zeros(3, N_EVENTS)
    out[:, keep] = Gamma
    return out


def test_ensemble_averages_aligned_events(ensemble_config):
    ens = build.Ensemble(ensemble_config, torch.zeros(N_EVENTS, 2))

    kept_by_all = SEEDS['s1'][0] & SEEDS['s2'][0]
    expected = ((full_size('s1') + full_size('s2')) / 2)[:, kept_by_all]
    assert torch.equal(ens.finite, kept_by_all)
    assert torch.allclose(ens.Gamma, expected)
    expected2 = ((full_size('s1') ** 2 + full_size('s2') ** 2) / 2)[:, kept_by_all]
    assert torch.allclose(ens.Gamma2, expected2)

    wcs = [1.0, 2.0, 3.0]
    assert torch.allclose(ens(wcs), get_lower_tri(wcs)[MASK] @ expected)
    assert ens.rank == {seed: int(MASK.sum()) for seed in SEEDS}


def test_ensemble_passes_each_seed_fit_to_per_seed(ensemble_config):
    residuals = {}
    build.Ensemble(
        ensemble_config, torch.zeros(N_EVENTS, 2), per_seed=lambda seed, fit: residuals.update({seed: fit.residuals})
    )
    assert [len(residuals[s]) for s in SEEDS] == [int(SEEDS[s][0].sum()) for s in SEEDS]


def test_ensemble_rejects_seeds_at_different_points(ensemble_config):
    other = MASK.clone()
    other[2] = True
    FakeRatioFit.masks = {'s2': other}
    with pytest.raises(ValueError):
        build.Ensemble(ensemble_config, torch.zeros(N_EVENTS, 2))


def test_is_cross_term():
    from topsbi.tools.ensembleCache import is_cross_term

    assert is_cross_term('ctGRe_cQd1/p_one_p_point_five')
    assert not is_cross_term('ctGRe/p_one') and not is_cross_term('sm')


def test_ensemble_rejects_seeds_with_different_background_processes(ensemble_config):
    FakeRatioFit.backgrounds = {'s2': ['bg']}
    with pytest.raises(ValueError, match='background_processes'):
        build.Ensemble(ensemble_config, torch.zeros(N_EVENTS, 2))


# RatioFit with a background column: one WC cA and one background process bg, networks trained against the
# reference theta_g = (cA = 1, mu_bg = 1) that return the true ratio of the toy
N_TOY = 200
TOY_C0 = [1.0, 1.0, 1.0]
TOY_SM = [1.0, 0.0, 1.0]
TOY_POINTS = {
    'cA/p_half': [1.0, 0.5, 1.0],
    'cA/n_half': [1.0, -0.5, 1.0],
    'cA/p_two': [1.0, 2.0, 1.0],
    'cA/n_two': [1.0, -2.0, 1.0],
}
MU_POINTS = {'bg/mu_0': [1.0, 1.0, 0.0], 'bg/mu_2': [1.0, 1.0, 2.0]}


def toy_components(x):
    """True s_q(x) (SM, cA, cA^2) and b(x) of every event, [N, 4]; every weight in the toy's range is positive."""
    x = x[:, 0].astype(np.float64)
    return np.stack([np.ones_like(x), 0.5 * x, 0.3 + x**2, 0.4 * (1 - x) + 0.1], axis=1)


def toy_ratio(x, point, reference):
    """True r(x; point) against reference; without strengths in the points the background is left out."""
    T, T0 = (expansion(p, 1, dtype=torch.float64).numpy() for p in (point, reference))
    c = toy_components(x)[:, : len(T)]
    return c @ T / (c @ T0)


def toy_gamma(x, reference):
    """True per-event coefficients Gamma [K, N] for the reference."""
    c = toy_components(x)[:, : len(expansion(reference, 1))]
    return (c / (c @ expansion(reference, 1, dtype=torch.float64).numpy())[:, None]).T


class ToyNetwork:
    """Stands in for NetworkRatio: the true ratio of the toy at the c1 of its training config."""

    def __init__(self, config, n_features, device='cpu'):
        with open(config) as f:
            self.config = yaml.safe_load(f)

    def __call__(self, features):
        return torch.from_numpy(toy_ratio(features.numpy(), self.config['c1'], self.config['c0'])).float()


def toy_configs(root, points, sm, reference, background_processes=('bg',)):
    """Training configs of the toy networks under root, {name: path}, the SM network under 'sm'."""
    paths = {}
    for name, point in {**points, 'sm': sm}.items():
        directory = root / name
        directory.mkdir(parents=True)
        config = {'name': str(directory), 'wcs': ['cA'], 'c0': reference, 'c1': point}
        if background_processes:
            config['background_processes'] = list(background_processes)
        (directory / 'training.yml').write_text(yaml.safe_dump(config))
        paths[name] = str(directory / 'training.yml')
    return paths


def seed_config(paths):
    return {'sm': paths['sm'], 'networks': [p for n, p in paths.items() if n != 'sm']}


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setattr(build, 'NetworkRatio', ToyNetwork)
    x = np.random.default_rng(0).uniform(0, 1, (N_TOY, 1))
    return types.SimpleNamespace(root=tmp_path, x=x, features=torch.from_numpy(x).float())


def fit_toy(toy, points, sm=TOY_SM, reference=TOY_C0, background_processes=('bg',), name='fit'):
    paths = toy_configs(toy.root / name, points, sm, reference, background_processes)
    config = {**seed_config(paths), 'features': {'x': {'loc': 0}}, 'device': 'cpu'}
    return build.RatioFit(config, toy.features)


def test_ratio_fit_background_column(toy):
    fit = fit_toy(toy, {**TOY_POINTS, **MU_POINTS})
    assert fit.n_wcs == 1 and fit.background_processes == ['bg']
    assert fit.mask.all()
    assert np.allclose(fit.Gamma.numpy(), toy_gamma(toy.x, TOY_C0), rtol=1e-3, atol=1e-4)
    point = [1.0, 0.3, 0.7]
    assert np.allclose(fit(point).numpy(), toy_ratio(toy.x, point, TOY_C0), rtol=1e-3, atol=1e-4)


def test_ratio_fit_blind_background_column(toy):
    # no network moves mu_bg: the column is dropped and the SM column holds the background exactly
    fit = fit_toy(toy, TOY_POINTS)
    assert fit.mask.tolist() == [True, True, True, False]
    gamma = toy_gamma(toy.x, TOY_C0)
    expected = np.vstack([gamma[0] + gamma[3], gamma[1:3]])
    assert np.allclose(fit.Gamma.numpy(), expected, rtol=1e-3, atol=1e-4)
    point = [1.0, 0.3, 1.0]
    assert np.allclose(fit(point).numpy(), toy_ratio(toy.x, point, TOY_C0), rtol=1e-3, atol=1e-4)
    with pytest.raises(ValueError, match='no network moved'):
        fit([1.0, 0.3, 0.7])


def test_ratio_fit_without_backgrounds(toy):
    points = {name: point[:2] for name, point in TOY_POINTS.items()}
    fit = fit_toy(toy, points, sm=[1.0, 0.0], reference=[1.0, 1.0], background_processes=())
    assert fit.background_processes == [] and fit.mask.tolist() == [True, True, True]
    assert np.allclose(fit.Gamma.numpy(), toy_gamma(toy.x, [1.0, 1.0]), rtol=1e-3, atol=1e-4)
    assert torch.allclose(fit([1.0, 0.3]), get_lower_tri([1.0, 0.3]) @ fit.Gamma)


def test_ratio_fit_checks_background_processes(toy):
    paths = toy_configs(toy.root / 'mixed', {**TOY_POINTS, **MU_POINTS}, TOY_SM, TOY_C0)
    config = yaml.safe_load(open(paths['bg/mu_0']))
    del config['background_processes']
    with open(paths['bg/mu_0'], 'w') as f:
        yaml.safe_dump(config, f)
    with pytest.raises(ValueError, match='background_processes'):
        build.RatioFit({**seed_config(paths), 'features': {'x': {'loc': 0}}, 'device': 'cpu'}, toy.features)
    # the SM row is (theta_SM, mu = 1): an SM network at another strength is refused
    with pytest.raises(ValueError, match='not trained at theta_1'):
        fit_toy(toy, {**TOY_POINTS, **MU_POINTS}, sm=[1.0, 0.0, 0.0], name='sm_mu_0')


# a second group of toy networks: points without background (mu_bg = 0) and a background-only network; with the SM
# network at (theta_SM, mu_bg = 1) they determine every column, as the background rows of a two-stage fit must
SIGNAL_POINTS = {'cA/p_half': [1.0, 0.5, 0.0], 'cA/n_two': [1.0, -2.0, 0.0], 'bg/only': [0.0, 0.0, 1.0]}


def test_ratio_fit_background_fit(toy):
    # two groups, the background column from the second group and the SM network alone
    mixed = toy_configs(toy.root / 'mixed', TOY_POINTS, TOY_SM, TOY_C0)
    signal = toy_configs(toy.root / 'signal', SIGNAL_POINTS, TOY_SM, TOY_C0)
    networks = {'mixed': list(mixed.values()), 'signal': [p for n, p in signal.items() if n != 'sm']}
    config = {'sm': signal['sm'], 'networks': networks, 'features': {'x': {'loc': 0}}, 'device': 'cpu'}
    fit = build.RatioFit({**config, 'background_fit': 'signal'}, toy.features)
    assert fit.groups == ['mixed'] * len(mixed) + ['signal'] * len(SIGNAL_POINTS)
    assert fit.background_rows.tolist() == [False] * len(mixed) + [True] * (len(SIGNAL_POINTS) + 1)
    # exact networks: the fit recovers the true Gamma
    assert np.allclose(fit.Gamma.numpy(), toy_gamma(toy.x, TOY_C0), rtol=1e-3, atol=1e-4)

    # on noisy ratios the fit is Gamma = Theta^- R, with Theta^- a left inverse of Theta that is not Theta^+
    Theta = fit.Theta.double()
    inverse = build.generalized_inverse(Theta, fit.background_rows, 1)
    assert torch.allclose(inverse @ Theta, torch.eye(Theta.shape[1], dtype=torch.float64), atol=1e-10)
    noise = torch.randn(fit.R[:, fit.finite].shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    R = fit.R[:, fit.finite].double() + 0.01 * noise
    Gamma = build.fit_gamma(Theta, R, fit.background_rows, 1)
    assert torch.allclose(Gamma, inverse @ R, atol=1e-10)
    assert not torch.allclose(Gamma, torch.linalg.pinv(Theta) @ R, atol=1e-6)
    # the background coefficient depends on the background rows alone
    shifted = R.clone()
    shifted[~fit.background_rows] += 0.1
    assert torch.allclose(build.fit_gamma(Theta, shifted, fit.background_rows, 1)[-1], Gamma[-1], atol=1e-10)
    # without background rows both are least squares over every row
    assert torch.allclose(build.fit_gamma(Theta, R), torch.linalg.pinv(Theta) @ R, atol=1e-10)
    # in blocks of events, from an array with a row of R that is not a row of Theta and with events left out: the
    # same fit and residuals
    extra = torch.vstack([R, 7 * R[:1]])
    rows = torch.tensor([True] * len(R) + [False])
    columns = torch.arange(R.shape[1]) % 3 != 0
    blocked = build.fit_gamma(Theta, extra.numpy(), fit.background_rows, 1, rows, columns, chunk=2)
    assert torch.allclose(blocked, Gamma[:, columns], atol=1e-10)
    residuals = ((Theta @ Gamma - R) ** 2).sum(0)[columns]
    assert torch.allclose(build.fit_residuals(Theta, blocked, extra.numpy(), rows, columns, chunk=2), residuals, atol=1e-10)
    assert torch.allclose(build.generalized_inverse(Theta), torch.linalg.pinv(Theta), atol=1e-12)

    with pytest.raises(ValueError, match='not a group'):
        build.RatioFit({**config, 'background_fit': 'other'}, toy.features)
    with pytest.raises(ValueError, match='needs an SM network'):
        build.RatioFit({**config, 'sm': None, 'background_fit': 'signal'}, toy.features)

    # row names: '<group>:' and the directory relative to the group's common directory; the SM network takes the
    # group whose directory holds it, and the names and groups follow the rows of the fit
    paths, names, groups = network_rows(networks, signal['sm'])
    assert paths == networks['mixed'] + networks['signal'] + [signal['sm']]
    assert names == [f'mixed:{n}' for n in mixed] + [f'signal:{n}' for n in SIGNAL_POINTS] + ['signal:sm']
    assert groups == fit.groups + ['signal'] and len(names) == fit.Theta.shape[0]
    with pytest.raises(ValueError, match='no group of networks holds'):
        network_rows(networks, str(toy.root / 'elsewhere' / 'training.yml'))
    # a list of networks: names relative to the common directory of every network and the SM network, no groups;
    # without an SM network the exact last row is 'sm'
    plain = [p for n, p in mixed.items() if n != 'sm']
    assert network_rows(plain, mixed['sm']) == (plain + [mixed['sm']], [*TOY_POINTS, 'sm'], [None] * len(mixed))
    # without an SM network the names are relative to the networks' own common directory (here .../mixed/cA)
    expected = [n.split('/')[1] for n in TOY_POINTS] + ['sm']
    assert network_rows(plain, None) == (plain + [None], expected, [None] * (len(plain) + 1))


def test_ensemble_background_column(toy):
    paths = toy_configs(toy.root / 'ensemble', {**TOY_POINTS, **MU_POINTS}, TOY_SM, TOY_C0)
    config = {'device': 'cpu', 'c0': TOY_C0, 'features': {'x': {'loc': 0}}, 'seeds': {0: seed_config(paths)}}
    config['seeds'][1] = config['seeds'][0]
    ens = build.Ensemble(config, toy.features)
    assert ens.n_wcs == 1 and ens.background_processes == ['bg'] and ens.mask.all()
    point = [1.0, 0.3, 0.7]
    assert np.allclose(ens(point).numpy(), toy_ratio(toy.x, point, TOY_C0), rtol=1e-3, atol=1e-4)
    blind = toy_configs(toy.root / 'blind', TOY_POINTS, TOY_SM, TOY_C0)
    ens = build.Ensemble({**config, 'seeds': {0: seed_config(blind)}}, toy.features)
    assert not ens.mask[-1]
    with pytest.raises(ValueError, match='no network moved'):
        ens(point)
