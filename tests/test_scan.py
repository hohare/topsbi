import numpy as np
import torch
import types

from topsbi.scan import scan, scan_settings, target_weights

WCS = ['cA', 'cB']


def exact_inputs(n=400, seed=0):
    """
    Synthetic scan inputs with two WCs whose learned ratio is exact: each event's weight is the square of an
    amplitude a + b cA + c cB plus a positive constant, so w^s(theta) > 0 everywhere, and Gamma = coefs / w0, so the
    learned ratio equals the truth ratio and an Asimov fit returns its target exactly.
    """
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(1.0, 0.3, n), rng.normal(0.0, 0.5, n), rng.normal(0.0, 0.5, n)
    d = rng.uniform(0.1, 0.3, n)
    # get_lower_tri order of (1, cA, cB): 1, cA, cA^2, cB, cA cB, cB^2
    coefs = torch.tensor(np.stack([a**2 + d, 2 * a * b, b**2, 2 * a * c, 2 * b * c, c**2], 1), dtype=torch.float64)
    w0 = coefs[:, 0].clone()
    return types.SimpleNamespace(
        coefs=coefs,
        keep_q=torch.ones(6, dtype=torch.bool),
        w0=w0,
        Gamma=(coefs / w0[:, None]).T.contiguous(),
        mask=torch.ones(6, dtype=torch.bool),
        f_hat=None,
        nuisances=[],
        log_kappa=torch.zeros((0, 2, n), dtype=torch.float64),
        rate_names=[],
        wcs=WCS,
        scale=1.0,
    )


def test_theta_base_holds_other_wcs():
    """With cB held at its target, the Asimov best fit of cA is the target; held at 0 it is not."""
    inputs = exact_inputs()
    theta_t = np.array([1.5, 1.5])
    target = target_weights(inputs, theta_t)
    box = np.array([[0.0, 3.0], [0.0, 3.0]])
    grid = [np.linspace(1.0, 2.0, 11)]
    held = scan(inputs, target, ['cA'], grid, box, theta_t=theta_t, theta_base=theta_t, ratio='truth', rate='truth')
    assert abs(held.best.theta[0] - 1.5) < 1e-4
    assert held.best.theta[1] == 1.5
    assert held.q_target < 1e-6
    at_zero = scan(inputs, target, ['cA'], grid, box, theta_t=theta_t, ratio='truth', rate='truth')
    assert at_zero.best.theta[1] == 0.0
    assert abs(at_zero.best.theta[0] - 1.5) > 1e-3


def test_theta_base_starts_profiled_wcs():
    """A profiled cB starts at theta_base and the learned Asimov fit returns the target."""
    inputs = exact_inputs()
    theta_t = np.array([1.5, 1.5])
    target = target_weights(inputs, theta_t)
    box = np.array([[0.5, 2.5], [0.5, 2.5]])
    s = scan(
        inputs, target, ['cA'], [np.linspace(1.0, 2.0, 11)], box, profile_wcs=['cB'], theta_t=theta_t,
        theta_base=theta_t, rate='truth',
    )
    assert np.allclose(s.best.theta, theta_t, atol=1e-4)
    assert s.dropped == 0


def test_scan_settings_theta_base():
    """theta_base 'target' copies the target; a dict sets the listed WCs; absent means 0."""
    inputs = types.SimpleNamespace(wcs=WCS, nuisances=[], rate_names=[])
    entry = {'pois': ['cA'], 'grid': {'cA': {'range': [0.0, 1.0], 'points': 3}}, 'target': {'theta': {'cB': 1.5}}}
    config = {'box': {}, 'scans': {'t': dict(entry, theta_base='target'), 'd': dict(entry, theta_base={'cB': 0.7}),
                                   'z': entry}}
    assert list(scan_settings(config, 't', inputs)[0]['theta_base']) == [0.0, 1.5]
    assert list(scan_settings(config, 'd', inputs)[0]['theta_base']) == [0.0, 0.7]
    assert list(scan_settings(config, 'z', inputs)[0]['theta_base']) == [0.0, 0.0]
