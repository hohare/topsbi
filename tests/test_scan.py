import numpy as np
import os
import pytest
import torch
import types
import yaml

from topsbi.scan import (
    background_caches,
    background_dropped,
    background_target_weights,
    calibrated_backgrounds,
    calibrated_inputs,
    check_background_options,
    check_nuisance_caches,
    check_scans,
    combine_series,
    combined_f_hat,
    curve_segments,
    impacts,
    load_background_inputs,
    load_config,
    nll,
    nested_bins,
    nll_terms,
    positive_background_events,
    positive_events,
    scan,
    scan_series,
    scan_settings,
    scan_targets,
    scan_uncertainty,
    summary_plot,
    target_weights,
    write_results,
)

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


def write_caches(tmp_path, sm=False):
    """Signal and tW caches whose manifests agree, with a second nuisance validation (w_sm, theta_ref = SM, not c0)
    if sm; returns the scan config being completed and the EFT config."""
    eft_name, nu_name = tmp_path / 'validation', tmp_path / 'nuisance_validation'
    sm_name = tmp_path / 'nuisance_validation_sm'
    data, tw = str(tmp_path / 'signal/validation.p'), str(tmp_path / 'tW/nominal')
    networks = {'tW': {1: 'tw_1.yml', 0: 'tw_0.yml'}}
    ensemble = {'c0': [1.0, 1.5], 'wcs': ['cA'], 'lumi': [1.0], 'seeds': {0: {'networks': ['a.yml']}},
                'model_hash': {0: ['h']}}
    nuisance = {'ensemble': 'eft.yml', 'theta_ref': [1.0, 1.5], 'nuisances': [{'name': 'n1'}], 'seeds': {0: {}},
                'model_hash': {0: [['g']]}}
    files = [
        (eft_name / 'ensemble', 'ensemble.yml', dict(ensemble, data=data)),
        (eft_name / 'ensemble_tW', 'ensemble.yml', dict(ensemble, data=f'{tw}/validation.p')),
        (nu_name / 'nuisance', 'nuisance.yml', dict(nuisance, data=data)),
        (nu_name / 'nuisance_tW', 'nuisance.yml', dict(nuisance, data=f'{tw}/validation.p')),
        (eft_name / 'background_ratios', 'manifest.yml',
         {'networks': {'tW': {0: 'tw_0.yml', 1: 'tw_1.yml'}}, 'model_hash': {}, 'reference_point': [1.0, 1.5],
          'data': {'signal': data, 'tW': f'{tw}/validation.p'}}),
    ]
    if sm:
        sm_nuisance = dict(nuisance, theta_ref=[1.0, 0.0], model_hash={0: [['s']]})
        files += [
            (sm_name / 'nuisance', 'nuisance.yml', dict(sm_nuisance, data=data)),
            (sm_name / 'nuisance_tW', 'nuisance.yml', dict(sm_nuisance, data=f'{tw}/validation.p')),
        ]
    for path, name, content in files:
        os.makedirs(path, exist_ok=True)
        with open(path / name, 'w') as f:
            yaml.safe_dump(content, f)
    for path in (eft_name / 'ensemble', eft_name / 'ensemble_tW'):
        np.save(path / 'mask_cache.npy', np.array([True, True, False]))
    caches = [str(nu_name / 'nuisance')] + ([str(sm_name / 'nuisance')] if sm else [])
    config = {'ensemble': 'eft.yml', 'data': data, 'c0': [1, 1.5], 'processes': ['tt'],
              'eft_cache': str(eft_name / 'ensemble'), 'nuisance_caches': caches}
    eft = {'name': str(eft_name), 'backgrounds': {'tW': tw}, 'background_networks': networks}
    return config, eft


def test_background_caches_match(tmp_path):
    """Matching caches resolve to the per-process directories and the ratio cache; a second nuisance validation at
    another theta_ref than c0 adds its own nuisance_<p> sibling."""
    config, eft = write_caches(tmp_path)
    caches, ratios = background_caches(config, eft)
    assert caches['tW']['eft_cache'].endswith('validation/ensemble_tW')
    assert [path[len(str(tmp_path)):] for path in caches['tW']['nuisance_caches']] == [
        '/nuisance_validation/nuisance_tW'
    ]
    assert caches['tW']['data'].endswith('tW/nominal/validation.p')
    assert ratios.endswith('validation/background_ratios')
    config, eft = write_caches(tmp_path / 'sm', sm=True)
    caches, _ = background_caches(config, eft)
    assert [path[len(str(tmp_path / 'sm')):] for path in caches['tW']['nuisance_caches']] == [
        '/nuisance_validation/nuisance_tW',
        '/nuisance_validation_sm/nuisance_tW',
    ]


def test_background_caches_mismatch(tmp_path, capsys):
    """Every disagreement with the signal caches is reported, then the run stops; each nuisance_<p> is compared with
    its own signal nuisance cache."""
    config, eft = write_caches(tmp_path, sm=True)
    for path, name, key, value in [
        (f'{eft["name"]}/ensemble_tW', 'ensemble.yml', 'model_hash', {0: ['other']}),
        (f'{tmp_path}/nuisance_validation/nuisance_tW', 'nuisance.yml', 'theta_ref', [1.0, 0.0]),
        (f'{tmp_path}/nuisance_validation_sm/nuisance_tW', 'nuisance.yml', 'theta_ref', [1.0, 1.5]),
        (f'{eft["name"]}/background_ratios', 'manifest.yml', 'reference_point', [1.0, 0.0]),
    ]:
        with open(f'{path}/{name}') as f:
            content = yaml.safe_load(f)
        content[key] = value
        with open(f'{path}/{name}', 'w') as f:
            yaml.safe_dump(content, f)
    np.save(f'{eft["name"]}/ensemble_tW/mask_cache.npy', np.array([True, False, False]))
    with pytest.raises(SystemExit) as stop:
        background_caches(config, eft)
    message = str(stop.value)
    for text in (
        'another model_hash',
        'other terms of T(theta)',
        f'another theta_ref than {tmp_path}/nuisance_validation/nuisance\n',
        f'another theta_ref than {tmp_path}/nuisance_validation_sm/nuisance\n',
        'has reference point',
    ):
        assert text in message


def test_load_config_nuisance_list(tmp_path):
    """nuisance takes one nuisance validation config or a list: the caches follow in order; a repeat stops."""
    eft, gen, sm = tmp_path / 'eft.yml', tmp_path / 'gen.yml', tmp_path / 'sm.yml'
    for path, content in [
        (eft, {'name': 'v', 'data': 'd.p', 'c0': [1.0, 1.5], 'wcs': ['cA']}),
        (gen, {'name': 'g'}),
        (sm, {'name': 's'}),
    ]:
        with open(path, 'w') as f:
            yaml.safe_dump(content, f)

    def config(nuisance):
        path = tmp_path / 'scan.yml'
        with open(path, 'w') as f:
            yaml.safe_dump({'name': 'out', 'ensemble': str(eft), 'sm_yield': 1.0, 'scans': {}, 'nuisance': nuisance}, f)
        return load_config(str(path))

    assert config(str(gen))['nuisance_caches'] == ['g/nuisance']
    assert config([str(gen), str(sm)])['nuisance_caches'] == ['g/nuisance', 's/nuisance']
    with pytest.raises(SystemExit) as stop:
        config([str(gen), str(gen)])
    assert 'the same nuisance cache twice' in str(stop.value)


def nuisance_cache(path, names, theta_ref, f_hat, finite, method='weight_shift', ensemble='eft.yml'):
    """A stand-in for NuisanceCache: path, manifest (ensemble), nuisances, theta_ref, f_hat (on its finite events) and
    finite."""
    return types.SimpleNamespace(
        path=path,
        manifest={'ensemble': ensemble},
        nuisances=[{'name': n, 'method': method} for n in names],
        theta_ref=theta_ref,
        f_hat=f_hat,
        finite=finite,
    )


def test_combined_nuisances():
    """A later nuisance cache replaces the base rows of its nuisances (base order kept) on the events every cache
    keeps, each nuisance records the theta_ref and cache it comes from, one cache returns its own rows, and every
    inconsistency between the caches stops the run."""
    rng = np.random.default_rng(3)
    n = 6
    base_finite = np.array([True, True, True, False, True, True])
    sm_finite = np.array([True, False, True, True, True, True])
    base_all, sm_all = rng.normal(size=(3, 2, n)), rng.normal(size=(1, 2, n))
    base = nuisance_cache('gen/nuisance', ['ISR', 'PU', 'fact'], [1.0, 1.5], base_all[..., base_finite], base_finite)
    sm = nuisance_cache('sm/nuisance', ['fact'], [1.0, 0.0], sm_all[..., sm_finite], sm_finite)
    config = {'ensemble': 'eft.yml'}
    check_nuisance_caches(config, [base, sm])
    keep = base_finite & sm_finite
    f_hat, names, reference = combined_f_hat([base, sm], keep)
    assert names == ['ISR', 'PU', 'fact']
    assert f_hat.dtype == torch.float64 and f_hat.shape == (3, 2, int(keep.sum()))
    assert np.allclose(f_hat[:2].numpy(), base_all[:2][..., keep])
    assert np.allclose(f_hat[2].numpy(), sm_all[0][..., keep])
    assert reference['PU'] == {'theta_ref': [1.0, 1.5], 'cache': 'gen/nuisance'}
    assert reference['fact'] == {'theta_ref': [1.0, 0.0], 'cache': 'sm/nuisance'}
    single, single_names, _ = combined_f_hat([base], base_finite)
    assert single_names == names and np.allclose(single.numpy(), base_all[..., base_finite])

    bad = [
        nuisance_cache('x/nuisance', ['hdamp'], [1.0, 0.0], sm_all[..., sm_finite], sm_finite),
        nuisance_cache('y/nuisance', ['ISR'], [1.0, 0.0], sm_all[..., sm_finite], sm_finite, method='kinematic_shift'),
        nuisance_cache('z/nuisance', ['fact'], [1.0, 0.0], sm_all[..., sm_finite], sm_finite, ensemble='other.yml'),
    ]
    with pytest.raises(SystemExit) as stop:
        check_nuisance_caches(config, [base, sm] + bad)
    message = str(stop.value)
    for text in (
        'x/nuisance: nuisance hdamp is not in the base cache gen/nuisance',
        'y/nuisance: nuisance ISR is kinematic_shift, weight_shift in gen/nuisance',
        'nuisance fact is replaced by both sm/nuisance and z/nuisance',
        'z/nuisance was built from other.yml, not eft.yml',
    ):
        assert text in message


def background_config(tmp_path, **options):
    """A scan config after load_config with backgrounds tW and Others and a DY combined-sample directory."""
    dy = tmp_path / 'DY/nominal'
    os.makedirs(dy, exist_ok=True)
    (dy / 'validation.p').write_bytes(b'')
    with open(dy / 'sample.yml', 'w') as f:
        yaml.safe_dump({'process': 'DY', 'eft': False}, f)
    return {
        'processes': ['tt'],
        'background_caches': {'tW': {}, 'Others': {}},
        'rate_systematics': {'lumi': {'kappa': 1.016}, 'smxsec_DY': {'kappa': 1.15, 'processes': ['DY']}},
        'backgrounds': dict({'rate_only': {'DY': {'shape': 'Others', 'data': str(dy)}}}, **options),
    }


def test_background_options_valid(tmp_path):
    """Own yields with DY from its sample, card yields for every process, and calibration (defaults or set) pass."""
    check_background_options(background_config(tmp_path))
    check_background_options(
        background_config(tmp_path, yields='card', card_yields={'tW': 5000.0, 'Others': 1200.0, 'DY': 300.0})
    )
    check_background_options(background_config(tmp_path, calibration={}))
    check_background_options(background_config(tmp_path, calibration={'bins': 10, 'folds': 4}))


def test_background_options_invalid(tmp_path):
    """Every problem of the block is reported, then the run stops."""
    config = background_config(
        tmp_path, yields='card', card_yields={'tW': -1.0, 'ttH': 3.0}, extra=1,
        calibration={'bins': 1, 'folds': 2.5, 'wc': 'cA'},
    )
    config['backgrounds']['rate_only']['DY']['shape'] = 'tt'
    config['rate_systematics']['smxsec_ttZ'] = {'kappa': 1.1, 'processes': ['ttZ']}
    config['scans'] = {'s0': {'uncertainty': {'n_boot': 5}}}
    with pytest.raises(SystemExit) as stop:
        check_background_options(config)
    message = str(stop.value)
    for text in (
        "unknown keys ['extra']",
        'shape must be one of the background processes',
        "card_yields for ['Others', 'DY']",
        "names ['ttH']",
        'card_yields tW must be a positive number',
        "processes ['ttZ']",
        "calibration: unknown keys ['wc']",
        'calibration bins must be an integer >= 2, not 1',
        'calibration folds must be an integer >= 2, not 2.5',
        "scans ['s0'] set uncertainty",
    ):
        assert text in message
    config = background_config(tmp_path)
    config['backgrounds']['rate_only']['DY']['yield'] = 250.0
    config['scans'] = {'s1': {'ratio': 'truth'}}
    with pytest.raises(SystemExit) as stop:
        check_background_options(config)
    assert 'exactly one of data and yield' in str(stop.value)
    assert "scans ['s1'] use a truth ratio" in str(stop.value)


def write_background_sample(directory, weights, eras, fractions):
    """A combined background sample (nominal layout, era in feature column 0) with the given per-split weights."""
    os.makedirs(directory, exist_ok=True)
    for split, values in weights.items():
        n = len(values)
        features = torch.stack([torch.tensor(eras[split], dtype=torch.float32), torch.zeros(n)], 1)
        w_ref = torch.tensor(values, dtype=torch.float64)[:, None]
        dataset = torch.utils.data.TensorDataset(
            features, torch.zeros(n, 1), torch.zeros(n, 2), torch.arange(n), torch.ones(n, 1), w_ref
        )
        torch.save(dataset, f'{directory}/{split}.p')
    with open(f'{directory}/sample.yml', 'w') as f:
        yaml.safe_dump({'eft': False, 'split': {s: {'weight_fraction': v} for s, v in fractions.items()}}, f)


def test_load_background_inputs(tmp_path, monkeypatch):
    """Yields, normalisation, per-set weights and rate systematics of the background components."""
    import topsbi.scan as scan_module

    tw_dir, others_dir, dy_dir = tmp_path / 'tW', tmp_path / 'Others', tmp_path / 'DY'
    write_background_sample(tw_dir, {'validation': [0.5, 1.0, 1.5]}, {'validation': [0, 1, 1]}, {'validation': 0.25})
    write_background_sample(
        others_dir, {'validation': [2.0, -0.5, 1.0, 3.0]}, {'validation': [0, 0, 1, 1]}, {'validation': 0.5}
    )
    write_background_sample(
        dy_dir, {'train': [1.0, 1.0], 'validation': [2.0]}, {'train': [0, 1], 'validation': [1]},
        {'train': 0.6, 'validation': 0.4},
    )
    n_signal, K = 6, 3
    rng = np.random.default_rng(1)
    ratios = tmp_path / 'background_ratios'
    os.makedirs(ratios)
    sizes = {'signal': n_signal, 'tW': 3, 'Others': 4}
    finite = {'signal': np.ones(6, bool), 'tW': np.ones(3, bool), 'Others': np.array([True, True, False, True])}
    for name, n in sizes.items():
        np.savez(
            ratios / f'{name}.npz', r_hat=rng.uniform(0.5, 2.0, (2, 2, n)).astype(np.float32),
            processes=np.array(['tW', 'Others']), seeds=np.array([0, 1]), finite=finite[name],
        )

    def ensemble(n):
        return types.SimpleNamespace(
            finite=np.ones(n, bool), Gamma=rng.normal(size=(K, n)), seeds=[0],
            seed_Gamma={0: rng.normal(size=(K, n))}, seed_finite={0: np.ones(n, bool)},
        )

    caches = {'tW/ens': ensemble(3), 'Others/ens': ensemble(4)}
    monkeypatch.setattr(scan_module, 'EnsembleCache', lambda path: caches[path])
    config = {
        'processes': ['tt'],
        'lumi': [1.0, 2.0],
        'year_loc': 0,
        'rate_systematics': {'lumi': {'kappa': 1.016}, 'smxsec_DY': {'kappa': 1.15, 'processes': ['DY']}},
        'backgrounds': {'rate_only': {'DY': {'shape': 'Others', 'data': str(dy_dir)}}},
        'background_caches': {
            'tW': {'data': str(tw_dir / 'validation.p'), 'eft_cache': 'tW/ens', 'nuisance_caches': None},
            'Others': {'data': str(others_dir / 'validation.p'), 'eft_cache': 'Others/ens', 'nuisance_caches': None},
        },
        'background_ratios': str(ratios),
    }
    inputs = types.SimpleNamespace(
        events=np.ones(n_signal, bool), w0=torch.tensor(rng.uniform(1.0, 2.0, n_signal), dtype=torch.float64),
        scale=3.0,
    )
    b = load_background_inputs(config, inputs, members=True)
    assert b.components == ['tW', 'Others', 'DY'] and b.shapes == ['tW', 'Others', 'Others']
    tw_yield = (0.5 * 1 + 1.0 * 2 + 1.5 * 2) * 1000 / 0.25
    others_yield = (2.0 * 1 - 0.5 * 1 + 1.0 * 2 + 3.0 * 2) * 1000 / 0.5
    dy_yield = (1.0 * 1 + 1.0 * 2 + 2.0 * 2) * 1000 / 1.0
    assert np.allclose(b.yields.numpy(), [tw_yield, others_yield, dy_yield])
    assert np.allclose((inputs.scale * inputs.w0 * b.signal_b).sum(1).numpy(), b.yields.numpy())
    others = b.sets['Others']
    assert others.events.tolist() == [True, True, False, True]
    assert others.Gamma.shape == (K, 3) and set(others.seed_Gamma) == {0}
    assert np.allclose(b.sets['tW'].w.sum(1).numpy(), [tw_yield, 0.0, 0.0])
    assert np.allclose(others.w.sum(1).numpy(), [0.0, others_yield, dy_yield])
    assert np.allclose(others.w[2].numpy(), others.w[1].numpy() * dy_yield / others_yield)
    assert np.allclose((others.b[2] / others.b[1]).numpy(), (b.norm[2] / b.norm[1]).item())
    assert np.allclose(b.log_kappa[1].numpy(), [[0.0, 0.0, -np.log(1.15)], [0.0, 0.0, np.log(1.15)]])
    assert np.allclose(b.log_kappa[0, 1].numpy(), np.log(1.016))
    assert np.allclose(b.signal_log_kappa.numpy(), [[-np.log(1.016), np.log(1.016)], [0.0, 0.0]])
    config['backgrounds'].update(yields='card', card_yields={'tW': 10.0, 'Others': 20.0, 'DY': 5.0})
    assert np.allclose(load_background_inputs(config, inputs).yields.numpy(), [10.0, 20.0, 5.0])


def test_positive_background_events():
    """Background events are selected by the sign of the signal term over the box, as signal events are, members too."""
    inputs = exact_inputs(n=50, seed=2)
    Gamma = inputs.Gamma.clone()
    Gamma[1, :5] = -10.0  # cA slope: negative at cA = 1 for the first five events
    member = Gamma.clone()
    member[1, 5:7] = -10.0  # and for two more in the member
    w = torch.stack([torch.ones(50), 2 * torch.ones(50)]).to(torch.float64)
    backgrounds = types.SimpleNamespace(
        components=['tW', 'DY'],
        shapes=['tW', 'tW'],
        sets={'tW': types.SimpleNamespace(Gamma=Gamma, seed_Gamma={0: member}, w=w)},
    )
    box = np.array([[0.0, 1.0], [0.0, 0.0]])
    keep = positive_background_events(backgrounds, inputs.mask, box=box)
    assert torch.equal(keep['tW'], positive_events([Gamma], inputs.mask, box=box))
    assert not keep['tW'][:5].any() and keep['tW'][5:].all()
    keep = positive_background_events(backgrounds, inputs.mask, box=box, members=True)
    assert not keep['tW'][:7].any() and keep['tW'][7:].all()
    assert background_dropped(backgrounds, keep) == pytest.approx({'tW': 7 / 50, 'DY': 7 / 50})
    backgrounds.sets['tW'].seed_Gamma = None
    with pytest.raises(ValueError):
        positive_background_events(backgrounds, inputs.mask, box=box, members=True)


def mixture_inputs(beta=0.1, seed=0, n=400):
    """
    exact_inputs with rate systematics lumi (every process, kappa 1.02) and smxsec_B (background only, 1.2) and one
    background component B whose events are the signal events reweighted by beta (1 + 0.5 tanh(coefs_1)): its term
    b(x) is exact, and its truth yield equals its learned yield.
    """
    inputs = exact_inputs(n=n, seed=seed)
    n = len(inputs.w0)
    inputs.rate_names = ['lumi', 'smxsec_B']
    inputs.log_kappa = torch.zeros(2, 2, n, dtype=torch.float64)
    inputs.log_kappa[0] = torch.tensor([-np.log(1.02), np.log(1.02)], dtype=torch.float64)[:, None]
    b = beta * (1 + 0.5 * torch.tanh(inputs.coefs[:, 1]))
    log_kappa_b = torch.tensor([[[-np.log(1.02)], [np.log(1.02)]], [[-np.log(1.2)], [np.log(1.2)]]], dtype=torch.float64)
    inputs.backgrounds = types.SimpleNamespace(
        components=['B'],
        shapes=['B'],
        log_kappa=log_kappa_b,
        signal_log_kappa=torch.tensor([[-np.log(1.02), np.log(1.02)], [0.0, 0.0]], dtype=torch.float64),
        signal_b=b[None],
        sets={
            'B': types.SimpleNamespace(
                Gamma=inputs.Gamma, seed_Gamma=None, f_hat=None, b=b[None], w=(inputs.scale * inputs.w0 * b)[None],
                era=None, events=np.ones(n, bool),
            )
        },
    )
    return inputs


def mixture_terms(inputs, theta_t, nu_t):
    """nll_terms of an Asimov dataset at (theta_t, nu_t), every event selected, nu_obs = nu_t."""
    n = len(inputs.w0)
    target = target_weights(inputs, theta_t, nu_t)
    kappa = 1.02 ** nu_t[0] * 1.2 ** nu_t[1]
    background_target = {'B': inputs.backgrounds.sets['B'].w[0] * kappa}
    return nll_terms(
        inputs, torch.ones(n, dtype=torch.bool), target, nu_obs=nu_t,
        background_keep={'B': torch.ones(n, dtype=torch.bool)}, background_target=background_target,
    )


def test_mixture_asimov_stationary():
    """An Asimov dataset with a background is a stationary point of the mixture NLL at (theta_t, nu_t), both rates."""
    inputs = mixture_inputs()
    theta_t, nu_t = np.array([0.7, -0.4]), np.array([0.3, 0.5])
    terms = mixture_terms(inputs, theta_t, nu_t)
    for rate in ('learned', 'truth'):
        theta = torch.tensor(theta_t, dtype=torch.float64, requires_grad=True)
        nu = torch.tensor(nu_t, dtype=torch.float64, requires_grad=True)
        nll(theta, nu, terms, rate=rate).backward()
        assert torch.allclose(theta.grad, torch.zeros(2, dtype=torch.float64), atol=1e-8)
        assert torch.allclose(nu.grad, torch.zeros(2, dtype=torch.float64), atol=1e-8)
        off = torch.tensor([0.3, 0.0], dtype=torch.float64, requires_grad=True)
        nll(torch.tensor(theta_t, dtype=torch.float64), off, terms, rate=rate).backward()
        assert off.grad[1] < -1e-3


def test_mixture_zero_background():
    """A background with zero yield reproduces the NLL without backgrounds; ratio truth with backgrounds stops."""
    inputs = mixture_inputs(beta=0.0)
    theta_t, nu_t = np.array([0.7, -0.4]), np.array([0.3, 0.0])
    with_b = mixture_terms(inputs, theta_t, nu_t)
    inputs.backgrounds = None
    without = nll_terms(inputs, torch.ones(len(inputs.w0), dtype=torch.bool), target_weights(inputs, theta_t, nu_t),
                        nu_obs=nu_t)
    theta, nu = torch.tensor([0.2, 0.1], dtype=torch.float64), torch.tensor([-0.4, 0.7], dtype=torch.float64)
    for rate in ('learned', 'truth'):
        assert torch.allclose(nll(theta, nu, with_b, rate=rate), nll(theta, nu, without, rate=rate), rtol=1e-12)
    with pytest.raises(ValueError):
        nll(theta, nu, with_b, ratio='truth')


def test_background_target_weights():
    """Asimov background targets are w_c kappa_c(nu_t); toys are reproducible, zero outside keep, and refuse
    negative expected weights."""
    inputs = mixture_inputs()
    nu_t = np.array([0.3, 0.5])
    asimov = background_target_weights(inputs, nu_t)
    w = inputs.backgrounds.sets['B'].w[0]
    assert torch.allclose(asimov['B'], w * 1.02**0.3 * 1.2**0.5, rtol=1e-12)
    assert torch.equal(background_target_weights(inputs)['B'], w)
    keep = {'B': torch.arange(len(w)) >= 100}
    toy = background_target_weights(inputs, nu_t, kind='toy', keep=keep, seed=3)
    assert torch.equal(toy['B'], background_target_weights(inputs, nu_t, kind='toy', keep=keep, seed=3)['B'])
    assert (toy['B'][:100] == 0).all() and toy['B'].sum() > 0
    inputs.backgrounds.sets['B'].w[0, 150] = -1.0
    with pytest.raises(ValueError):
        background_target_weights(inputs, nu_t, kind='toy', keep=keep, seed=3)


def test_scan_with_background():
    """An Asimov scan with a background returns its target, keeps every event, and a zero-yield background
    reproduces the scan without backgrounds."""
    theta_t, nu_t = np.array([0.7, -0.4]), np.array([0.3, 0.5])
    box = np.array([[-0.5, 2.0], [-1.5, 1.0]])
    grid = [np.linspace(0.3, 1.1, 9)]
    inputs = mixture_inputs()
    target = target_weights(inputs, theta_t, nu_t)
    s = scan(
        inputs, target, ['cA'], grid, box, profile_wcs=['cB'], profile_nu=['lumi', 'smxsec_B'], nu_obs=nu_t,
        theta_t=theta_t, theta_base=theta_t, background_target=background_target_weights(inputs, nu_t),
    )
    assert np.allclose(s.best.theta, theta_t, atol=1e-4)
    assert np.allclose(s.best.nu, nu_t, atol=1e-3)
    assert s.q_target < 1e-6
    assert s.background_dropped == {'B': 0} and s.background_dropped_fraction == {'B': 0.0}
    with pytest.raises(ValueError):
        scan(inputs, target, ['cA'], grid, box, theta_t=theta_t)
    inputs.seed_Gamma = {0: inputs.Gamma, 1: inputs.Gamma.clone()}
    inputs.backgrounds.sets['B'].seed_Gamma = {0: inputs.Gamma, 1: inputs.Gamma.clone()}
    u = scan_uncertainty(
        inputs, target, dict(pois=['cA'], grids=grid, box=box, profile_wcs=['cB'], theta_base=theta_t), n_boot=1,
        background_target=background_target_weights(inputs, nu_t), theta_t=theta_t,
    )
    assert np.allclose(u.members['best'], u.nominal.best.theta[0], atol=1e-6)
    assert u.members['background_dropped'] == [{'B': 0}, {'B': 0}]
    assert u.members['best_nu'].shape == (2, len(u.nominal.best.nu))
    assert np.allclose(u.members['best_nu'], u.nominal.best.nu, atol=1e-6)
    assert len(u.bootstrap['best']) == 1 and all(u.bootstrap['success'])

    inputs = mixture_inputs(beta=0.0)
    target = target_weights(inputs, theta_t)
    args = dict(profile_wcs=['cB'], theta_t=theta_t, theta_base=theta_t)
    with_b = scan(inputs, target, ['cA'], grid, box, background_target=background_target_weights(inputs), **args)
    inputs.backgrounds = None
    without = scan(inputs, target, ['cA'], grid, box, **args)
    assert np.allclose(with_b.q, without.q, atol=1e-8)
    assert without.background_keep is None


def test_uncertainty_selection():
    """scan_uncertainty selects with every member only when member or subset scans run: a bootstrap-only nominal is
    the plain scan (same selection and q), and with member scans the events a member makes non-positive are dropped
    from the signal and background selections."""
    theta_t, nu_t = np.array([0.7, -0.4]), np.array([0.3, 0.5])
    box = np.array([[-0.5, 2.0], [-1.5, 1.0]])
    grid = [np.linspace(0.3, 1.1, 9)]
    inputs = mixture_inputs()
    n = len(inputs.w0)
    member = inputs.Gamma.clone()
    member[:, :5] = -member[:, :5]  # r_hat < 0 everywhere for the first five events in this member
    inputs.seed_Gamma = {0: inputs.Gamma, 1: member}
    inputs.backgrounds.sets['B'].seed_Gamma = {0: inputs.Gamma, 1: member}
    target = target_weights(inputs, theta_t, nu_t)
    background_target = background_target_weights(inputs, nu_t)
    args = dict(pois=['cA'], grids=grid, box=box, profile_wcs=['cB'], theta_base=theta_t)
    plain = scan(inputs, target, theta_t=theta_t, background_target=background_target, **args)
    assert plain.keep.all() and plain.background_keep['B'].all()

    boot = scan_uncertainty(
        inputs, target, args, member_scans=False, n_boot=1, background_target=background_target, theta_t=theta_t
    )
    assert torch.equal(boot.nominal.keep, plain.keep)
    assert torch.equal(boot.nominal.background_keep['B'], plain.background_keep['B'])
    assert np.allclose(boot.nominal.q, plain.q, rtol=0.0, atol=1e-10)
    assert boot.members is None and len(boot.bootstrap['best']) == 1 and all(boot.bootstrap['success'])

    strict = scan_uncertainty(inputs, target, args, background_target=background_target, theta_t=theta_t)
    expected = torch.arange(n) >= 5
    assert torch.equal(strict.nominal.keep, expected)
    assert torch.equal(strict.nominal.background_keep['B'], expected)
    assert strict.members['background_dropped'] == [{'B': 5}, {'B': 5}]


def impact_scan():
    """
    An Asimov scan of cA with a background (mixture_inputs) at theta_t = (0.7, -0.4), nu_t = (0.3, 0.5), cB and both
    rate systematics profiled, set up from a scan entry with impacts: true as main sets it up (check_scans,
    scan_settings, scan_targets), and its impacts.
    """
    inputs = mixture_inputs()
    entry = {
        'pois': ['cA'],
        'grid': {'cA': {'range': [0.3, 1.1], 'points': 17}},
        'target': {'theta': {'cA': 0.7, 'cB': -0.4}, 'nu': {'lumi': 0.3, 'smxsec_B': 0.5}},
        'nu_obs': 'target',
        'theta_base': 'target',
        'box': {'cB': [-1.5, 1.0]},
        'profile_wcs': ['cB'],
        'profile_nu': 'all',
        'impacts': True,
    }
    config = {'wcs': WCS, 'box': {}, 'scans': {'i': entry}}
    check_scans(config)
    scan_args, target_spec, spec = scan_settings(config, 'i', inputs)
    target, background_target = scan_targets(inputs, target_spec)
    s = scan(inputs, target, theta_t=target_spec['theta_t'], background_target=background_target, **scan_args)
    imp = impacts(inputs, target, s, scan_args, background_target=background_target)
    return types.SimpleNamespace(inputs=inputs, theta_t=target_spec['theta_t'], spec=spec, scan=s, impacts=imp)


def test_impacts_gaussian():
    """At an Asimov best fit the pulls vanish and the data constrain both rate systematics (0 < sigma_hat < 1); every
    impact fit converges at or above the best fit's NLL; near the Gaussian minimum the symmetric part of each impact,
    (up - down) / 2, is Cov(cA, nu_n) / sigma_hat_n (the lnN rate systematics make the two sides differ by up to about
    10%), and the Hessian sigma_hat of cA is the 68% half-width of the scan."""
    r = impact_scan()
    imp, s = r.impacts, r.scan
    assert imp.names == ['lumi', 'smxsec_B']
    assert imp.free_names == ['cA', 'cB', 'lumi', 'smxsec_B']
    assert np.allclose(imp.nu_obs, [0.3, 0.5])
    assert np.allclose(imp.pull, 0.0, atol=1e-3) and np.allclose(imp.constrained_pull, 0.0, atol=1e-2)
    assert ((imp.sigma > 0) & (imp.sigma < 1)).all() and not imp.at_bound.any()
    assert imp.min_eigenvalue > 0
    assert imp.success_up.all() and imp.success_down.all() and not imp.lower_minimum.any()
    assert np.allclose(np.diag(imp.correlation), 1.0)
    for i, n in enumerate(imp.names):
        expected = imp.covariance[0, imp.free_names.index(n)] / imp.sigma[i]
        symmetric = (imp.impact_up[i, 0] - imp.impact_down[i, 0]) / 2
        assert abs(symmetric - expected) < 0.02 * abs(expected)
        assert abs(imp.impact_up[i, 0] - expected) < 0.2 * abs(expected)
        assert abs(imp.impact_down[i, 0] + expected) < 0.2 * abs(expected)
    lower, upper = s.intervals[0.6827]
    assert abs(imp.poi_sigma[0] - (upper - lower) / 2) < 0.05 * imp.poi_sigma[0]
    assert imp.poi_hat[0] == s.best.theta[0]


def test_impacts_config_and_results(tmp_path):
    """check_scans refuses impacts on two pois, without profiled nuisances or not a boolean; write_results writes the
    impacts block to scans.yml and the covariance and correlation to scans.npz."""
    base = {'pois': ['cA'], 'grid': {'cA': {'range': [0.0, 1.0], 'points': 3}}, 'profile_nu': ['lumi'], 'impacts': True}
    check_scans({'wcs': WCS, 'scans': {'ok': base}})
    two = dict(base, pois=['cA', 'cB'], grid={'cA': base['grid']['cA'], 'cB': base['grid']['cA']})
    for bad in (two, dict(base, profile_nu=[]), dict(base, impacts='yes')):
        with pytest.raises(SystemExit):
            check_scans({'wcs': WCS, 'scans': {'bad': bad}})

    r = impact_scan()
    inputs, imp = r.inputs, r.impacts
    inputs.events = np.ones(len(inputs.w0), dtype=bool)
    inputs.backgrounds.yields = [float(inputs.backgrounds.sets['B'].w.sum())]
    config = {'ensemble': 'eft.yml', 'data': 'data', 'sm_yield': 1.0, 'rate_systematics': {}}
    entries = {
        'i': dict(scan=r.scan, theta_t=r.theta_t, spec=r.spec, uncertainty=None, background_calibration=None, impacts=imp)
    }
    name = str(tmp_path / 'scans')
    write_results(name, config, inputs, entries)
    with open(f'{name}/scans.yml') as f:
        out = yaml.safe_load(f)['scans']['i']['impacts']
    assert list(out['nuisances']) == ['lumi', 'smxsec_B']
    assert out['failed'] == 0 and out['lower_minimum'] == 0 and out['free_names'] == imp.free_names
    lumi = out['nuisances']['lumi']
    assert lumi['impact_up']['cA'] == pytest.approx(imp.impact_up[0, 0], rel=1e-12)
    assert lumi['sigma'] == pytest.approx(imp.sigma[0], rel=1e-12) and lumi['success'] and not lumi['at_bound']
    assert out['poi_sigma']['cA'] == pytest.approx(imp.poi_sigma[0], rel=1e-12)
    arrays = np.load(f'{name}/scans.npz')
    assert np.allclose(arrays['i/impacts_covariance'], imp.covariance, rtol=0.0, atol=0.0)
    assert np.allclose(arrays['i/impacts_correlation'], imp.correlation, rtol=0.0, atol=0.0)


def test_calibrated_inputs():
    """Calibrated inputs: a fold's values are the other folds' bin ratios, background events of the same fold and bin
    share them, the background term moves to the SM reference (b / g0_hat, rate kept), an Asimov scan covers its target,
    and the calibration keys are checked and resolved."""
    inputs = mixture_inputs(n=4000)
    inputs.scale = 0.05
    inputs.backgrounds.sets['B'].w *= 0.05
    n = len(inputs.w0)
    folds = np.arange(n) % 4
    cal = calibrated_inputs(inputs, 'cA', folds, n_bins=(4, 3), background_folds={'B': folds})

    # by hand, fold 0: bins of (g1 / g0, g2 / g0) on cA's axis (g0 = 1 here), edges from folds 1-3
    coefs = inputs.coefs.numpy()
    w_sm = coefs[:, 0]
    g = inputs.Gamma.numpy()
    stats = np.stack([g[1] / g[0], g[2] / g[0]])
    fit, apply = folds != 0, folds == 0
    bins_fit, bins_apply = nested_bins(stats[:, fit], w_sm[fit], stats[:, apply], (4, 3))
    den = np.bincount(bins_fit, weights=w_sm[fit], minlength=12)
    num = np.stack([np.bincount(bins_fit, weights=coefs[fit, q], minlength=12) for q in range(6)], 1)
    assert np.allclose(cal.Gamma.numpy()[:, apply], (num / den[:, None])[bins_apply].T, rtol=1e-12)
    s = cal.backgrounds.sets['B']
    assert torch.allclose(s.Gamma, cal.Gamma)
    assert torch.equal(cal.mask, inputs.keep_q) and cal.seed_Gamma is None
    # theta_0 = SM and g0_hat = 1 here: the background term is unchanged
    assert torch.allclose(cal.backgrounds.signal_b, inputs.backgrounds.signal_b, rtol=1e-12)
    assert torch.allclose(s.b, inputs.backgrounds.sets['B'].b, rtol=1e-12)

    # a learned g0_hat != 1 (the same rows of Gamma on the signal and background events): b / g0_hat, rate kept
    moved = mixture_inputs(n=4000)
    moved.Gamma[0] *= 1 + 0.5 * torch.tanh(moved.coefs[:, 3])
    cal_moved = calibrated_inputs(moved, 'cA', folds, n_bins=(4, 3), background_folds={'B': folds})
    b, g0 = moved.backgrounds.signal_b[0].numpy(), moved.Gamma[0].numpy()
    expected = b / g0 * (moved.w0.numpy() * b).sum() / (w_sm * b / g0).sum()
    assert np.allclose(cal_moved.backgrounds.signal_b[0].numpy(), expected, rtol=1e-12)
    assert np.allclose(cal_moved.backgrounds.sets['B'].b[0].numpy(), expected, rtol=1e-12)
    rate = (cal_moved.w0 * cal_moved.backgrounds.signal_b[0]).sum()
    assert rate.item() == pytest.approx((moved.w0 * moved.backgrounds.signal_b[0]).sum().item(), rel=1e-12)

    # background calibration: a flattened term b^0.8 (same rate) misses the background yield in bins of b and closes
    # after calibrated_backgrounds (the background events are the signal events here, w = scale w0 b)
    b_true = inputs.backgrounds.signal_b[0].numpy()
    w0 = inputs.w0.numpy()
    flat = b_true**0.8 * (w0 * b_true).sum() / (w0 * b_true**0.8).sum()
    flattened = types.SimpleNamespace(**vars(inputs))
    flattened.backgrounds = types.SimpleNamespace(
        **{
            **vars(inputs.backgrounds),
            'signal_b': torch.from_numpy(flat[None].copy()),
            'sets': {'B': types.SimpleNamespace(**{**vars(s), 'b': torch.from_numpy(flat[None].copy())})},
        }
    )
    cal_b = calibrated_backgrounds(flattened, folds, {'B': folds}, n_bins=10)
    assert torch.equal(flattened.backgrounds.signal_b[0], torch.from_numpy(flat))
    fixed = cal_b.backgrounds.signal_b[0].numpy()
    assert np.allclose(cal_b.backgrounds.sets['B'].b[0].numpy(), fixed, rtol=1e-12)
    record = cal_b.backgrounds.calibration['B']
    assert record['factors'].shape == (4, 10) and record['edges'].shape == (4, 9) and not record['unfilled'].any()
    bins = np.searchsorted(np.quantile(b_true, np.arange(1, 10) / 10), b_true, side='right')
    truth = np.bincount(bins, weights=w0 * b_true, minlength=10)
    before = np.bincount(bins, weights=w0 * flat, minlength=10) / truth - 1
    after = np.bincount(bins, weights=w0 * fixed, minlength=10) / truth - 1
    assert np.abs(before).max() > 0.05
    assert np.abs(after).max() < 0.02
    assert (w0 * fixed).sum() == pytest.approx((w0 * b_true).sum(), rel=2e-3)

    theta_t, nu_t = np.array([0.7, -0.4]), np.array([0.3, 0.5])
    box = np.array([[-0.5, 2.0], [-1.5, 1.0]])
    r = scan(
        cal, target_weights(cal, theta_t, nu_t), ['cA'], [np.linspace(0.3, 1.1, 9)], box, profile_wcs=['cB'],
        profile_nu=['lumi', 'smxsec_B'], nu_obs=nu_t, theta_t=theta_t, theta_base=theta_t,
        background_target=background_target_weights(cal, nu_t),
    )
    assert r.q_target < 1.0 and r.stable and r.success.all()

    entry = {'pois': ['cA'], 'grid': {'cA': {'range': [0.0, 1.0], 'points': 3}}, 'ratio': 'calibrated'}
    config = {'wcs': WCS, 'box': {}, 'scans': {'c': entry}}
    check_scans(config)
    args, _, spec = scan_settings(config, 'c', types.SimpleNamespace(wcs=WCS, nuisances=[], rate_names=[]))
    assert args['ratio'] == 'learned' and spec['calibration'] == {'wc': 'cA', 'bins': [10, 10], 'folds': 5}
    grid2 = {'cA': entry['grid']['cA'], 'cB': entry['grid']['cA']}
    for bad in (
        dict(entry, uncertainty={'members': True}),
        dict(entry, ratio='learned', calibration={'wc': 'cA'}),
        dict(entry, pois=['cA', 'cB'], grid=grid2),
    ):
        with pytest.raises(SystemExit):
            check_scans(dict(config, scans={'c': bad}))


def test_curve_segments():
    """Regions below a level of a sampled curve: a V-shaped curve gives one segment with its exact crossings (linear
    interpolation is exact on a curve linear between the bracketing points), two wells give two segments in order,
    and a run reaching the grid edge is open there."""
    x = np.linspace(-1.0, 1.0, 21)
    assert np.allclose(curve_segments(x, 4 * np.abs(x - 0.2), 1.0), [[-0.05, 0.45]])
    wells = 10 * np.minimum(np.abs(x + 0.5), np.abs(x - 0.5))
    assert np.allclose(curve_segments(x, wells, 1.0), [[-0.6, -0.4], [0.4, 0.6]])
    edge = curve_segments(x, 2 * np.abs(x + 1.0), 1.0)
    assert len(edge) == 1 and edge[0][0] is None and edge[0][1] == pytest.approx(-0.5)


def test_summary_series_and_plot(tmp_path):
    """scan_series reads the stored regions of 1D scan outputs and refuses a scan of another WC; combine_series reads a
    Combine tree (best-fit row dropped) at the chi2_1 levels; summary_plot draws series with open ends, two segments,
    a hollow 68% bar, WCs a series lacks and a title."""
    import uproot

    out = tmp_path / 'scan_cA'
    os.makedirs(out)
    stored = {0.6827: [[-0.2, 0.3]], 0.95: [[None, 0.5]]}
    with open(out / 'scans.yml', 'w') as f:
        yaml.safe_dump({'scans': {'s': {'pois': ['cA'], 'segments': stored}}}, f)
    learned = scan_series({'cA': (str(out), 's')}, 'learned', '#2a78d6', '-')
    assert learned['segments'] == {'cA': stored}
    with pytest.raises(ValueError):
        scan_series({'cB': (str(out), 's')}, 'learned', '#2a78d6', '-')

    x = np.linspace(-1.0, 1.0, 41)
    path = tmp_path / 'higgsCombine.cB.MultiDimFit.root'
    with uproot.recreate(path) as f:
        f['limit'] = {
            'cB': np.concatenate([[0.0], x]),
            'deltaNLL': np.concatenate([[5.0], 0.5 * (x / 0.25) ** 2]),
            'quantileExpected': np.concatenate([[-1.0], np.zeros(len(x))]),
        }
    combine = combine_series({'cB': str(path)}, 'Combine', '#eb6834', '--')
    assert np.allclose(combine['segments']['cB'][0.6827], [[-0.25, 0.25]], atol=1e-4)
    assert np.allclose(combine['segments']['cB'][0.95], [[-0.49, 0.49]], atol=5e-3)

    combine['segments']['cA'] = {0.6827: [[-0.6, -0.4], [0.4, 0.6]], 0.95: [[-0.7, 0.7]]}
    combine['hollow'] = True
    outname = tmp_path / 'summary.png'
    summary_plot(
        ['cA', 'cB', 'cQj31'], [learned, combine], str(outname), lumi=138, note='synthetic', title='Others profiled'
    )
    assert outname.stat().st_size > 0
