import importlib

import numpy as np
import pytest
import torch
import yaml

from topsbi.tools.data import get_lower_tri

MODULES = [
    'topsbi.model.net',
    'topsbi.tools.build_ratio',
    'topsbi.tools.data',
    'topsbi.tools.metrics',
    'topsbi.tools.plots',
    'topsbi.train',
    'topsbi.validation',
]


@pytest.mark.parametrize('module', MODULES)
def test_import(module):
    importlib.import_module(module)


def test_get_lower_tri_matches_processor_ordering():
    """get_lower_tri must use the same (i, j<=i) ordering as expand_array in ttbarEFT's tensor_processor."""
    wcs = [1.0, 2.0, 3.0, 5.0]
    expected = [wcs[i] * wcs[j] for i in range(len(wcs)) for j in range(i + 1)]
    assert get_lower_tri(wcs, dtype=torch.float64).tolist() == expected
    point = [1.0, 0.1, -0.3, 1.7]
    assert torch.equal(get_lower_tri(point, dtype=torch.float64),
                       get_lower_tri(torch.tensor(point, dtype=torch.float64), dtype=torch.float64))


def test_eft_training(train_config, tmp_path):
    from topsbi.train import main

    main(train_config)

    out = tmp_path / 'out'
    assert (out / 'model.pt').exists()
    performance = yaml.safe_load((out / 'complete' / 'performance.yml').read_text())
    assert 0.0 <= performance['auc'] <= 1.0
    for key in ['r_pull', 'r_pull_err', 'r_ci_diff', 'r_ci_diff_err', 'rate_closure', 'rate_pull', 'r_chi2_ndf', 'r_z',
                'r_rms_bias', 'group_chi2_ndf', 'group_z', 'group_rms_bias', 'group_worst_pull', 'feature_chi2_ndf',
                'feature_rms_bias', 'nominal_r_chi2_ndf', 'best_epoch']:
        assert key in performance
    assert len(performance['r_ci_diff']) == 3
    for plot in ['group_closure.png', 'r_closure.png', 'r_spread.png', 'kinematics/feat0.png']:
        assert (out / 'complete' / plot).exists()
    assert (out / 'complete' / 'animations' / 'feat0.gif').exists()


def test_holdout_training(holdout_config, tmp_path, capsys):
    """With holdout set, the final metrics are computed on it rather than on the early-stopping test split."""
    from topsbi.train import main

    main(holdout_config)

    assert '[INFO] final metrics on holdout' in capsys.readouterr().out
    performance = yaml.safe_load((tmp_path / 'out' / 'complete' / 'performance.yml').read_text())

    # recompute on the holdout directly; the metrics do not depend on the overall weight scale
    from topsbi.model.net import Net
    from topsbi.tools.data import prepare_weights
    from topsbi.tools.plots import network_plots

    from conftest import N_FEATURES

    net = Net(N_FEATURES, 'cpu', holdout_config['network'])
    net.load_state_dict(torch.load(tmp_path / 'out' / 'model.pt'))
    features, w0, w1 = prepare_weights(torch.load(holdout_config['holdout'], weights_only=False), holdout_config)
    network_plots(features, w0, w1, net, [1.0], [1.0], str(tmp_path / 'direct'))
    direct = yaml.safe_load((tmp_path / 'direct' / 'performance.yml').read_text())
    for key in ['auc', 'r_pull', 'r_rms_bias', 'r_chi2_ndf']:
        assert performance[key] == pytest.approx(direct[key], abs=1e-4)


def test_get_weights_stitched():
    """eft weights are L_y * S @ T(c): the per-sample normalization in S is kept, not divided out by cg."""
    from conftest import N_WCS
    from topsbi.tools.data import get_weights

    gen = torch.Generator().manual_seed(1)
    coefs = torch.rand(50, (N_WCS + 1) * (N_WCS + 2) // 2, generator=gen, dtype=torch.float64) * 1e-8
    years = torch.randint(0, 4, (50,), generator=gen).float()
    c1 = [1.0] + [0.0] * N_WCS
    c1[4] = 0.1
    config = {'c0': [1.0] + [0.0] * N_WCS, 'c1': c1, 'lumi': [19.52, 16.81, 41.48, 59.83], 'device': 'cpu'}

    w0, w1 = get_weights(coefs, config, years)

    lumi = torch.tensor(config['lumi'], dtype=torch.float64)[years.long()]
    for w, c in [(w0, config['c0']), (w1, c1)]:
        assert w.dtype == torch.float32
        torch.testing.assert_close(w.double(), lumi * (coefs @ get_lower_tri(c, dtype=torch.float64)) * 1000, rtol=1e-6, atol=0)

    with pytest.raises(KeyError):
        get_weights(coefs, {k: v for k, v in config.items() if k != 'lumi'}, years)


def test_find_zero_terms_keeps_tail_terms():
    """a term that is negligible inclusively but large in a tail is kept; a term that is negligible everywhere is not"""
    from topsbi.tools.data import find_zero_terms

    n = 10000
    x = torch.linspace(0, 1, n, dtype=torch.float64)[:, None]  # one feature; columns: sm, a, a*a, b, b*a, b*b
    coefs = torch.full((n, 6), 0.1, dtype=torch.float64)
    coefs[:, 0] = 1.0
    coefs[:, 1] = 1e-8  # a: negligible everywhere
    coefs[:, 4] = 0.0
    coefs[-n // 100 :, 4] = 1e-5  # b*a: only in the top 1% of x; inclusive ratio 1e-7, top 5% bin 2e-6
    lumi = torch.ones(n, dtype=torch.float64)

    inclusive = (coefs.abs().sum(0) / coefs[:, 0].abs().sum()).tolist()
    assert inclusive[1] < 1e-6 and inclusive[4] < 1e-6  # an inclusive criterion drops both
    assert find_zero_terms(coefs, ['a', 'b'], lumi, x) == [['a', 'sm']]
    assert find_zero_terms(coefs, ['a', 'b'], lumi, x, n_bins=1) == [['a', 'sm'], ['b', 'a']]  # one bin: inclusive


def test_expansion_appends_background_strengths():
    """E(theta) = T(theta) without background strengths; with them, each strength is one column after T(theta)"""
    from topsbi.tools.data import expansion

    theta = [1.0, 0.5, -2.0, 3.0]
    assert torch.equal(expansion(theta, 3), get_lower_tri(theta))
    E = expansion(theta + [0.7, 0.0], 3, dtype=torch.float64)
    assert E.dtype == torch.float64 and len(E) == 10 + 2
    assert torch.equal(E[:10], get_lower_tri(theta, dtype=torch.float64))
    assert E[10:].tolist() == [0.7, 0.0]
    with pytest.raises(ValueError):
        expansion([1.0, 0.5], 3)


def test_masks_with_background_processes():
    """background columns follow T(theta), are always kept and cannot be zero terms; the data width is checked"""
    from topsbi.tools.data import kept_columns, zero_term_mask

    wcs, zero_terms = ['a', 'b'], [['b', 'a']]
    plain = zero_term_mask(wcs, zero_terms)
    mask = zero_term_mask(wcs, zero_terms, ['bg'])
    assert len(plain) == 6 and len(mask) == 7
    assert torch.equal(mask[:6], plain) and mask[6] and not mask[4]
    with pytest.raises(ValueError):
        zero_term_mask(wcs, [['bg', 'sm']], ['bg'])

    config = {'device': 'cpu', 'wcs': wcs, 'background_processes': ['bg']}
    assert kept_columns(config, 7).all()
    with pytest.raises(ValueError):
        kept_columns(config, 6)
    assert kept_columns({'device': 'cpu'}, 6).all()  # neither zero_terms nor background_processes: no check


def test_get_weights_background_strength():
    """a background event weighs mu b: at mu = 1 the weights are those of the concatenated samples, at mu = 0 zero"""
    from topsbi.tools.data import get_weights

    g = torch.Generator().manual_seed(0)
    n_s, n_b = 200, 50
    signal = torch.randn(n_s, 6, generator=g, dtype=torch.float64)  # wcs a, b: columns sm, a, a*a, b, b*a, b*b
    b = torch.rand(n_b, generator=g, dtype=torch.float64) - 0.25  # negative weights allowed
    years = torch.randint(0, 2, (n_s + n_b,), generator=g).to(torch.float32)
    plain = {'device': 'cpu', 'lumi': [1.5, 2.0], 'c0': [1.0, 0.5, -1.0], 'c1': [1.0, 2.0, 0.3]}

    # without the background column: background events are flat in theta (SM column only)
    concatenated = torch.cat([signal, torch.zeros(n_b, 6, dtype=torch.float64)])
    concatenated[n_s:, 0] = b
    w0_ref, w1_ref = get_weights(concatenated, plain, years)

    coefs = torch.zeros(n_s + n_b, 7, dtype=torch.float64)
    coefs[:n_s, :6] = signal
    coefs[n_s:, 6] = b
    config = {**plain, 'wcs': ['a', 'b'], 'background_processes': ['bg']}
    w0, w1 = get_weights(coefs, {**config, 'c0': plain['c0'] + [1.0], 'c1': plain['c1'] + [1.0]}, years)
    assert torch.equal(w0, w0_ref) and torch.equal(w1, w1_ref)

    v0, v1 = get_weights(coefs, {**config, 'c0': plain['c0'] + [1.0], 'c1': plain['c1'] + [0.0]}, years)
    assert torch.equal(v0, w0_ref) and torch.equal(v1[:n_s], w1_ref[:n_s])
    assert (v1[n_s:] == 0).all()

    with pytest.raises(ValueError):
        get_weights(coefs, {**config, 'c0': plain['c0'] + [1.0], 'c1': plain['c1']}, years)


def test_find_zero_terms_ignores_backgrounds():
    """background events and columns do not change the zero terms, also where a bin holds background events only"""
    from topsbi.tools.data import find_zero_terms

    n_s, n_b = 2000, 500
    g = torch.Generator().manual_seed(1)
    x = torch.rand(n_s, 2, generator=g, dtype=torch.float64)
    signal = torch.zeros(n_s, 6, dtype=torch.float64)
    signal[:, 0], signal[:, 1], signal[:, 3], signal[:, 5] = 1.0, x[:, 0], 1e-9, 0.3
    expected = find_zero_terms(signal, ['a', 'b'], torch.ones(n_s), x)
    assert expected == [['a', 'a'], ['b', 'sm'], ['b', 'a']]

    coefs = torch.zeros(n_s + n_b, 7, dtype=torch.float64)
    coefs[:n_s, :6] = signal
    coefs[n_s:, 6] = 2.0
    features = torch.cat([x, torch.full((n_b, 2), 5.0, dtype=torch.float64)])  # background-only top bin
    lumi = torch.ones(n_s + n_b)
    assert find_zero_terms(coefs, ['a', 'b'], lumi, features, background_processes=['bg']) == expected
    with pytest.raises(ValueError):
        find_zero_terms(coefs, ['a', 'b'], lumi, features)


def test_check_config_background_processes():
    """background_processes: a list of names for eft trainings with wcs; c0 / c1 hold SM, WCs and strengths"""
    from topsbi.train import check_config

    base = {'method': 'eft', 'wcs': ['a', 'b'], 'c0': [1.0, 0.5, 0.5], 'c1': [1.0, 0.0, 0.0]}
    check_config(base)
    check_config({**base, 'background_processes': ['bg'], 'c0': base['c0'] + [1.0], 'c1': base['c1'] + [0.0]})
    with pytest.raises(ValueError):  # strength given, process not declared
        check_config({**base, 'c0': base['c0'] + [1.0], 'c1': base['c1'] + [0.0]})
    with pytest.raises(ValueError):  # process declared, strength missing
        check_config({**base, 'background_processes': ['bg']})
    for bad in ('bg', [], ['bg', 'bg'], [1]):
        with pytest.raises(ValueError):
            check_config({**base, 'background_processes': bad})
    with pytest.raises(KeyError):
        check_config({'method': 'eft', 'background_processes': ['bg'], 'c0': [1.0, 1.0], 'c1': [1.0, 0.0]})
    nuisance = {'method': 'weight_shift', 'data': 'd', 'systematic': 's', 'direction': 'up', 'lumi': [1.0], 'seed': 0}
    with pytest.raises(ValueError):
        check_config({**nuisance, 'wcs': ['a'], 'background_processes': ['bg']})


SHARED_KEYS = ['auc', 'rate_closure', 'rate_pull', 'r_chi2_ndf', 'r_z', 'r_rms_bias', 'r_pull', 'r_ci_diff',
               'nominal_r_pull', 'group_chi2_ndf', 'feature_chi2_ndf', 'feature_rms_bias', 'nominal_r_chi2_ndf',
               'best_epoch', 'best_test_loss']


def check_nuisance_outputs(out, paired):
    assert (out / 'model.pt').exists()
    performance = yaml.safe_load((out / 'complete' / 'performance.yml').read_text())
    for key in SHARED_KEYS:
        assert key in performance
    assert (out / 'complete' / 'r_spread.png').exists() == paired  # needs the per-event true ratio
    for plot in ['r_closure.png', 'group_closure.png', 'kinematics/feat0.png', 'animations/feat0.gif', 'loss.png']:
        assert (out / 'complete' / plot).exists()
    # the epoch-0 summary has no loss or learning-rate plot
    assert not any((out / 'incomplete' / 'epoch_0000' / plot).exists() for plot in ('loss.png', 'learning_rate.png'))
    assert (out / 'incomplete' / 'epoch_0000' / 'r_closure.png').exists()
    losses = np.load(out / 'complete' / 'loss.npz')
    assert len(losses['train']) == len(losses['test']) == 3  # before training and after each of the 2 epochs
    # frames after 0, 1 and 2 completed epochs, 1 s each, then the best network held 3 s (merged by Pillow into one)
    from PIL import Image

    gif = Image.open(out / 'complete' / 'animations' / 'feat0.gif')
    durations = []
    for i in range(gif.n_frames):
        gif.seek(i)
        durations.append(gif.info['duration'])
    assert gif.n_frames == 4 and durations[0] == 1000 and durations[-1] == 3000
    return performance


def test_weight_shift_training(nuisance_config, tmp_path):
    from topsbi.train import main

    main({**nuisance_config, 'method': 'weight_shift', 'systematic': 'ISR'})
    check_nuisance_outputs(tmp_path / 'nuisance', paired=True)


def test_kinematic_shift_training(nuisance_config, tmp_path):
    from topsbi.train import main

    main({**nuisance_config, 'method': 'kinematic_shift', 'systematic': 'JES_X'})
    performance = check_nuisance_outputs(tmp_path / 'nuisance', paired=False)
    assert len(performance['feature_rms_bias']) == len(performance['feature_chi2_ndf'])


def test_kinematic_shift_keeps_variation_eras(nuisance_config):
    """A variation produced for one era only is trained against the nominal events of that era."""
    from topsbi.tools.data import load_nuisance_data

    config = {**nuisance_config, 'method': 'kinematic_shift', 'systematic': 'JER_2017'}
    for split in load_nuisance_data(config).values():
        assert torch.all(split['nominal'][0][:, 21] == 2)
        assert torch.all(split['shifted'][0][:, 21] == 2)

    # validation.p with other eras than train.p is a broken sample
    path = f"{config['data']}/JER_2017Up/validation.p"
    features, w_ref, ids, event_id = torch.load(path, weights_only=False)[:]
    features = features.clone()
    features[:, 21] = 1
    torch.save(torch.utils.data.TensorDataset(features, w_ref, ids, event_id), path)
    with pytest.raises(ValueError, match='eras'):
        load_nuisance_data(config)


def test_merged_weight_systematics(tmp_path):
    """A list of per-era weight systematics gives their columns; w1 = w0 times the product of their ratios."""
    from topsbi.tools.data import syst_ratio_column, weight_shift_sample

    (tmp_path / 'nominal').mkdir()
    (tmp_path / 'nominal' / 'systematics.yml').write_text(
        yaml.safe_dump({'syst_ratio': ['ISR', 'b_2017', 'b_2018'], 'directions': ['Up', 'Down']})
    )
    assert syst_ratio_column(str(tmp_path), ['b_2017', 'b_2018'], 'down') == ([1, 2], 1)
    assert syst_ratio_column(str(tmp_path), 'b_2018', 'Up') == (2, 0)
    with pytest.raises(ValueError, match='unknown'):
        syst_ratio_column(str(tmp_path), ['b_2017', 'b_2016'], 'up')

    features = torch.zeros(4, 22)
    features[2:, 21] = 1  # events 0, 1 in era 0, events 2, 3 in era 1
    syst_ratio = torch.ones(4, 3, 2)
    syst_ratio[:2, 1, 1] = 0.9  # b_2017 Down acts on the era-0 events only
    syst_ratio[2:, 2, 1] = 1.2  # b_2018 Down acts on the era-1 events only
    tensors = [features, None, None, None, syst_ratio, torch.ones(4, 1)]
    config = {'lumi': [1.0, 1.0], 'device': 'cpu'}
    _, w0, w1 = weight_shift_sample(tensors, config, [1, 2], 1)
    assert torch.allclose(w1 / w0, torch.tensor([0.9, 0.9, 1.2, 1.2]))

    # an event shifted by two of the merged systematics breaks the per-era factorisation
    syst_ratio[0, 2, 1] = 1.1
    with pytest.raises(ValueError, match='more than one'):
        weight_shift_sample(tensors, config, [1, 2], 1)


def test_kinematic_shift_merged_eras(nuisance_config):
    """Per-era variations trained as one: the union of their events against the nominal events of their eras."""
    import os

    from topsbi.tools.data import load_nuisance_data

    root = nuisance_config['data']
    os.makedirs(f'{root}/JER_2018Up')
    for split in ('train', 'validation'):
        features, _, ids, event_id, _, w_ref = torch.load(f'{root}/nominal/{split}.p', weights_only=False)[:]
        rows = features[:, 21] == 3
        shifted = features[rows].clone()
        shifted[:, 0] *= 1.01
        torch.save(
            torch.utils.data.TensorDataset(shifted, w_ref[rows], ids[rows], event_id[rows]),
            f'{root}/JER_2018Up/{split}.p',
        )

    kinematic = {**nuisance_config, 'method': 'kinematic_shift'}
    single = {name: load_nuisance_data({**kinematic, 'systematic': name}) for name in ('JER_2017', 'JER_2018')}
    merged = load_nuisance_data({**kinematic, 'systematic': ['JER_2017', 'JER_2018']})
    for key, split in merged.items():
        assert set(split['nominal'][0][:, 21].tolist()) == {2.0, 3.0}
        for sample in ('nominal', 'shifted'):
            assert len(split[sample][0]) == sum(len(single[name][key][sample][0]) for name in single)

    # the same era in two merged variations
    with pytest.raises(ValueError, match='also in another'):
        load_nuisance_data({**kinematic, 'systematic': ['JER_2017', 'JER_2017']})


@pytest.mark.parametrize('systematic', [[], ['JER_2017', 'JER_2017'], ['JER_2017', 3], 5])
def test_check_config_systematic(nuisance_config, systematic):
    """'systematic' is a name or a non-empty list of distinct names."""
    from topsbi.train import check_config

    with pytest.raises(ValueError, match="'systematic'"):
        check_config({**nuisance_config, 'method': 'kinematic_shift', 'systematic': systematic})


def test_background_training(background_config, tmp_path):
    from topsbi.train import hypothesis, main

    main(background_config)
    performance = check_nuisance_outputs(tmp_path / 'background', paired=False)
    assert performance['process'] == 'tW' and performance['reference'] == 'sm'
    assert performance['reference_point'] == [1.0] + [0.0] * 16
    assert hypothesis(background_config) == r'\mathrm{tW}'


def test_background_data(background_config):
    """signal rows labelled 0, background rows 1 with their negative weights kept; event indices disjoint"""
    from topsbi.tools.data import load_background_data

    splits = load_background_data(background_config)
    for split in splits.values():
        features, weights, labels = split['rows']
        (x0, w0, i0), (x1, w1, i1) = split['nominal'], split['shifted']
        assert torch.equal(features, torch.cat([x0, x1])) and torch.equal(weights, torch.cat([w0, w1]))
        assert torch.equal(labels, torch.cat([torch.zeros(len(x0)), torch.ones(len(x1))]))
        assert (w0 > 0).all() and (w1 < 0).any()
        assert torch.equal(torch.cat([i0, i1]), torch.arange(len(x0) + len(x1)))
    # train.p of each sample is split into train and test; validation.p of both is the holdout
    assert len(splits['train']['nominal'][0]) + len(splits['test']['nominal'][0]) == 2000
    assert len(splits['train']['shifted'][0]) + len(splits['test']['shifted'][0]) == 1000
    assert len(splits['holdout']['nominal'][0]) == 1000 and len(splits['holdout']['shifted'][0]) == 500


def test_background_rejects_swapped_samples(background_config):
    from topsbi.tools.data import load_background_data
    from topsbi.train import check_config

    swapped = {**background_config, 'data': background_config['background'], 'background': background_config['data']}
    with pytest.raises(ValueError, match='not a background sample'):
        load_background_data(swapped)
    with pytest.raises(KeyError, match='background'):
        check_config({key: value for key, value in background_config.items() if key != 'background'})


@pytest.mark.parametrize('method', ['eft', 'weight_shift', 'kinematic_shift', 'background'])
def test_closure_inputs(method, train_config, nuisance_config, background_config):
    """closure_inputs gives closure_metrics' sample arguments for both data kinds"""
    import numpy as np

    from topsbi.model.net import Model
    from topsbi.tools.metrics import closure_metrics
    from topsbi.tools.data import get_feature_normalization
    from topsbi.train import closure_inputs, load_training_data, nominal_features

    if method == 'eft':
        config = train_config
    elif method == 'background':
        config = background_config
    else:
        config = {**nuisance_config, 'method': method, 'systematic': 'ISR' if method == 'weight_shift' else 'JES_X'}
    kind, splits = load_training_data(config)
    net = Model(22, config['method'], 'cpu', config.get('network'), config['seed']).net
    # as in main: without the input standardisation an untrained network saturates on raw features
    net.set_feature_normalization(*get_feature_normalization(nominal_features(kind, splits['train'])))
    f0, (x0, w0, i0), f1, (x1, w1, i1) = closure_inputs(net, kind, splits['test'])
    assert len(f0) == len(x0) == len(w0) == len(i0) and len(f1) == len(x1) == len(w1) == len(i1)
    assert all(isinstance(a, np.ndarray) for a in (f0, x0, w0, i0, f1, x1, w1, i1))
    if kind == 'paired':
        x, _, _ = splits['test']
        assert len(f0) == len(x)  # untrained networks give finite f_hat everywhere
        assert np.array_equal(f0, f1) and np.array_equal(x0, x1) and np.array_equal(i0, np.arange(len(x)))
    else:
        assert kind == 'labelled' and method in ('kinematic_shift', 'background')
        assert len(f1) == len(splits['test']['shifted'][0])  # every shifted row is kept
    closure_metrics(f0, (x0, w0, i0), f1, (x1, w1, i1), {})


def test_hypothesis_and_plot_luminosity():
    import numpy as np

    from topsbi.train import hypothesis, plot_luminosity

    assert hypothesis({'method': 'eft'}) == r'\theta'
    assert hypothesis({'method': 'weight_shift'}) == hypothesis({'method': 'kinematic_shift'}) == r'\theta,\nu'
    lumi = [19.52, 16.81, 41.48, 59.83]
    x = np.zeros((8, 22))
    x[:, 21] = [0, 1, 2, 3, 3, 2, 1, 0]
    assert plot_luminosity({'lumi': lumi}, x) == 138
    x[:, 21] = 2  # a variation produced for one era
    assert plot_luminosity({'lumi': lumi}, x) == 41
    # the era column follows year_loc
    x = np.zeros((4, 22))
    x[:, 5] = [0, 3, 3, 0]
    assert plot_luminosity({'lumi': lumi, 'year_loc': 5}, x) == 79


def test_eft_combined_training(train_config, nuisance_config, tmp_path):
    """EFT on the nominal directory of a combined sample: event_split train/test, validation.p as the holdout"""
    from topsbi.tools.data import event_split, load_combined
    from topsbi.train import load_training_data, main

    nominal = f"{nuisance_config['data']}/nominal"
    config = {**train_config, 'data': nominal, 'name': str(tmp_path / 'eft_combined')}
    del config['bootstrap']
    _, splits = load_training_data(config)
    rows = load_combined(f'{nominal}/train.p', 'nominal', ('features', 'ids', 'event_id'))
    test = event_split(rows, 'nominal', config['seed'])
    assert len(splits['test'][0]) == int(test.sum()) and len(splits['train'][0]) == int((~test).sum())
    assert len(splits['holdout'][0]) == len(load_combined(f'{nominal}/validation.p', 'nominal', ('features',))[0])
    main(config)
    out = tmp_path / 'eft_combined' / 'complete'
    for plot in ['r_closure.png', 'kinematics/feat0.png', 'animations/feat0.gif', 'performance.yml']:
        assert (out / plot).exists()


def test_closure_options():
    from topsbi.train import check_config, closure_options

    lumi = [19.52, 16.81, 41.48, 59.83]
    default = closure_options({'lumi': lumi})
    assert default['group_cols'] == (21, 19) and default['group_caps'] == {19: 4}
    assert default['group_labels'][19] == ('jets', {4: '≥4'})
    assert default['min_neff'] == 30
    assert closure_options({'lumi': lumi, 'method': 'kinematic_shift'})['min_neff'] == 100
    # without eras, the eras are labelled by their index
    assert default['value_names'][21] == {0: '0', 1: '1', 2: '2', 3: '3'} == default['group_labels'][21][1]
    named = closure_options({'lumi': lumi, 'eras': ['2016APV', '2016', '2017', '2018']})
    assert named['value_names'][21][0] == '2016APV' == named['group_labels'][21][1][0]
    custom = closure_options(
        {'lumi': [1, 2], 'year_loc': 5, 'eras': ['A', 'B'], 'closure_groups': [{'loc': 5}, {'loc': 7, 'cap': 2}]}
    )
    assert custom['group_cols'] == (5, 7) and custom['group_caps'] == {7: 2}
    assert custom['value_names'] == {5: {0: 'A', 1: 'B'}} and custom['group_labels'][7] == ('', {2: '≥2'})
    for bad in (
        {'eras': ['A'], 'lumi': [1, 2]},
        {'closure_groups': [{'loc': 5}, {'loc': 5}]},
        {'closure_groups': [{'column': 5}]},
        {'threads': 0},
    ):
        with pytest.raises(ValueError):
            check_config({'method': 'eft', **bad})


def test_moved_era_column_and_threads(train_config, tmp_path):
    """era index in column 5 instead of 21, custom era names and groups, one thread"""
    from threadpoolctl import threadpool_info

    from topsbi.train import main

    features, coefs, ids = torch.load(f"{train_config['data']}/train.p", weights_only=False)[:]
    features = features.clone()
    features[:, [5, 21]] = features[:, [21, 5]]
    data = tmp_path / 'moved'
    data.mkdir()
    torch.save(torch.utils.data.TensorDataset(features, coefs, ids), data / 'train.p')
    config = {
        **train_config,
        'data': str(data),
        'name': str(tmp_path / 'moved_out'),
        'year_loc': 5,
        'eras': ['A', 'B', 'C', 'D'],
        'closure_groups': [{'loc': 5}],
        'threads': 1,
    }
    main(config)
    assert (tmp_path / 'moved_out' / 'complete' / 'group_closure.png').exists()
    assert torch.get_num_threads() == 1
    assert all(pool['num_threads'] == 1 for pool in threadpool_info())


def test_epoch_batches():
    from topsbi.train import epoch_batches

    n, size = 1003, 100
    x, w = torch.arange(n).float()[:, None].repeat(1, 3), torch.arange(n).float()

    def order(seed):
        batches = list(epoch_batches((x, w), size, torch.Generator().manual_seed(seed)))
        assert [len(b[0]) for b in batches] == [size] * (n // size) + [n % size]
        assert all(torch.equal(b[0][:, 0], b[1]) for b in batches)  # rows of every tensor stay aligned
        return torch.cat([b[1] for b in batches])

    first = order(0)
    assert torch.equal(first.sort().values, w)  # every row exactly once
    assert torch.equal(order(0), first) and not torch.equal(order(1), first)
