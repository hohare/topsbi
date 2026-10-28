from matplotlib.transforms import blended_transform_factory
from topsbi.tools.data import get_lower_tri, kept_columns, lumi_per_event
from topsbi.tools.ensembleCache import EnsembleCache
from topsbi.tools.nuisanceCache import NuisanceCache
from topsbi.tools.plots import _no_tex_mapping, wc_tex

import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh
import scipy.optimize
import scipy.stats

import argparse, datetime, glob, logging, os, sys, time, torch, types, yaml

# taken from the EFT validation config, so the scans use the events, reference point and luminosities of the ensemble
SHARED_KEYS = ('data', 'c0', 'wcs', 'lumi', 'year_loc', 'zero_terms')
# first three slots of the categorical palette (tools/plots.py), the ones that validate all-pairs: a scan and up to
# two overlays; line styles as secondary encoding
SCAN_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
SCAN_STYLES = ['-', '--', '-.']
LEVEL_STYLES = ['solid', 'dashed', 'dotted']
# keys of a scan entry in the config's scans
SCAN_KEYS = (
    'pois',
    'grid',
    'box',
    'target',
    'nu_obs',
    'ratio',
    'rate',
    'profile_wcs',
    'profile_nu',
    'selection_wcs',
    'nu_bound',
    'cls',
    'max_selections',
    'overlay',
    'uncertainty',
    'label',
)
TARGET_KEYS = ('theta', 'nu', 'kind', 'seed')
UNCERTAINTY_KEYS = ('members', 'n_boot', 'subsets', 'seed')
WILKS_KEYS = ('scan', 'toys', 'first_seed', 'randomize_nu_obs')


def load_config(config_path):
    """
    The scan config, completed from the EFT validation config it names (ensemble): data, c0, wcs, lumi, year_loc and
    zero_terms are taken from there (a scan config that sets one of them stops the run). eft_cache is {name}/ensemble
    of that config; nuisance_cache is {name}/nuisance of the nuisance validation config (nuisance), if one is named.

    Args:
        config_path: scan config yml (name, ensemble, sm_yield, scans; optional nuisance, processes, process_column,
            rate_systematics, box, n_threads, plot_lumi; the keys of a scan entry are listed in check_scans)
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
    if 'sm_yield' not in config:
        sys.exit('[error]: sm_yield, the expected SM yield of the analysis, is required')
    config['eft_cache'] = f'{eft["name"]}/ensemble'
    if config.get('nuisance'):
        with open(config['nuisance']) as f:
            config['nuisance_cache'] = f'{yaml.safe_load(f)["name"]}/nuisance'
    config.setdefault('processes', ['all'])
    config.setdefault('rate_systematics', {})
    config['device'] = 'cpu'
    return config


def rate_log_kappa(rate_systematics, processes, process):
    """
    log kappa of every rate systematic for each event: the yield of process p at nu_k = +-1 is scaled by
    kappa_{k,p}^Up or kappa_{k,p}^Down, interpolated in nu_k like the shape nuisances (nuisanceCache.interpolate,
    which gives nu_k log kappa for a symmetric kappa). kappa is one value (Down = 1 / kappa) or [Down, Up]; a
    systematic without processes applies to every process.

    Args:
        rate_systematics: {name: {'kappa': float or [down, up], optional 'processes': [process names]}}
        processes: process names, in the order of the process index
        process: [N] process index of each event
    Returns:
        names: rate systematic names, row k of log_kappa
        log_kappa: [n_rate, 2, N] float64 log kappa^Down, log kappa^Up of each event's process (0 if not affected)
    """
    names = list(rate_systematics)
    process = np.asarray(process)
    log_kappa = np.zeros((len(names), 2, len(process)))
    for k, (name, syst) in enumerate(rate_systematics.items()):
        affected = syst.get('processes', processes)
        unknown = sorted(set(affected) - set(processes))
        if unknown:
            sys.exit(f'[error]: rate systematic {name}: processes {unknown} are not in processes {processes}')
        kappa = syst['kappa']
        down, up = (1 / kappa, kappa) if np.isscalar(kappa) else kappa
        events = np.isin(process, [processes.index(p) for p in affected])
        log_kappa[k][:, events] = np.log([down, up])[:, None]
    return names, log_kappa


def load_inputs(config, members=False):
    """
    The inputs of every scan, on the events kept by the EFT ensemble and, if named, the nuisance ensemble: the weight
    coefficients coefs = L_era s_q(z) x 1000 (so w^s(theta) = coefs . T(theta)[keep_q]), w0 = w^s(theta_0), era and
    process of each event, the ensemble's Gamma and mask, the nuisances' f_hat, the rate systematics' log kappa, and
    scale = sm_yield / sum_i w^s_i(theta_SM), which brings yields of config['data'] to the expected yield.

    Args:
        config: scan config (load_config)
        members: also read every member's Gamma, on the same events
    Returns:
        SimpleNamespace: coefs [N, n_kept], keep_q (kept columns of T(theta) for coefs), w0 [N], era [N],
        process [N], Gamma [K, N], mask (kept columns of T(theta) for Gamma), seed_Gamma {e: [K, N]} (members only),
        f_hat [n_n, 2, N], nuisances, log_kappa [n_rate, 2, N], rate_names, processes, wcs, scale, events (kept
        events of config['data']); arrays are float64 torch tensors (era, process: int64; masks: bool)
    """
    eft = EnsembleCache(config['eft_cache'])
    if eft.manifest['data'] != config['data'] or eft.manifest['c0'] != config['c0']:
        sys.exit(f'[error]: {config["eft_cache"]} was built from other data or c0 than {config["ensemble"]}')
    keep = eft.finite.copy()
    nuisance = None
    if 'nuisance_cache' in config:
        nuisance = NuisanceCache(config['nuisance_cache'])
        if nuisance.manifest['ensemble'] != config['ensemble'] or nuisance.theta_ref != config['c0']:
            sys.exit(f'[error]: {config["nuisance_cache"]} was not built from {config["ensemble"]} at c0')
        keep &= nuisance.finite
    print(f'{keep.sum()} of {len(keep)} events kept by every member')

    def gamma(G, finite):
        return torch.from_numpy(np.asarray(G[:, keep[finite]], dtype=np.float64))

    tensors = torch.load(config['data'], weights_only=False, mmap=True).tensors
    rows = torch.from_numpy(keep)
    era = tensors[0][rows, config.get('year_loc', 21)].long()
    keep_q = kept_columns(config, tensors[1].shape[1])
    coefs = tensors[1][rows][:, keep_q].to(torch.float64) * (lumi_per_event(era, config) * 1000)[:, None]
    w0 = coefs @ get_lower_tri(config['c0'], dtype=torch.float64)[keep_q]
    sm = [1.0] + [0.0] * len(config['wcs'])
    scale = config['sm_yield'] / (coefs @ get_lower_tri(sm, dtype=torch.float64)[keep_q]).sum().item()

    processes = config['processes']
    if config.get('process_column'):
        tensor, column = config['process_column']
        process = tensors[tensor][rows, column].long()
        if process.min() < 0 or process.max() >= len(processes):
            sys.exit(f'[error]: process index outside 0..{len(processes) - 1} in {config["process_column"]}')
    elif len(processes) == 1:
        process = torch.zeros(len(era), dtype=torch.int64)
    else:
        sys.exit('[error]: process_column is required with more than one process')
    for p, n in zip(processes, torch.bincount(process, minlength=len(processes)).tolist()):
        if n == 0:
            print(f'[warning]: no events of process {p}')
    rate_names, log_kappa = rate_log_kappa(config['rate_systematics'], processes, process.numpy())

    out = types.SimpleNamespace(
        coefs=coefs,
        keep_q=keep_q,
        w0=w0,
        era=era,
        process=process,
        Gamma=gamma(eft.Gamma, eft.finite),
        mask=torch.from_numpy(eft.mask),
        seed_Gamma={e: gamma(eft.seed_Gamma[e], eft.seed_finite[e]) for e in eft.seeds} if members else None,
        f_hat=None,
        nuisances=[],
        log_kappa=torch.from_numpy(log_kappa),
        rate_names=rate_names,
        processes=processes,
        wcs=list(config['wcs']),
        scale=scale,
        events=keep,
    )
    if nuisance is not None:
        out.f_hat = torch.from_numpy(np.asarray(nuisance.f_hat[..., keep[nuisance.finite]], dtype=np.float64))
        out.nuisances = [n['name'] for n in nuisance.nuisances]
    return out


def nuisance_terms(f_hat, log_kappa):
    """
    Slope and symmetric parts of every nuisance, shape (f_hat) then rate (log kappa), so that the interpolation of
    nuisanceCache.interpolate is sum_n a_n nu_n + b_n h(nu_n) (step_h), with a = (Up - Down) / 2, b = (Up + Down) / 2.

    Args:
        f_hat: [n_n, 2, N] f_hat^Down, f_hat^Up of the shape nuisances, or None
        log_kappa: [n_rate, 2, N] log kappa^Down, log kappa^Up of the rate nuisances
    Returns:
        A, B: [n_n + n_rate, N] float64
    """
    terms = log_kappa if f_hat is None else torch.cat([f_hat, log_kappa])
    return (terms[:, 1] - terms[:, 0]) / 2, (terms[:, 1] + terms[:, 0]) / 2


def step_h(nu):
    """h(nu) = (3 nu^6 - 10 nu^4 + 15 nu^2) / 8 for |nu| <= 1, |nu| outside (Combine's smooth step times nu)."""
    return torch.where(nu.abs() <= 1, (3 * nu**6 - 10 * nu**4 + 15 * nu**2) / 8, nu.abs())


def log_ratio(theta, nu, Gamma, mask, A, B):
    """
    log r_hat(x; theta, nu) = log[T(theta)[mask] . Gamma] + sum_n a_n nu_n + b_n h(nu_n) for every event, with
    theta = (theta_1, ..., theta_n) without the SM entry and nu in the order of nuisance_terms; differentiable in
    theta and nu. Events with T(theta)[mask] . Gamma <= 0 give nan (positive_events removes them beforehand).

    Args:
        theta: [n_wcs] float64 tensor
        nu: [n_nu] float64 tensor (shape nuisances, then rate nuisances); empty for none
        Gamma: [K, N] ensemble mean or one member's Gamma
        mask: kept columns of T(theta)
        A, B: [n_nu, N] from nuisance_terms
    Returns:
        [N] float64 log r_hat
    """
    T = get_lower_tri(torch.cat([theta.new_ones(1), theta]), dtype=torch.float64)[mask]
    return torch.log(T @ Gamma) + nu @ A + step_h(nu) @ B


def _box_minimum(Gamma, mask, box):
    """
    Minimum of r_hat(x; theta) = T(theta)[mask] . Gamma over box for every event, exact for at most two moving WCs:
    r_hat is quadratic in the moving WCs phi, r_hat = c + g . phi + phi . Q phi, with c, g, Q read off from r_hat
    at a few points; its minimum over the box is at a corner, at the vertex along an edge or at the interior
    stationary point.

    Args:
        Gamma: [K, N]
        mask: kept columns of T(theta)
        box: [n_wcs, 2] float64 tensor, lower / upper value of each WC
    Returns:
        [N] float64 minimum of r_hat over the box
    """
    moving = torch.nonzero(box[:, 0] != box[:, 1]).flatten()
    base = box[:, 0].clone()
    base[moving] = 0.0

    def r(phi):
        theta = base.clone()
        theta[moving] = torch.as_tensor(phi, dtype=torch.float64)
        return get_lower_tri(torch.cat([theta.new_ones(1), theta]), dtype=torch.float64)[mask] @ Gamma

    c = r([0.0] * len(moving))
    if len(moving) == 0:
        return c
    if len(moving) > 2:
        raise ValueError(f'the exact minimum needs at most two moving WCs, the box has {len(moving)}')

    def vertex(g, q, lo, hi):
        """argmin over [lo, hi] of g phi + q phi^2 (lo when the minimum is at an endpoint only)"""
        return torch.where(q > 0, (-g / (2 * torch.where(q > 0, q, 1.0))).clamp(lo, hi), torch.full_like(g, lo))

    lo, hi = box[moving, 0].tolist(), box[moving, 1].tolist()
    if len(moving) == 1:
        up, down = r([1.0]), r([-1.0])
        g, q = (up - down) / 2, (up + down) / 2 - c

        def quad(p):
            return c + g * p + q * p**2

        ends = [quad(torch.full_like(c, v)) for v in (lo[0], hi[0])]
        return torch.stack(ends + [quad(vertex(g, q, lo[0], hi[0]))]).min(0).values
    up1, down1, up2, down2 = r([1.0, 0.0]), r([-1.0, 0.0]), r([0.0, 1.0]), r([0.0, -1.0])
    g1, q11 = (up1 - down1) / 2, (up1 + down1) / 2 - c
    g2, q22 = (up2 - down2) / 2, (up2 + down2) / 2 - c
    q12 = r([1.0, 1.0]) - c - g1 - g2 - q11 - q22

    def quad(p1, p2):
        return c + g1 * p1 + g2 * p2 + q11 * p1**2 + q22 * p2**2 + q12 * p1 * p2

    def full(v):
        return torch.full_like(c, v)

    candidates = [quad(full(a), full(b)) for a in (lo[0], hi[0]) for b in (lo[1], hi[1])]
    for a in (lo[0], hi[0]):  # edges phi_1 = a: minimise over phi_2
        candidates.append(quad(full(a), vertex(g2 + q12 * a, q22, lo[1], hi[1])))
    for b in (lo[1], hi[1]):  # edges phi_2 = b: minimise over phi_1
        candidates.append(quad(vertex(g1 + q12 * b, q11, lo[0], hi[0]), full(b)))
    det = 4 * q11 * q22 - q12**2  # interior stationary point: [[2 q11, q12], [q12, 2 q22]] phi = -g
    safe = torch.where(det != 0, det, 1.0)
    p1, p2 = (-2 * q22 * g1 + q12 * g2) / safe, (-2 * q11 * g2 + q12 * g1) / safe
    inside = (det != 0) & (p1 >= lo[0]) & (p1 <= hi[0]) & (p2 >= lo[1]) & (p2 <= hi[1])
    candidates.append(torch.where(inside, quad(p1, p2), candidates[0]))
    return torch.stack(candidates).min(0).values


def _quadratic_form(Gamma, mask, n_wcs):
    """
    T(theta)[mask] . Gamma as a quadratic in theta (without the SM entry): c [N], lin [N, n_wcs] and the symmetric
    Q [N, n_wcs, n_wcs] with T(theta)[mask] . Gamma = c + lin . theta + theta^T Q theta.

    Args:
        Gamma: [K, N] coefficients (rows: the kept columns of T(theta))
        mask: kept columns of T(theta) (bool mask or indices)
        n_wcs: number of WCs
    Returns:
        c, lin, Q (float64 tensors)
    """
    terms = [(i, j) for i in range(n_wcs + 1) for j in range(i + 1)]
    kept = np.zeros(len(terms), dtype=bool)
    kept[np.asarray(mask)] = True
    N = Gamma.shape[1]
    c = torch.zeros(N, dtype=torch.float64)
    lin = torch.zeros(N, n_wcs, dtype=torch.float64)
    Q = torch.zeros(N, n_wcs, n_wcs, dtype=torch.float64)
    for row, k in enumerate(np.flatnonzero(kept)):
        i, j = terms[k]
        g = Gamma[row].to(torch.float64)
        if i == 0:
            c += g
        elif j == 0:
            lin[:, i - 1] += g
        elif i == j:
            Q[:, i - 1, i - 1] += g
        else:
            Q[:, i - 1, j - 1] += g / 2
            Q[:, j - 1, i - 1] += g / 2
    return c, lin, Q


def _box_certified(Gamma, mask, box, chunk=50000):
    """
    Events whose T(theta)[mask] . Gamma is certainly positive over a box in which more than two WCs move: as the
    quadratic c + lin . theta + theta^T Q theta (_quadratic_form), positive for every theta (Q positive definite and
    its minimum c - lin^T Q^-1 lin / 4 > 0), or with a lower bound over the box > 0 (the exact minimum of each WC's
    linear and square terms over its interval plus each cross term at the worst corner of its pair's rectangle). An
    event kept is positive everywhere in the box; some events dropped may be positive too.

    Args:
        Gamma: [K, N] coefficients
        mask: kept columns of T(theta)
        box: [n_wcs, 2] lower / upper value of each WC
        chunk: events handled at once
    Returns:
        keep: [N] bool tensor
    """
    box = torch.as_tensor(box, dtype=torch.float64)
    lo, hi = box[:, 0], box[:, 1]
    n_wcs = len(box)
    keep = torch.empty(Gamma.shape[1], dtype=torch.bool)
    for start in range(0, Gamma.shape[1], chunk):
        c, lin, Q = _quadratic_form(Gamma[:, start : start + chunk], mask, n_wcs)
        e, V = torch.linalg.eigh(Q)
        definite = e[:, 0] > 1e-12 * e.abs().max(1).values.clamp(min=1e-300)
        proj = (V.transpose(1, 2) @ lin[:, :, None]).squeeze(-1)
        minimum = c - 0.25 * (proj**2 / torch.where(definite[:, None], e, torch.ones_like(e))).sum(1)
        bound = c.clone()
        diag = torch.diagonal(Q, dim1=1, dim2=2)
        for a in range(n_wcs):
            L, S = lin[:, a], diag[:, a]
            ends = (lo[a].expand_as(L), hi[a].expand_as(L))
            vertex = torch.where(S > 0, torch.clamp(-L / (2 * S), lo[a], hi[a]), ends[0])
            bound += torch.stack([L * x + S * x**2 for x in (*ends, vertex)]).min(0).values
            for b in range(a):
                g = 2 * Q[:, a, b]
                bound += torch.stack([g * x * y for x in (lo[a], hi[a]) for y in (lo[b], hi[b])]).min(0).values
        keep[start : start + chunk] = (definite & (minimum > 0)) | (bound > 0)
    return keep


def positive_events(Gammas, mask, box=None, points=None, chunk=64):
    """
    Events whose T(theta)[mask] . Gamma (the learned ratio r_hat(x; theta) for the ensemble mean and, if used, each
    member; or w^s(theta) with coefs.T and keep_q) is positive for every Gamma in Gammas: over the whole box, exactly
    when at most two WCs move (_box_minimum) and conservatively when more move (_box_certified: every event kept is
    positive in the box), and at every point in points. Nuisances do not change the sign of r_hat.

    Args:
        Gammas: list of [K, N] Gamma
        mask: kept columns of T(theta)
        box: [n_wcs, 2] lower / upper value of each WC (fixed WCs: lower = upper), or None
        points: [n_points, n_wcs] WC points, or None
        chunk: points evaluated at once
    Returns:
        keep: [N] bool tensor
    """
    keep = torch.ones(Gammas[0].shape[1], dtype=torch.bool)
    for Gamma in Gammas:
        if box is not None:
            box_t = torch.as_tensor(box, dtype=torch.float64)
            if int((box_t[:, 0] != box_t[:, 1]).sum()) <= 2:
                keep &= _box_minimum(Gamma, mask, box_t) > 0
            else:
                keep &= _box_certified(Gamma, mask, box_t)
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float64)
            for start in range(0, len(points), chunk):
                block = points[start : start + chunk]
                ones = block.new_ones(len(block), 1)
                T = torch.stack([get_lower_tri(p, dtype=torch.float64)[mask] for p in torch.cat([ones, block], 1)])
                keep &= ((T @ Gamma) > 0).all(0)
    return keep


def nll_terms(inputs, keep, target, Gamma=None, nu_obs=None):
    """
    The per-event arrays of one negative log-likelihood, restricted to the selected events (keep): Gamma, coefs, w0,
    the nuisance parts A, B (nuisance_terms) and the target weights w^t, and the centre nu_obs of the nuisance
    constraint (global observables). Events with w^t = 0 enter the rate only.

    Args:
        inputs: load_inputs result
        keep: [N] bool, positive_events
        target: [N] target weights w^t (Asimov: scale w^s(theta_t); toy or data: event counts), in expected-yield units
        Gamma: [K, N] Gamma to use (a member's), default the ensemble mean
        nu_obs: [n_nu] centre of the constraint exp(-|nu - nu_obs|^2 / 2) (shape, then rate nuisances); default 0,
            nu_t for an Asimov at nu_t, a draw from N(nu_t, 1) for a frequentist toy
    Returns:
        SimpleNamespace: Gamma, coefs, w0, A, B (selected events), on (selected events with w^t != 0), wt (w^t on
        them), nu_obs, mask, keep_q, scale
    """
    Gamma = inputs.Gamma if Gamma is None else Gamma
    keep = torch.as_tensor(keep, dtype=torch.bool)
    A, B = nuisance_terms(inputs.f_hat, inputs.log_kappa)
    target = torch.as_tensor(target, dtype=torch.float64)[keep]
    on = target != 0
    nu_obs = (
        torch.zeros(A.shape[0], dtype=torch.float64) if nu_obs is None else torch.as_tensor(nu_obs, dtype=torch.float64)
    )
    return types.SimpleNamespace(
        Gamma=Gamma[:, keep],
        coefs=inputs.coefs[keep],
        w0=inputs.w0[keep],
        A=A[:, keep],
        B=B[:, keep],
        on=on,
        wt=target[on],
        nu_obs=nu_obs,
        mask=inputs.mask,
        keep_q=inputs.keep_q,
        scale=inputs.scale,
    )


def nll(theta, nu, terms, ratio='learned', rate='learned'):
    """
    Negative log of the extended likelihood with a normalised shape,
    NLL(theta, nu) = lambda - W log lambda - sum_i w^t_i log r_i + W log sum_j w0_j r_j + |nu - nu_obs|^2 / 2
    (constants dropped), W = sum_i w^t_i, where r_i(theta, nu) is the per-event ratio to theta_0, lambda(theta, nu)
    the expected yield and nu_obs the centre of the nuisance constraint (nll_terms); differentiable in theta and nu.
    With rate = 'learned' (lambda = scale sum_j w0_j r_j) and nu_obs = 0 it equals eqn:ratio_scans,
    lambda - sum_i w^t_i log r_i + |nu|^2 / 2, up to a constant.
    ratio: 'learned', r_i = r_hat(x_i; theta) exp(nu-part_i) (log_ratio), or 'truth',
    r_i = w^s_i(theta) / w^s_i(theta_0) exp(nu-part_i); rate: 'learned', lambda = scale sum_j w0_j r_j, or 'truth',
    lambda = scale sum_j w^s_j(theta) exp(nu-part_j). The nuisance part (nu-part_i = sum_n a_n nu_n + b_n h(nu_n)) is
    the learned one in every case.

    Args:
        theta: [n_wcs] float64 tensor, without the SM entry
        nu: [n_nu] float64 tensor (shape nuisances, then rate nuisances)
        terms: nll_terms result
        ratio, rate: 'learned' or 'truth'
    Returns:
        scalar float64 tensor
    """
    T = get_lower_tri(torch.cat([theta.new_ones(1), theta]), dtype=torch.float64)
    shift = nu @ terms.A + step_h(nu) @ terms.B
    w_theta = terms.coefs @ T[terms.keep_q] if 'truth' in (ratio, rate) else None
    base = T[terms.mask] @ terms.Gamma if ratio == 'learned' else w_theta / terms.w0
    r = base * torch.exp(shift)
    log_r = torch.log(base[terms.on]) + shift[terms.on]
    shape = (terms.w0 * r).sum()
    lam = terms.scale * (shape if rate == 'learned' else (w_theta * torch.exp(shift)).sum())
    W = terms.wt.sum()
    constraint = ((nu - terms.nu_obs) ** 2).sum() / 2
    return lam - W * torch.log(lam) - (terms.wt * log_r).sum() + W * torch.log(shape) + constraint


def target_weights(inputs, theta_t, nu_t=None, kind='asimov', keep=None, seed=None):
    """
    Target weights w^t of a scan, in expected-yield units, from the expected weights w_i = scale w^s_i(theta_t)
    exp(nu-part_i(nu_t)) (the nuisance part is the learned one, as in nll): 'asimov' returns w; 'toy' draws
    N_t ~ Poisson(sum_i w_i) and N_t events with replacement with probability w_i / sum_j w_j, and returns each
    event's count k_i, so a toy follows the weighted distribution at (theta_t, nu_t) for any event weights.

    Args:
        inputs: load_inputs result
        theta_t: [n_wcs] target WC point, without the SM entry
        nu_t: [n_nu] target nuisances (shape, then rate), default 0
        kind: 'asimov' or 'toy'
        keep: [N] bool, events a toy is drawn from (positive_events); default every event
        seed: seed of the toy's random generator
    Returns:
        [N] float64 w^t (0 outside keep for a toy)
    """
    T = get_lower_tri([1.0] + list(theta_t), dtype=torch.float64)
    w = inputs.scale * (inputs.coefs @ T[inputs.keep_q])
    if nu_t is not None:
        nu = torch.as_tensor(nu_t, dtype=torch.float64)
        A, B = nuisance_terms(inputs.f_hat, inputs.log_kappa)
        w = w * torch.exp(nu @ A + step_h(nu) @ B)
    if kind == 'asimov':
        return w
    if kind != 'toy':
        raise ValueError(f"kind must be 'asimov' or 'toy', not {kind}")
    if keep is not None:
        w = torch.where(torch.as_tensor(keep, dtype=torch.bool), w, 0.0)
    if (w < 0).any():
        raise ValueError(f'{int((w < 0).sum())} events have negative expected weight at the toy target')
    generator = torch.Generator()
    generator.manual_seed(seed if seed is not None else torch.seed())
    n_t = int(torch.poisson(w.sum().reshape(1), generator=generator).item())
    picks = torch.multinomial(w / w.sum(), n_t, replacement=True, generator=generator)
    return torch.bincount(picks, minlength=len(w)).to(torch.float64)


def fit(
    terms,
    theta,
    nu,
    free,
    bounds,
    ratio='learned',
    rate='learned',
    hessian=False,
    ftol=1e-10,
    gtol=1e-6,
    maxiter=1000,
    edm_tol=1e-5,
    restarts=5,
):
    """
    Minimise nll over the free parameters with L-BFGS-B (scipy) and the autograd gradient (torch); the parameters are
    p = (theta, nu), fixed ones keep their start value. The NLL is minimised relative to its start value, so ftol acts
    on differences of order one. Convergence is judged by the estimated distance to the minimum,
    EDM = g^T H^-1 g / 2, with g the gradient at the end (components pointing out of the bounds removed) and H^-1
    L-BFGS-B's inverse-Hessian estimate; while EDM > edm_tol the minimisation restarts from the end point (at most
    restarts times). Every evaluated theta is recorded (visited), for positive_events.

    Args:
        terms: nll_terms result
        theta: [n_wcs] start (and fixed) WC values, without the SM entry
        nu: [n_nu] start (and fixed) nuisance values (shape, then rate)
        free: [n_wcs + n_nu] bool, parameters to minimise over
        bounds: [n_wcs + n_nu, 2] lower / upper bound of each parameter (used for free ones)
        ratio, rate: passed to nll
        hessian: also return the Hessian of the NLL in the free parameters at the minimum and its inverse
        ftol, gtol, maxiter: L-BFGS-B options
        edm_tol: largest EDM of a converged fit, in NLL units
        restarts: largest number of restarts
    Returns:
        SimpleNamespace: theta, nu (numpy, at the minimum), nll (value there), success (EDM <= edm_tol and no
        non-finite NLL), edm, max_grad, message (scipy's, last run), runs, nit, nfev, nonfinite (evaluations with a
        non-finite NLL), visited [n_eval, n_wcs], hessian and covariance ([n_free, n_free], or None; covariance None
        if the Hessian is not positive definite)
    """
    start = torch.cat([torch.as_tensor(theta, dtype=torch.float64), torch.as_tensor(nu, dtype=torch.float64)])
    n_wcs = len(theta)
    index = torch.nonzero(torch.as_tensor(free, dtype=torch.bool)).flatten()
    bounds = np.asarray(bounds, dtype=np.float64)[index.numpy()]
    visited, nonfinite = [], [0]

    def value(p):
        full = start.index_put((index,), p)
        return nll(full[:n_wcs], full[n_wcs:], terms, ratio=ratio, rate=rate), full

    x = start[index].numpy()
    nit = nfev = runs = 0
    while True:
        offset = value(torch.as_tensor(x, dtype=torch.float64))[0].item()
        if not np.isfinite(offset):
            raise ValueError('the NLL is not finite at the start point; check the event selection (positive_events)')

        def fun(y):
            p = torch.tensor(y, dtype=torch.float64, requires_grad=True)
            v, full = value(p)
            visited.append(full[:n_wcs].detach().numpy().copy())
            if not torch.isfinite(v):
                nonfinite[0] += 1
                return np.inf, np.zeros_like(y)
            v.backward()
            return v.item() - offset, p.grad.numpy().copy()

        result = scipy.optimize.minimize(
            fun,
            x,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': ftol, 'gtol': gtol, 'maxiter': maxiter},
        )
        x, runs = result.x, runs + 1
        nit, nfev = nit + int(result.nit), nfev + int(result.nfev)
        grad = np.asarray(result.jac, dtype=np.float64)
        outward = ((x <= bounds[:, 0]) & (grad > 0)) | ((x >= bounds[:, 1]) & (grad < 0))
        grad = np.where(outward, 0.0, grad)
        edm = float(grad @ result.hess_inv.matvec(grad)) / 2
        if edm <= edm_tol or runs > restarts:
            break
    best = start.index_put((index,), torch.as_tensor(x, dtype=torch.float64))
    out = types.SimpleNamespace(
        theta=best[:n_wcs].numpy(),
        nu=best[n_wcs:].numpy(),
        nll=float(result.fun) + offset,
        success=bool(edm <= edm_tol and nonfinite[0] == 0),
        edm=edm,
        max_grad=float(np.abs(grad).max(initial=0.0)),
        message=str(result.message),
        runs=runs,
        nit=nit,
        nfev=nfev,
        nonfinite=nonfinite[0],
        visited=np.array(visited),
        hessian=None,
        covariance=None,
    )
    if hessian:
        H = torch.autograd.functional.hessian(lambda p: value(p)[0], torch.as_tensor(x, dtype=torch.float64))
        out.hessian = H.numpy()
        if np.linalg.eigvalsh(out.hessian).min() > 0:
            out.covariance = np.linalg.inv(out.hessian)
    return out


def _fixed_value(terms, theta, nu, ratio, rate):
    """nll at one point when no parameter is free, in the form of a fit result."""
    theta = np.asarray(theta, dtype=np.float64).copy()
    nu = np.asarray(nu, dtype=np.float64).copy()
    value = nll(torch.as_tensor(theta), torch.as_tensor(nu), terms, ratio=ratio, rate=rate).item()
    return types.SimpleNamespace(
        theta=theta,
        nu=nu,
        nll=value,
        success=bool(np.isfinite(value)),
        edm=0.0,
        nonfinite=int(not np.isfinite(value)),
        visited=theta[None],
    )


def scan(
    inputs,
    target,
    pois,
    grids,
    box,
    profile_wcs=(),
    profile_nu=(),
    nu_bound=5.0,
    nu_obs=None,
    ratio='learned',
    rate='learned',
    Gammas=None,
    cls=(0.6827, 0.95),
    max_selections=5,
    xtol=1e-7,
    theta_t=None,
    selection_wcs=(),
    keep=None,
):
    """
    Likelihood scan of one or two WCs (pois) on a grid: at every grid point the pois are fixed and the profiled WCs and
    nuisances are fitted (fit, warm-started from the nearest point already fitted, in order of distance from the best
    fit); q = 2 (NLL - NLL_min), with NLL_min from a fit with the pois free too. When a grid point has a lower NLL than
    that best fit (a second minimum), the best fit is repeated from that grid point's fit and kept if lower
    (best_refit). With theta_t, q at the target, q_target = 2 (NLL(theta_t) - NLL_min), from one more fit with the pois
    fixed at theta_t (warm-started from the nearest grid fit). Events: the NLL takes the log of r_hat for a learned
    ratio and of w^s for a truth ratio (the rate only sums them). The selection keeps
    the events where that quantity is positive for every Gamma in Gammas (learned ratio) over the box of the moving WCs
    (the pois, the profiled WCs and selection_wcs; the other WCs at 0): exactly when at most two WCs move, and
    conservatively when more move, so that every fit inside the box stays finite (positive_events); keep gives the
    initial selection instead. The visited points of the fits are checked too; when they remove events the scan is
    repeated on the smaller selection (at most max_selections times). A warm start whose NLL is not finite is recorded
    as visited, so the next pass removes those events, and the fit starts from the profiled WCs at 0 instead
    (start_fallbacks counts them in the last pass). For one poi, the region q < chi2 quantile(cl, 1) at each confidence
    level cl is found on the grid points and the best fit: every run of consecutive points below the level is one
    segment, its ends found by brentq on the profiled q between the last point below and the next point above (the
    stored q at those two points, fits between them warm-started from the nearer one; the best fit counts as a point)
    (None where the run reaches the grid edge); segments lists them all, intervals the one containing the minimum. Other WCs
    not profiled stay at 0; nuisances not profiled stay at nu_obs (0 by default).

    Args:
        inputs: load_inputs result
        target: [N] target weights w^t (target_weights)
        pois: one or two WC names
        grids: one grid of values per poi
        box: [n_wcs, 2] lower / upper bound of every WC (the moving WCs' rows set the positivity box, the profiled
            WCs' rows bound their fits)
        profile_wcs: WC names fitted at every grid point
        profile_nu: nuisance names (shape nuisances, then rate systematics, as inputs.nuisances + inputs.rate_names)
            fitted at every grid point
        nu_bound: |nu| bound of profiled nuisances
        nu_obs: [n_nu] centre of the nuisance constraint (nll_terms), default 0
        ratio, rate: passed to nll
        Gammas: list of Gamma the event selection must keep positive (the ensemble mean and the members, when member
            scans follow), default [inputs.Gamma]; the scan itself uses inputs.Gamma
        cls: confidence levels of the intervals (one poi) and of the q levels returned (both)
        max_selections: largest number of passes over the event selection
        xtol: brentq tolerance on interval ends
        theta_t: [n_wcs] target WC point (its pois entries are used), or None for no q_target
        selection_wcs: further WC names whose box rows enter the positivity box (not fitted)
        keep: [N] bool initial event selection, or None to select from the box
    Returns:
        SimpleNamespace: pois, grid [P, n_poi], q [P], nll_min, levels {cl: q level}, intervals {cl: [lower, upper]}
        (one poi: the segment containing the minimum; None where open), segments {cl: [[lower, upper], ...]} (one poi:
        every segment, in order), q_target (None without theta_t), best (fit at the minimum: theta, nu, nll, success,
        edm), best_refit (the best fit was repeated from a lower grid point), start_fallbacks (fits of the last pass
        started from the profiled WCs at 0), theta [P, n_wcs], nu [P, n_nu], success [P], edm [P], at_bound [P] (a
        profiled parameter on its bound), best_at_bound, grid_below_best (a grid point has a lower NLL than the best
        fit), keep [N], dropped (events removed), dropped_sm_fraction (their share of the SM yield, truth weights),
        selections, stable
    """
    wcs = inputs.wcs
    names_nu = list(inputs.nuisances) + list(inputs.rate_names)
    n_wcs, n_nu = len(wcs), len(names_nu)
    if len(pois) not in (1, 2) or len(grids) != len(pois):
        raise ValueError('a scan has one or two parameters of interest, each with a grid')
    poi = [wcs.index(p) for p in pois]
    free = np.zeros(n_wcs + n_nu, dtype=bool)
    free[[wcs.index(w) for w in profile_wcs]] = True
    free[[n_wcs + names_nu.index(n) for n in profile_nu]] = True
    if free[poi].any():
        raise ValueError('a parameter of interest cannot also be profiled')
    free_best = free.copy()
    free_best[poi] = True
    box = np.asarray(box, dtype=np.float64)
    bounds = np.concatenate([box, np.tile([-nu_bound, nu_bound], (n_nu, 1))])
    theta0 = np.zeros(n_wcs)
    nu0 = np.zeros(n_nu) if nu_obs is None else np.asarray(nu_obs, dtype=np.float64).copy()
    grids = [np.asarray(g, dtype=np.float64) for g in grids]
    points = np.stack(np.meshgrid(*grids, indexing='ij'), -1).reshape(-1, len(pois))
    spacing = np.array([np.diff(g).min() if len(g) > 1 else 1.0 for g in grids])
    Gammas = [inputs.Gamma] if Gammas is None else Gammas
    levels = {cl: float(scipy.stats.chi2.ppf(cl, len(pois))) for cl in cls}
    profiled_wcs = free[:n_wcs].copy()

    def positive(**kwargs):
        if ratio == 'learned':
            return positive_events(Gammas, inputs.mask, **kwargs)
        return positive_events([inputs.coefs.T], inputs.keep_q, **kwargs)

    moving = sorted(set(poi) | {wcs.index(w) for w in profile_wcs} | {wcs.index(w) for w in selection_wcs})
    selection_box = np.zeros((n_wcs, 2))
    selection_box[moving] = box[moving]
    keep = positive(box=selection_box) if keep is None else torch.as_tensor(keep, dtype=torch.bool).clone()
    stable = False
    for selections in range(1, max_selections + 1):
        terms = nll_terms(inputs, keep, target, nu_obs=nu_obs)
        visited = []
        fallbacks = [0]

        def run(theta, nu, free_mask):
            if free_mask.any():
                start_nll = nll(torch.as_tensor(theta), torch.as_tensor(nu), terms, ratio=ratio, rate=rate)
                if not torch.isfinite(start_nll):
                    visited.append(np.asarray(theta, dtype=np.float64)[None].copy())
                    theta = np.where(profiled_wcs, 0.0, theta)
                    fallbacks[0] += 1
                f = fit(terms, theta, nu, free_mask, bounds, ratio=ratio, rate=rate)
            else:
                f = _fixed_value(terms, theta, nu, ratio, rate)
            visited.append(f.visited)
            f.visited = None
            return f

        best = run(theta0, nu0, free_best)
        fits = [None] * len(points)
        done = []
        for p in np.argsort((((points - best.theta[poi]) / spacing) ** 2).sum(1)):
            start = best
            if done:
                start = fits[done[int(np.argmin((((points[done] - points[p]) / spacing) ** 2).sum(1)))]]
            theta = start.theta.copy()
            theta[poi] = points[p]
            fits[p] = run(theta, start.nu.copy(), free)
            done.append(p)
        nll_grid = np.array([f.nll for f in fits])
        best_refit = False
        lowest = int(np.argmin(nll_grid))
        if nll_grid[lowest] < best.nll - 1e-6:
            again = run(fits[lowest].theta.copy(), fits[lowest].nu.copy(), free_best)
            if again.nll < best.nll:
                best, best_refit = again, True
        nll_min = min(best.nll, nll_grid.min())
        q = 2 * (nll_grid - nll_min)

        q_target = None
        if theta_t is not None:
            target_point = np.asarray(theta_t, dtype=np.float64)[poi]
            near = fits[int(np.argmin((((points - target_point) / spacing) ** 2).sum(1)))]
            theta = near.theta.copy()
            theta[poi] = target_point
            q_target = float(2 * (run(theta, near.nu.copy(), free).nll - nll_min))

        intervals, segments = {}, {}
        if len(pois) == 1:
            grid = points[:, 0]
            c_best = float(best.theta[poi[0]])
            x, qx, x_fits = grid, q, list(fits)
            if not np.isclose(grid, c_best, rtol=0.0, atol=1e-12 * spacing[0]).any():
                x = np.append(grid, c_best)
                qx = np.append(q, 2 * (best.nll - nll_min))
                x_fits.append(best)
            order = np.argsort(x)
            x, qx, x_fits = x[order], qx[order], [x_fits[k] for k in order]
            k_min = int(np.argmin(qx))

            def profiled_q(c, ends):
                for k in ends:
                    if c == x[k]:
                        return qx[k]
                near = x_fits[min(ends, key=lambda k: abs(x[k] - c))]
                theta = near.theta.copy()
                theta[poi] = c
                return 2 * (run(theta, near.nu.copy(), free).nll - nll_min)

            def crossing(a, b, level):
                return float(scipy.optimize.brentq(lambda c: profiled_q(c, (a, b)) - level, x[a], x[b], xtol=xtol))

            for cl, level in levels.items():
                below = qx < level
                pieces = []
                i = 0
                while i < len(x):
                    if not below[i]:
                        i += 1
                        continue
                    j = i
                    while j + 1 < len(x) and below[j + 1]:
                        j += 1
                    lower = None if i == 0 else crossing(i - 1, i, level)
                    upper = None if j == len(x) - 1 else crossing(j, j + 1, level)
                    pieces.append([lower, upper])
                    if i <= k_min <= j:
                        intervals[cl] = [lower, upper]
                    i = j + 1
                segments[cl] = pieces

        new = keep & positive(points=np.unique(np.concatenate(visited), axis=0))
        if torch.equal(new, keep):
            stable = True
            break
        keep = new

    def on_bound(f):
        x = np.concatenate([f.theta, f.nu])[free]
        lo, hi = bounds[free, 0], bounds[free, 1]
        width = 1e-9 * (hi - lo)
        return bool(((x <= lo + width) | (x >= hi - width)).any())

    w_sm = inputs.coefs @ get_lower_tri([1.0] + [0.0] * n_wcs, dtype=torch.float64)[inputs.keep_q]
    return types.SimpleNamespace(
        pois=list(pois),
        grid=points,
        q=q,
        nll_min=nll_min,
        levels=levels,
        intervals=intervals,
        segments=segments,
        q_target=q_target,
        best=types.SimpleNamespace(theta=best.theta, nu=best.nu, nll=best.nll, success=best.success, edm=best.edm),
        best_refit=best_refit,
        start_fallbacks=fallbacks[0],
        theta=np.array([f.theta for f in fits]),
        nu=np.array([f.nu for f in fits]),
        success=np.array([f.success for f in fits]),
        edm=np.array([f.edm for f in fits]),
        at_bound=np.array([on_bound(f) for f in fits]),
        best_at_bound=on_bound(best),
        grid_below_best=bool(nll_grid.min() < best.nll - 1e-6),
        keep=keep,
        dropped=int((~keep).sum()),
        dropped_sm_fraction=float(w_sm[~keep].sum() / w_sm.sum()),
        selections=selections,
        stable=stable,
    )


def scan_uncertainty(inputs, target, scan_args, member_scans=True, n_boot=0, subsets=None, seed=0, theta_t=None):
    """
    The scan of scan_args and the uncertainty of its minimum, all on one event selection: the nominal scan's, selected
    with the ensemble mean and every member (scan with Gammas = [mean] + members) and passed to the other scans as
    their initial selection. Members: the same scan with each member's Gamma; their spread of the minimum, and
    spread / sqrt(n_e), the error of the mean's minimum for independent members. Bootstrap (MC statistics of
    config['data']): n_boot scans with every event's target weight, w0 and weight coefficients multiplied by a
    Poisson(1) count. Subsets: for each n_e in subsets, that many random draws of n_e members whose mean Gamma is
    scanned, with the spread of their minima. theta_t gives the nominal scan its q_target (scan); the other scans have
    none.

    Args:
        inputs: load_inputs result with members (load_inputs(config, members=True))
        target: [N] target weights w^t (target_weights)
        scan_args: arguments of scan other than inputs, target, Gammas, theta_t and keep (pois, grids, box, ...)
        member_scans: scan every member
        n_boot: number of bootstrap scans
        subsets: {n_e: number of draws} of member subsets, or None
        seed: seed of the bootstrap counts and the subset draws
        theta_t: [n_wcs] target WC point of the nominal scan's q_target, or None
    Returns:
        SimpleNamespace: nominal (scan result), members, bootstrap, subsets (None when not run); members and bootstrap
        are dicts with best [n_runs, n_poi] (best-fit pois), intervals (per run), success (per run: the best fit and
        every grid fit converged), dropped (per run), spread [n_poi] (standard deviation of best) and, for members,
        error = spread / sqrt(n_e); subsets is {n_e: dict with the same keys and draws (member indices per run)}
    """
    if inputs.seed_Gamma is None:
        raise ValueError('member Gamma are not loaded: use load_inputs(config, members=True)')
    member_Gamma = list(inputs.seed_Gamma.values())
    args = dict(scan_args, Gammas=[inputs.Gamma] + member_Gamma)
    poi = [inputs.wcs.index(p) for p in args['pois']]

    def replica(**changes):
        return types.SimpleNamespace(**{**vars(inputs), **changes})

    def collect(results):
        best = np.array([s.best.theta[poi] for s in results])
        return dict(
            best=best,
            intervals=[s.intervals for s in results],
            success=[bool(s.success.all() and s.best.success) for s in results],
            dropped=[s.dropped for s in results],
            spread=best.std(0, ddof=1) if len(best) > 1 else np.full(len(poi), np.nan),
        )

    out = types.SimpleNamespace(
        nominal=scan(inputs, target, theta_t=theta_t, **args), members=None, bootstrap=None, subsets=None
    )
    args = dict(args, keep=out.nominal.keep)
    if member_scans:
        out.members = collect([scan(replica(Gamma=G), target, **args) for G in member_Gamma])
        out.members['error'] = out.members['spread'] / np.sqrt(len(member_Gamma))
    rng = np.random.default_rng(seed)
    if n_boot:
        results = []
        for _ in range(n_boot):
            p = torch.as_tensor(rng.poisson(1.0, len(inputs.w0)), dtype=torch.float64)
            weighted = replica(w0=inputs.w0 * p, coefs=inputs.coefs * p[:, None])
            results.append(scan(weighted, torch.as_tensor(target, dtype=torch.float64) * p, **args))
        out.bootstrap = collect(results)
    if subsets:
        out.subsets = {}
        for n_e, draws in subsets.items():
            picks = [sorted(rng.choice(len(member_Gamma), n_e, replace=False).tolist()) for _ in range(draws)]
            results = [
                scan(replica(Gamma=torch.stack([member_Gamma[i] for i in pick]).mean(0)), target, **args)
                for pick in picks
            ]
            out.subsets[n_e] = dict(collect(results), draws=picks)
    return out


def _plain(x):
    """numpy and torch values (also inside dicts, lists and tuples) as plain Python, for yaml."""
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {_plain(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _spread_summary(part):
    """yaml summary of the members or bootstrap part of scan_uncertainty."""
    out = dict(
        n=len(part['best']),
        best=part['best'][:, 0] if part['best'].shape[1] == 1 else part['best'],
        spread=part['spread'],
        all_success=all(part['success']),
    )
    if 'error' in part:
        out['error'] = part['error']
    return out


def grid_resolution(s, min_points=(5, 20)):
    """
    How well the grid of a scan resolves q. Per confidence level: the grid points with q below the level; for one
    poi, the interval half-width in grid steps (median step) and the largest difference between the level and q
    interpolated linearly between grid points at the interval ends (what a line through the grid points misses);
    for two pois, whether the region below the level reaches the grid edge (an open contour). coarse: fewer than
    min_points[0] (one poi) or min_points[1] (two pois) grid points below the lowest level.

    Args:
        s: scan result (pois, grid, q, levels, intervals)
        min_points: least grid points below the lowest level for one and for two pois
    Returns:
        dict: points_inside {cl: n}, coarse, and for one poi step, half_width_steps {cl: value or None (open)},
        interpolation_miss {cl: value or None (no closed end)}; for two pois open {cl: bool}
    """
    cls = sorted(s.levels)
    inside = {cl: int((s.q < s.levels[cl]).sum()) for cl in cls}
    out = {'points_inside': inside}
    if len(s.pois) == 1:
        grid = s.grid[:, 0]
        step = float(np.median(np.diff(grid)))
        half, miss = {}, {}
        for cl in cls:
            ends = s.intervals.get(cl, [None, None])
            closed = [e for e in ends if e is not None]
            half[cl] = float((ends[1] - ends[0]) / 2 / step) if None not in ends else None
            misses = [abs(float(np.interp(e, grid, s.q)) - s.levels[cl]) for e in closed]
            miss[cl] = max(misses) if misses else None
        out.update(step=step, half_width_steps=half, interpolation_miss=miss, coarse=inside[cls[0]] < min_points[0])
    else:
        n1 = len(np.unique(s.grid[:, 0]))
        q = s.q.reshape(n1, -1)
        edge = np.concatenate([q[0], q[-1], q[:, 0], q[:, -1]])
        out['open'] = {cl: bool((edge < s.levels[cl]).any()) for cl in cls}
        out['coarse'] = inside[cls[0]] < min_points[1]
    return out


def write_results(name, config, inputs, entries):
    """
    Write the scans to {name}/scans.yml (summary) and {name}/scans.npz (arrays, keys <label>/<array>).

    scans.yml: inputs (configs, data, sm_yield, scale, events, wcs, nuisances, rate systematics, time) and per scan
    its config entry (spec), pois, target theta_t (pois), best fit (pois, nll_min, success, edm, at a bound), q
    levels and intervals per confidence level (one poi: the segment containing the minimum) and all segments of each
    level (one poi), with a target q_target = q(theta_t) (when the scan computed it) and, for one poi, the shift
    best - theta_t, its size in units of the interval half-width on the side of theta_t (cl 0.6827; nan if that side
    is open) and the signed root sign(best - theta_t) sqrt(q_target), the profiled WCs and nuisances at the best fit,
    fit diagnostics (points, failed, on a bound, largest edm, grid below the best fit, best fit repeated from a grid
    point, fits started from the profiled WCs at 0), the grid resolution (grid_resolution), the event selection
    (dropped, dropped_sm_fraction, selections, stable) and, with scan_uncertainty, members (best per member, spread, error), bootstrap (best per replica, spread)
    and subsets (per n_e: spread, mean 68% half-width for one poi, draws).
    scans.npz: grid, q, theta, nu, success, edm, at_bound, keep (np.packbits of the selection, N events) and
    members_best, bootstrap_best, subsets_<n_e>_best when present.

    Args:
        name: output directory
        config: scan config (load_config)
        inputs: load_inputs result
        entries: {label: {'scan': scan result, 'theta_t': [n_wcs] target or None, 'spec': the scan's config entry,
            'uncertainty': scan_uncertainty result or None (its nominal is 'scan')}}
    """
    os.makedirs(name, mode=0o755, exist_ok=True)
    wcs = inputs.wcs
    names_nu = list(inputs.nuisances) + list(inputs.rate_names)
    summary = {
        'inputs': {
            'ensemble': config['ensemble'],
            'nuisance': config.get('nuisance'),
            'data': config['data'],
            'sm_yield': config['sm_yield'],
            'scale': inputs.scale,
            'events': int(inputs.events.sum()),
            'wcs': wcs,
            'nuisances': names_nu,
            'rate_systematics': config['rate_systematics'],
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
        },
        'scans': {},
    }
    arrays = {}
    for label, entry in entries.items():
        s = entry['scan']
        spec = entry.get('spec') or {}
        poi = [wcs.index(p) for p in s.pois]
        out = {
            'spec': spec,
            'pois': s.pois,
            'best': {p: s.best.theta[i] for p, i in zip(s.pois, poi)},
            'nll_min': s.nll_min,
            'best_success': s.best.success,
            'best_edm': s.best.edm,
            'best_at_bound': s.best_at_bound,
            'levels': s.levels,
            'intervals': s.intervals,
        }
        if len(poi) == 1:
            out['segments'] = s.segments
        theta_t = entry.get('theta_t')
        if theta_t is not None:
            theta_t = np.asarray(theta_t, dtype=np.float64)
            out['target'] = {p: theta_t[i] for p, i in zip(s.pois, poi)}
            q_target = getattr(s, 'q_target', None)
            if q_target is not None:
                out['q_target'] = q_target
            if len(poi) == 1:
                c, t = s.best.theta[poi[0]], theta_t[poi[0]]
                if 0.6827 in s.intervals:
                    lower, upper = s.intervals[0.6827]
                    side = (
                        (c - lower if lower is not None else np.nan)
                        if t < c
                        else (upper - c if upper is not None else np.nan)
                    )
                    out['shift'] = c - t
                    out['shift_sigma'] = (c - t) / side if c != t else 0.0
                if q_target is not None:
                    out['root_q_target'] = float(np.sign(c - t) * np.sqrt(max(q_target, 0.0)))
        profiled_wcs = spec.get('profile_wcs', [])
        profiled_nu = spec.get('profile_nu', [])
        if profiled_wcs:
            out['profiled_wcs'] = {w: s.best.theta[wcs.index(w)] for w in profiled_wcs}
        if profiled_nu:
            out['profiled_nu'] = {n: s.best.nu[names_nu.index(n)] for n in profiled_nu}
        out['fits'] = {
            'points': len(s.q),
            'failed': int((~s.success).sum()),
            'at_bound': int(s.at_bound.sum()),
            'max_edm': float(s.edm.max()),
            'grid_below_best': s.grid_below_best,
            'best_refit': s.best_refit,
            'start_fallbacks': getattr(s, 'start_fallbacks', 0),
        }
        out['grid'] = grid_resolution(s)
        out['selection'] = {
            'dropped': s.dropped,
            'dropped_sm_fraction': s.dropped_sm_fraction,
            'selections': s.selections,
            'stable': s.stable,
        }
        for key in ('grid', 'q', 'theta', 'nu', 'success', 'edm', 'at_bound'):
            arrays[f'{label}/{key}'] = np.asarray(getattr(s, key))
        arrays[f'{label}/keep'] = np.packbits(np.asarray(s.keep))

        u = entry.get('uncertainty')
        if u is not None:
            if u.members is not None:
                out['members'] = _spread_summary(u.members)
                arrays[f'{label}/members_best'] = u.members['best']
            if u.bootstrap is not None:
                out['bootstrap'] = _spread_summary(u.bootstrap)
                arrays[f'{label}/bootstrap_best'] = u.bootstrap['best']
            if u.subsets is not None:
                out['subsets'] = {}
                for n_e, sub in u.subsets.items():
                    item = {'draws': sub['draws'], 'spread': sub['spread'], 'all_success': all(sub['success'])}
                    if len(poi) == 1:
                        widths = [
                            (i[0.6827][1] - i[0.6827][0]) / 2
                            for i in sub['intervals']
                            if 0.6827 in i and None not in i[0.6827]
                        ]
                        item['mean_half_width'] = float(np.mean(widths)) if widths else np.nan
                    out['subsets'][n_e] = item
                    arrays[f'{label}/subsets_{n_e}_best'] = sub['best']
        summary['scans'][label] = out
    with open(f'{name}/scans.yml', 'w') as f:
        yaml.safe_dump(_plain(summary), f, sort_keys=False)
    np.savez_compressed(f'{name}/scans.npz', **arrays)


def _scan_label(spec):
    """Legend text of a scan: the spec's 'label', else its ratio and rate and whether it profiles."""
    if 'label' in spec:
        return spec['label']
    text = f"{spec.get('ratio', 'learned')} ratio, {spec.get('rate', 'learned')} rate"
    if spec.get('profile_wcs') or spec.get('profile_nu'):
        text += ', profiled'
    return text


def _grid_axes(grid):
    """The two grid axes of a 2D scan grid [P, 2] laid out by np.meshgrid(..., indexing='ij')."""
    n1 = len(np.unique(grid[:, 0]))
    square = grid.reshape(n1, -1, 2)
    return square[:, 0, 0], square[0, :, 1]


def _q_plot_1d(ax, scans, arrays, shown):
    """
    q against the poi for the scans in shown (first the main scan), with levels, the ends of every segment below each
    level (intervals for a scans.yml without segments), target and members. The x range covers the grid points with q
    up to the plot top (twice the highest level), the segment ends, the target and the members, one grid step wider on
    each side, within the main scan's grid.
    """
    main = scans[shown[0]]
    poi = main['pois'][0]
    levels = main['levels']
    top = 2 * max(levels.values())
    inside = []
    for other, color, style in zip(shown, SCAN_COLORS, SCAN_STYLES):
        o = scans[other]
        grid = arrays[f'{other}/grid'][:, 0]
        q = arrays[f'{other}/q']
        ax.plot(
            grid,
            q,
            color=color,
            linestyle=style,
            linewidth=2,
            marker='o',
            markersize=5,
            label=_scan_label(o['spec']),
        )
        inside.extend(grid[q <= top])
        segments = o.get('segments') or {cl: [ends] for cl, ends in o['intervals'].items()}
        for cl, pieces in segments.items():
            x = [e for piece in pieces for e in piece if e is not None]
            inside.extend(x)
            ax.plot(
                x,
                [o['levels'][cl]] * len(x),
                linestyle='none',
                marker='o',
                markersize=9,
                markerfacecolor='white',
                markeredgecolor=color,
                markeredgewidth=2,
                label='_nolegend_',
            )
    to_axes = blended_transform_factory(ax.transAxes, ax.transData)
    for cl, level in levels.items():
        ax.axhline(level, color='grey', linestyle=':', linewidth=1)
        ax.text(1.0, level, f' {100 * cl:.0f}%', transform=to_axes, va='center', fontsize=14, color='grey')
    if 'target' in main:
        inside.append(main['target'][poi])
        ax.axvline(main['target'][poi], color='k', linestyle='--', linewidth=1, label=r'$\theta_t$')
    if f'{shown[0]}/members_best' in arrays:
        members = arrays[f'{shown[0]}/members_best'][:, 0]
        inside.extend(members)
        ax.plot(
            members,
            np.zeros(len(members)),
            linestyle='none',
            marker='|',
            markersize=20,
            markeredgewidth=2,
            color=SCAN_COLORS[0],
            alpha=0.7,
            label=f'members ({len(members)})',
        )
    grid = arrays[f'{shown[0]}/grid'][:, 0]
    step = float(np.diff(np.sort(grid)).min()) if len(grid) > 1 else 0.0
    lo, hi = (min(inside) - step, max(inside) + step) if inside else (grid.min(), grid.max())
    ax.set_xlim(max(lo, grid.min()), min(hi, grid.max()))
    ax.set_ylim(-0.05 * top, top)
    ax.set_xlabel(f'${wc_tex(poi)}$')
    ax.set_ylabel(r'$-2\Delta\log\mathcal{L}$')
    ax.legend(loc='best', fontsize=14)


def _q_plot_2d(ax, scans, arrays, shown):
    """
    q contours at the levels for the scans in shown (first the main scan), with grid points, best fit and target. A
    scan with no grid point below its lowest level has no contour; a note on the plot names it.
    """
    main = scans[shown[0]]
    p1, p2 = main['pois']
    grid = arrays[f'{shown[0]}/grid']
    ax.plot(grid[:, 0], grid[:, 1], linestyle='none', marker='.', markersize=4, color='#d9d9d9', label='_nolegend_')
    missing = []
    for other, color in zip(shown, SCAN_COLORS):
        o = scans[other]
        a1, a2 = _grid_axes(arrays[f'{other}/grid'])
        q = arrays[f'{other}/q'].reshape(len(a1), len(a2))
        cls = sorted(o['levels'])
        if q.min() > o['levels'][cls[0]]:
            missing.append(_scan_label(o['spec']))
        else:
            ax.contour(
                a1,
                a2,
                q.T,
                levels=[o['levels'][cl] for cl in cls],
                colors=color,
                linestyles=LEVEL_STYLES[: len(cls)],
                linewidths=2,
            )
        ax.plot([], [], color=color, linewidth=2, label=_scan_label(o['spec']))
        ax.plot(
            o['best'][p1],
            o['best'][p2],
            linestyle='none',
            marker='P',
            markersize=11,
            color=color,
            markeredgecolor='white',
            label='_nolegend_',
        )
    for cl, style in zip(sorted(main['levels']), LEVEL_STYLES):
        ax.plot([], [], color='grey', linestyle=style, linewidth=2, label=f'{100 * cl:.0f}%')
    ax.plot([], [], linestyle='none', marker='P', markersize=11, color='grey', label='best fit')
    if 'target' in main:
        ax.plot(
            main['target'][p1],
            main['target'][p2],
            linestyle='none',
            marker='*',
            markersize=14,
            color='k',
            label=r'$\theta_t$',
        )
    if missing:
        ax.text(
            0.5,
            0.03,
            'no grid point below the lowest level: ' + '; '.join(missing),
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=12,
            color='#52514e',
        )
    ax.set_xlim(grid[:, 0].min(), grid[:, 0].max())
    ax.set_ylim(grid[:, 1].min(), grid[:, 1].max())
    ax.set_xlabel(f'${wc_tex(p1)}$')
    ax.set_ylabel(f'${wc_tex(p2)}$')
    ax.legend(loc='best', fontsize=14)


def _pull_plot(s, outname, lumi=None):
    """nu_hat - nu_obs of every profiled nuisance of one scan at its best fit, with the +-1 and +-2 bands."""
    names = list(s['profiled_nu'])
    nu_obs = s['spec'].get('nu_obs') or {}
    pull = np.array([s['profiled_nu'][n] - nu_obs.get(n, 0.0) for n in names])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 2 + 0.35 * len(names)))
    ax.axvspan(-2, 2, color='#E6EEE6', label=r'$\pm 2$')
    ax.axvspan(-1, 1, color='#4086B8', alpha=0.6, label=r'$\pm 1$')
    ax.axvline(0, color='grey', linestyle=':', linewidth=1)
    ax.plot(pull, y, linestyle='none', marker='o', markersize=7, color='k', label='_nolegend_')
    ax.set_yticks(y, names, fontsize=12)
    ax.set_ylim(len(names) - 0.5, -0.5)
    extent = max(2.5, 1.1 * float(np.abs(pull).max(initial=0)))
    ax.set_xlim(-extent, extent)
    ax.set_xlabel(r'$\hat{\nu} - \nu_{\mathrm{obs}}$')
    ax.legend(loc='lower right', fontsize=14)
    mh.cms.label('Preliminary', data=False, lumi=lumi, com=13, ax=ax)
    fig.savefig(outname, bbox_inches='tight')
    plt.close(fig)


def plot_results(name, lumi=None):
    """
    Plots of the scans in {name}/scans.yml and {name}/scans.npz (write_results), written to {name}/plots/.

    <label>.png: for one poi, q = 2 (NLL - NLL_min) at the grid points against the poi, with the q levels, the
    interval ends (open markers on their level), the target theta_t and, with scan_uncertainty, every member's best
    fit (ticks at q = 0); for two pois, the contours of q at the levels, the grid points, the best fit and theta_t.
    The scans named in the spec's 'overlay' (same pois, at most two) are drawn on the same axes; only the main scan's
    target and members are drawn. <label>_pulls.png, for a scan with profiled nuisances: nu_hat - nu_obs of each at
    the best fit (nu_obs from the spec's 'nu_obs', {name: value}, default 0), with the +-1 and +-2 bands of the
    unit-width constraint.

    Args:
        name: output directory of write_results
        lumi: integrated luminosity (fb^-1) of the CMS label, or None
    """
    with open(f'{name}/scans.yml') as f:
        scans = yaml.safe_load(f)['scans']
    arrays = dict(np.load(f'{name}/scans.npz'))
    os.makedirs(f'{name}/plots', mode=0o755, exist_ok=True)
    mh.style.use('CMS')
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        for label, s in scans.items():
            shown = [label] + list(s['spec'].get('overlay', []))
            if len(shown) > len(SCAN_COLORS):
                raise ValueError(f'{label}: at most {len(SCAN_COLORS) - 1} overlays')
            for other in shown[1:]:
                if other not in scans:
                    raise ValueError(f'{label}: overlay {other} is not a scan in {name}/scans.yml')
                if scans[other]['pois'] != s['pois']:
                    raise ValueError(f'{label}: overlay {other} scans {scans[other]["pois"]}, not {s["pois"]}')
            fig, ax = plt.subplots()
            if len(s['pois']) == 1:
                _q_plot_1d(ax, scans, arrays, shown)
            else:
                _q_plot_2d(ax, scans, arrays, shown)
            mh.cms.label('Preliminary', data=False, lumi=lumi, com=13, ax=ax)
            fig.savefig(f'{name}/plots/{label}.png', bbox_inches='tight')
            plt.close(fig)
            if s.get('profiled_nu'):
                _pull_plot(s, f'{name}/plots/{label}_pulls.png', lumi=lumi)
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)


def _grid(entry, poi, label):
    """Grid values of one poi of a scan entry: grid[poi] values, or range and points (np.linspace); increasing."""
    g = entry['grid'][poi]
    if 'values' in g:
        values = np.asarray(g['values'], dtype=np.float64)
    elif 'range' in g and 'points' in g:
        values = np.linspace(g['range'][0], g['range'][1], int(g['points']))
    else:
        sys.exit(f'[error]: scan {label}: the grid of {poi} needs values, or range and points')
    if len(values) < 2 or (np.diff(values) <= 0).any():
        sys.exit(f'[error]: scan {label}: the grid of {poi} needs at least two increasing values')
    return values


def _check_names(names, known, what, label):
    """Stop the run if a scan entry names a WC or nuisance that does not exist."""
    unknown = [n for n in names if n not in known]
    if unknown:
        sys.exit(f'[error]: scan {label}: unknown {what} {unknown}')


def check_scans(config):
    """
    Check the scan entries of a config before any input is loaded. A scan entry (config['scans'][label]) has pois
    (one or two WCs) and grid ({poi: {values: [...]} or {range: [lo, hi], points: n}}), and optional target
    ({theta: {wc: value}, nu: {name: value}, kind: asimov or toy, seed (required for a toy)}), nu_obs ('target' or
    {name: value}), ratio and rate (learned or truth), profile_wcs and profile_nu (names or 'all'), selection_wcs
    (names or 'all': further WCs whose box enters the event selection), nu_bound, cls, max_selections, box
    ({wc: [lo, hi]}, over the top-level box),
    overlay (labels of other scans), uncertainty ({members, n_boot, subsets, seed}, scan_uncertainty) and label
    (legend text).

    Args:
        config: scan config (load_config)
    """
    scans = config.get('scans') or {}
    if not scans:
        sys.exit('[error]: the config lists no scans')
    for label, entry in scans.items():
        unknown = sorted(set(entry) - set(SCAN_KEYS))
        if unknown:
            sys.exit(f'[error]: scan {label}: unknown keys {unknown}')
        pois = entry.get('pois') or []
        if len(pois) not in (1, 2):
            sys.exit(f'[error]: scan {label}: pois needs one or two WCs')
        _check_names(pois, config['wcs'], 'WCs', label)
        missing = [p for p in pois if p not in (entry.get('grid') or {})]
        if missing:
            sys.exit(f'[error]: scan {label}: no grid for {missing}')
        for poi in pois:
            _grid(entry, poi, label)
        for key in ('ratio', 'rate'):
            if entry.get(key, 'learned') not in ('learned', 'truth'):
                sys.exit(f"[error]: scan {label}: {key} must be learned or truth, not {entry[key]}")
        target = entry.get('target') or {}
        unknown = sorted(set(target) - set(TARGET_KEYS))
        if unknown:
            sys.exit(f'[error]: scan {label}: unknown target keys {unknown}')
        if target.get('kind', 'asimov') not in ('asimov', 'toy'):
            sys.exit(f"[error]: scan {label}: target kind must be asimov or toy, not {target['kind']}")
        if target.get('kind') == 'toy' and target.get('seed') is None:
            sys.exit(f'[error]: scan {label}: a toy target needs a seed')
        _check_names(target.get('theta', {}), config['wcs'], 'WCs in target', label)
        unknown = sorted(set(entry.get('uncertainty') or {}) - set(UNCERTAINTY_KEYS))
        if unknown:
            sys.exit(f'[error]: scan {label}: unknown uncertainty keys {unknown}')
        overlay = entry.get('overlay', [])
        if len(overlay) > len(SCAN_COLORS) - 1:
            sys.exit(f'[error]: scan {label}: at most {len(SCAN_COLORS) - 1} overlays')
        for other in overlay:
            if other not in scans:
                sys.exit(f'[error]: scan {label}: overlay {other} is not a scan of the config')
            if list(scans[other].get('pois', [])) != list(pois):
                sys.exit(f'[error]: scan {label}: overlay {other} does not scan {pois}')
        _check_names(entry.get('box', {}), config['wcs'], 'WCs in box', label)
    _check_names(config.get('box', {}), config['wcs'], 'WCs in box', 'top-level')


def scan_settings(config, label, inputs):
    """
    The settings of one scan entry (check_scans) with names resolved against the loaded inputs.

    Args:
        config: scan config (load_config, check_scans)
        label: scan label (key of config['scans'])
        inputs: load_inputs result
    Returns:
        scan_args (keyword arguments of scan other than inputs, target, Gammas, theta_t and keep), target (dict:
        theta_t [n_wcs], nu_t [n_nu] or None, kind, seed) and spec (the entry as written to scans.yml: profile_wcs,
        profile_nu and selection_wcs as name lists, nu_obs as {name: value} of its nonzero entries, box as the
        {wc: [lo, hi]} used)
    """
    entry = config['scans'][label]
    wcs = inputs.wcs
    names_nu = list(inputs.nuisances) + list(inputs.rate_names)
    pois = list(entry['pois'])
    grids = [_grid(entry, poi, label) for poi in pois]

    profile_wcs = entry.get('profile_wcs', [])
    profile_wcs = [w for w in wcs if w not in pois] if profile_wcs == 'all' else list(profile_wcs)
    _check_names(profile_wcs, wcs, 'WCs in profile_wcs', label)
    profile_nu = entry.get('profile_nu', [])
    profile_nu = list(names_nu) if profile_nu == 'all' else list(profile_nu)
    _check_names(profile_nu, names_nu, 'nuisances in profile_nu', label)
    selection_wcs = entry.get('selection_wcs', [])
    selection_wcs = [w for w in wcs if w not in pois] if selection_wcs == 'all' else list(selection_wcs)
    _check_names(selection_wcs, wcs, 'WCs in selection_wcs', label)

    box_used = {w: list(v) for w, v in {**config.get('box', {}), **entry.get('box', {})}.items()}
    for poi, g in zip(pois, grids):
        box_used.setdefault(poi, [float(g[0]), float(g[-1])])
    unbounded = [w for w in dict.fromkeys(profile_wcs + selection_wcs) if w not in box_used]
    if unbounded:
        sys.exit(f'[error]: scan {label}: profiled and selection WCs {unbounded} need a box entry')
    box = np.zeros((len(wcs), 2))
    for w, (lo, hi) in box_used.items():
        box[wcs.index(w)] = [lo, hi]

    target = entry.get('target') or {}
    theta_t = np.zeros(len(wcs))
    for w, v in target.get('theta', {}).items():
        theta_t[wcs.index(w)] = v
    _check_names(target.get('nu', {}), names_nu, 'nuisances in target', label)
    nu_t = np.zeros(len(names_nu))
    for n, v in target.get('nu', {}).items():
        nu_t[names_nu.index(n)] = v
    nu_obs_entry = entry.get('nu_obs') or {}
    if nu_obs_entry == 'target':
        nu_obs = nu_t.copy()
    else:
        _check_names(nu_obs_entry, names_nu, 'nuisances in nu_obs', label)
        nu_obs = np.zeros(len(names_nu))
        for n, v in nu_obs_entry.items():
            nu_obs[names_nu.index(n)] = v

    scan_args = dict(
        pois=pois,
        grids=grids,
        box=box,
        profile_wcs=profile_wcs,
        profile_nu=profile_nu,
        selection_wcs=selection_wcs,
        nu_bound=entry.get('nu_bound', 5.0),
        nu_obs=nu_obs,
        ratio=entry.get('ratio', 'learned'),
        rate=entry.get('rate', 'learned'),
        cls=tuple(entry.get('cls', (0.6827, 0.95))),
        max_selections=entry.get('max_selections', 5),
    )
    target_spec = dict(
        theta_t=theta_t,
        nu_t=nu_t if nu_t.any() else None,
        kind=target.get('kind', 'asimov'),
        seed=target.get('seed'),
    )
    spec = dict(
        entry,
        profile_wcs=profile_wcs,
        profile_nu=profile_nu,
        selection_wcs=selection_wcs,
        nu_obs={n: float(v) for n, v in zip(names_nu, nu_obs) if v != 0},
        box=box_used,
    )
    return scan_args, target_spec, spec


def check_wilks(config):
    """Check the wilks entries of a config: {label: {scan (a scan label), toys, first_seed, randomize_nu_obs}}."""
    for label, entry in (config.get('wilks') or {}).items():
        unknown = sorted(set(entry) - set(WILKS_KEYS))
        if unknown:
            sys.exit(f'[error]: wilks {label}: unknown keys {unknown}')
        if entry.get('scan') not in (config.get('scans') or {}):
            sys.exit(f'[error]: wilks {label}: scan {entry.get("scan")} is not a scan of the config')
        if int(entry.get('toys', 0)) < 1:
            sys.exit(f'[error]: wilks {label}: toys must be at least 1')


def wilks_toys(inputs, scan_args, target_spec, seeds, randomize_nu_obs=True):
    """
    q(theta_t) = 2 (NLL(theta_t) - NLL_min) on toys at the target of a scan: per seed, a toy (target_weights,
    kind='toy', drawn from the events with w^s(theta_t) > 0) and a scan of the pois on the one-point grid theta_t
    (scan: the best fit and the fit at theta_t, with the scan's profiled WCs and nuisances and its event selection).
    With randomize_nu_obs, the global observables of the profiled nuisances are drawn per toy, nu_obs ~ N(nu_t, 1)
    (the auxiliary measurements fluctuate with the toy); the other nuisances keep the scan's nu_obs.

    Args:
        inputs: load_inputs result
        scan_args: scan arguments from scan_settings (its grids are replaced by theta_t)
        target_spec: target from scan_settings (theta_t, nu_t)
        seeds: toy seeds (seed of the toy and of its nu_obs draw)
        randomize_nu_obs: draw the profiled nuisances' nu_obs per toy
    Returns:
        dict of arrays over toys: seeds, q, best [n, n_poi], success (best fit and fit at theta_t converged),
        at_bound (a fitted parameter of the best fit on its bound), dropped, and for one poi signed_root =
        sign(best - theta_t) sqrt(q)
    """
    wcs = inputs.wcs
    names_nu = list(inputs.nuisances) + list(inputs.rate_names)
    poi = [wcs.index(p) for p in scan_args['pois']]
    theta_t = target_spec['theta_t']
    nu_t = target_spec['nu_t'] if target_spec['nu_t'] is not None else np.zeros(len(names_nu))
    profiled = [names_nu.index(n) for n in scan_args['profile_nu']]
    keep = positive_events([inputs.coefs.T], inputs.keep_q, points=theta_t[None])
    args = dict(scan_args, grids=[[theta_t[i]] for i in poi])
    q, best, success, at_bound, dropped = [], [], [], [], []
    for seed in seeds:
        target = target_weights(inputs, theta_t, nu_t=target_spec['nu_t'], kind='toy', keep=keep, seed=int(seed))
        nu_obs = np.array(scan_args['nu_obs'], dtype=np.float64)
        if randomize_nu_obs and profiled:
            draw = np.random.default_rng(int(seed)).standard_normal(len(names_nu))
            nu_obs[profiled] = nu_t[profiled] + draw[profiled]
        s = scan(inputs, target, **dict(args, nu_obs=nu_obs))
        q.append(float(s.q[0]))
        best.append(s.best.theta[poi])
        success.append(bool(s.success[0] and s.best.success))
        at_bound.append(bool(s.best_at_bound))
        dropped.append(s.dropped)
    best = np.array(best)
    out = dict(
        seeds=np.asarray(seeds),
        q=np.array(q),
        best=best,
        success=np.array(success),
        at_bound=np.array(at_bound),
        dropped=np.array(dropped),
    )
    if len(poi) == 1:
        out['signed_root'] = np.sign(best[:, 0] - theta_t[poi[0]]) * np.sqrt(np.maximum(out['q'], 0.0))
    return out


def wilks_summary(q, k, cls, signed_root=None):
    """
    Compare toy q(theta_t) with the chi2_k distribution of Wilks' theorem: per confidence level the coverage (fraction
    of toys with q below the chi2_k quantile, binomial error) and the toy quantile (the toy-calibrated level) beside
    the chi2_k quantile; the KS test of q against chi2_k; for one poi the mean (with its error) and standard
    deviation of the signed root, N(0, 1) under Wilks' theorem.

    Args:
        q: [n] q(theta_t) of the converged toys
        k: number of pois
        cls: confidence levels
        signed_root: [n] signed root of the same toys (one poi), or None
    Returns:
        dict: n, and with at least two toys coverage {cl: [fraction, error]}, levels {cl: [chi2_k quantile, toy
        quantile]}, ks_pvalue, signed_root {mean, mean_error, std} (one poi)
    """
    n = len(q)
    out = {'n': n}
    if n < 2:
        return out
    out['coverage'], out['levels'] = {}, {}
    for cl in cls:
        level = float(scipy.stats.chi2.ppf(cl, k))
        fraction = float((q < level).mean())
        out['coverage'][cl] = [fraction, float(np.sqrt(fraction * (1 - fraction) / n))]
        out['levels'][cl] = [level, float(np.quantile(q, cl))]
    out['ks_pvalue'] = float(scipy.stats.kstest(q, 'chi2', args=(k,)).pvalue)
    if signed_root is not None:
        std = float(signed_root.std(ddof=1))
        out['signed_root'] = {'mean': float(signed_root.mean()), 'mean_error': std / np.sqrt(n), 'std': std}
    return out


def plot_wilks(q, k, summary, title, outname, lumi=None):
    """
    Histogram of the toy q(theta_t) (density, log scale; points with Poisson errors, empty bins not drawn) against
    the chi2_k density of Wilks' theorem, with the chi2_k levels (grey) and the toy-calibrated levels (orange) of each
    confidence level (line style per level), the coverage of the chi2_k levels and the KS p-value.

    Args:
        q: [n] q(theta_t) of the converged toys
        k: number of pois
        summary: wilks_summary result
        title: legend title (the wilks label)
        outname: file to write
        lumi: integrated luminosity (fb^-1) of the CMS label, or None
    """
    levels = summary['levels']
    x_max = max(1.5 * max(max(v) for v in levels.values()), float(np.quantile(q, 0.99)))
    edges = np.linspace(0, x_max, 31)
    counts = np.histogram(q, edges)[0]
    scale = 1 / (len(q) * np.diff(edges))
    centers = (edges[:-1] + edges[1:]) / 2
    filled = counts > 0
    mh.style.use('CMS')
    fig, ax = plt.subplots()
    ax.errorbar(
        centers[filled],
        (counts * scale)[filled],
        xerr=(np.diff(edges) / 2)[filled],
        yerr=(np.sqrt(counts) * scale)[filled],
        linestyle='none',
        marker='o',
        markersize=6,
        color=SCAN_COLORS[0],
        label=f'toys ({len(q)})',
    )
    x = np.linspace(x_max / 1000, x_max, 400)
    ax.plot(x, scipy.stats.chi2.pdf(x, k), color='k', linewidth=2, label=rf'$\chi^2_{{{k}}}$')
    lines = []
    for (cl, (chi2_level, toy_level)), style in zip(sorted(levels.items()), LEVEL_STYLES):
        ax.axvline(chi2_level, color='grey', linestyle=style, linewidth=1.5)
        ax.axvline(toy_level, color=SCAN_COLORS[1], linestyle=style, linewidth=1.5)
        fraction, error = summary['coverage'][cl]
        lines.append(f'{100 * cl:.0f}%: coverage {fraction:.3f} $\\pm$ {error:.3f}, toy level {toy_level:.2f}')
        ax.plot([], [], color='grey', linestyle=style, linewidth=1.5, label=rf'$\chi^2$ level {100 * cl:.0f}%')
        ax.plot([], [], color=SCAN_COLORS[1], linestyle=style, linewidth=1.5, label=f'toy level {100 * cl:.0f}%')
    lines.append(f'KS p-value {summary["ks_pvalue"]:.3g}')
    ax.text(0.97, 0.52, '\n'.join(lines), transform=ax.transAxes, ha='right', va='top', fontsize=12)
    ax.set_yscale('log')
    ax.set_xlim(0, x_max)
    ax.set_xlabel(r'$q(\theta_t)$')
    ax.set_ylabel('density')
    ax.legend(loc='upper right', fontsize=13, title=title, title_fontsize=13)
    mh.cms.label('Preliminary', data=False, lumi=lumi, com=13, ax=ax)
    fig.savefig(outname, bbox_inches='tight')
    plt.close(fig)


def run_wilks(config, inputs, first_seed=None, toys=None):
    """
    One batch of Wilks toys (wilks_toys) for every wilks entry of a config, written to
    {name}/wilks/<label>_<first seed>.npz.

    Args:
        config: scan config (load_config, check_scans, check_wilks)
        inputs: load_inputs result
        first_seed: first seed of the batch (default: the entry's first_seed, default 0)
        toys: toys in the batch (default: the entry's toys)
    """
    name = config['name']
    os.makedirs(f'{name}/wilks', mode=0o755, exist_ok=True)
    for label, entry in config['wilks'].items():
        start = time.time()
        scan_args, target_spec, _ = scan_settings(config, entry['scan'], inputs)
        first = entry.get('first_seed', 0) if first_seed is None else first_seed
        n = entry['toys'] if toys is None else toys
        result = wilks_toys(
            inputs, scan_args, target_spec, np.arange(first, first + n), entry.get('randomize_nu_obs', True)
        )
        np.savez_compressed(f'{name}/wilks/{label}_{first:06d}.npz', **result)
        print(
            f'{label}: seeds {first}-{first + n - 1}, failed {int((~result["success"]).sum())}, '
            f'at bound {int(result["at_bound"].sum())}, {time.time() - start:.1f} s'
        )


def merge_wilks(config):
    """
    Merge the Wilks toy batches in {name}/wilks/ of every wilks entry (a seed may appear in one batch only), compare
    the converged toys (both fits successful) with chi2_k (wilks_summary) and write {name}/wilks.yml (per entry: scan,
    pois, target, toys, failed, at_bound, batches, seed range, randomize_nu_obs and the summary), {name}/wilks.npz
    (merged arrays, keys <label>/<array>) and {name}/plots/wilks/<label>.png (plot_wilks).

    Args:
        config: scan config (load_config, check_scans, check_wilks)
    """
    name = config['name']
    os.makedirs(f'{name}/plots/wilks', mode=0o755, exist_ok=True)
    summary, arrays = {}, {}
    for label, entry in config['wilks'].items():
        files = sorted(glob.glob(f'{name}/wilks/{label}_*.npz'))
        if not files:
            sys.exit(f'[error]: wilks {label}: no toy batches in {name}/wilks')
        batches = [dict(np.load(f)) for f in files]
        merged = {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}
        seeds, counts = np.unique(merged['seeds'], return_counts=True)
        if (counts > 1).any():
            sys.exit(f'[error]: wilks {label}: seeds {seeds[counts > 1].tolist()} appear in more than one batch')
        scan_entry = config['scans'][entry['scan']]
        pois = list(scan_entry['pois'])
        cls = tuple(scan_entry.get('cls', (0.6827, 0.95)))
        ok = merged['success']
        result = wilks_summary(
            merged['q'][ok], len(pois), cls, merged['signed_root'][ok] if 'signed_root' in merged else None
        )
        theta = (scan_entry.get('target') or {}).get('theta', {})
        summary[label] = {
            'scan': entry['scan'],
            'pois': pois,
            'target': {p: float(theta.get(p, 0.0)) for p in pois},
            'toys': int(len(merged['q'])),
            'failed': int((~ok).sum()),
            'at_bound': int(merged['at_bound'].sum()),
            'batches': len(files),
            'seeds': [int(seeds.min()), int(seeds.max())],
            'randomize_nu_obs': entry.get('randomize_nu_obs', True),
            **result,
        }
        for key, value in merged.items():
            arrays[f'{label}/{key}'] = value
        if result['n'] >= 2:
            plot_wilks(
                merged['q'][ok], len(pois), result, label, f'{name}/plots/wilks/{label}.png', config.get('plot_lumi')
            )
    with open(f'{name}/wilks.yml', 'w') as f:
        yaml.safe_dump(_plain(summary), f, sort_keys=False)
    np.savez_compressed(f'{name}/wilks.npz', **arrays)


def main(config_path, plot_only=False, wilks=False, merge=False, first_seed=None, toys=None):
    """
    Run the scans of a scan config: every entry of config['scans'] (check_scans) is scanned on its target (Asimov
    or toy, target_weights; a toy is drawn from the events with w^s(theta_t) > 0), with q at the target (scan,
    q_target) and with scan_uncertainty when the entry has uncertainty. {name}/scans.yml and scans.npz (write_results)
    are rewritten after every scan; the plots (plot_results) are drawn at the end. A scan whose grid has too few points
    below the lowest level (grid_resolution) is reported as coarse. With wilks, only one batch of the Wilks toys of
    config['wilks'] is run (run_wilks); with merge, only the batches are merged and compared with chi2_k
    (merge_wilks).

    Args:
        config_path: scan config yml
        plot_only: only redraw the plots from an earlier run's scans.yml and scans.npz
        wilks: run one batch of Wilks toys instead of the scans
        merge: merge the Wilks toy batches instead of running scans
        first_seed, toys: seeds of the Wilks batch (run_wilks)
    """
    config = load_config(config_path)
    name = config['name']
    if plot_only:
        plot_results(name, lumi=config.get('plot_lumi'))
        return
    torch.set_num_threads(config.get('n_threads', 1))
    check_scans(config)
    check_wilks(config)
    if (wilks or merge) and not config.get('wilks'):
        sys.exit('[error]: the config lists no wilks entries')
    if merge:
        merge_wilks(config)
        return
    scans = config['scans']
    if wilks:
        run_wilks(config, load_inputs(config), first_seed=first_seed, toys=toys)
        return
    inputs = load_inputs(config, members=any(entry.get('uncertainty') for entry in scans.values()))
    settings = {label: scan_settings(config, label, inputs) for label in scans}

    entries = {}
    for label, (scan_args, target_spec, spec) in settings.items():
        start = time.time()
        keep = None
        if target_spec['kind'] == 'toy':
            keep = positive_events([inputs.coefs.T], inputs.keep_q, points=target_spec['theta_t'][None])
        target = target_weights(
            inputs,
            target_spec['theta_t'],
            nu_t=target_spec['nu_t'],
            kind=target_spec['kind'],
            keep=keep,
            seed=target_spec['seed'],
        )
        u_entry = spec.get('uncertainty')
        if u_entry:
            uncertainty = scan_uncertainty(
                inputs,
                target,
                scan_args,
                member_scans=u_entry.get('members', True),
                n_boot=u_entry.get('n_boot', 0),
                subsets=u_entry.get('subsets'),
                seed=u_entry.get('seed', 0),
                theta_t=target_spec['theta_t'],
            )
            result = uncertainty.nominal
        else:
            uncertainty = None
            result = scan(inputs, target, theta_t=target_spec['theta_t'], **scan_args)
        entries[label] = dict(scan=result, theta_t=target_spec['theta_t'], spec=spec, uncertainty=uncertainty)
        write_results(name, config, inputs, entries)
        poi = [inputs.wcs.index(p) for p in result.pois]
        best = ', '.join(f'{p} {v:.6g}' for p, v in zip(result.pois, result.best.theta[poi]))
        resolution = grid_resolution(result)
        lowest = min(result.levels)
        coarse = (
            f', [warning] coarse grid: {resolution["points_inside"][lowest]} points below the {100 * lowest:.0f}% level'
            if resolution['coarse']
            else ''
        )
        print(
            f'{label}: best {best}, failed fits {int((~result.success).sum())}, dropped {result.dropped}, '
            f'stable {result.stable}, {time.time() - start:.1f} s{coarse}'
        )
    plot_results(name, lumi=config.get('plot_lumi'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='scan configuration yml file')
    parser.add_argument('--plot', action='store_true', help='only redraw the plots of an earlier run')
    parser.add_argument('--wilks', action='store_true', help='run one batch of the Wilks toys instead of the scans')
    parser.add_argument('--first-seed', type=int, help='first toy seed of the Wilks batch')
    parser.add_argument('--toys', type=int, help='toys in the Wilks batch')
    parser.add_argument('--merge', action='store_true', help='merge the Wilks toy batches and compare with chi2')
    args = parser.parse_args()
    main(args.config, args.plot, args.wilks, args.merge, args.first_seed, args.toys)
