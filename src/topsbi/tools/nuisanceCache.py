from topsbi.tools.build_ratio import NetworkRatio
from topsbi.tools.data import NUISANCE_COLUMNS, load_combined, syst_ratio_column

import datetime, hashlib, os, shutil, torch, tqdm, yaml
import numpy as np

DIRECTIONS = ('down', 'up')  # direction index of f_hat: 0 = Down, 1 = Up


def nuisance_networks(seeds):
    """
    Pair every member's nuisance training configs by term, systematic and direction. The term of a network is the
    process its training data belong to, read from the sample.yml of its nominal data: 'signal' for an EFT sample
    (eft: true, the default), otherwise the sample's process. A background nuisance is named
    <systematic>_<process>, so that it is distinct from the signal nuisance of the same systematic. A network whose
    systematic is a list of per-era names (one network for several per-era nuisances, each acting on its own era's
    events only) gives one nuisance per listed name, every one of them pointing to the same configs; its names are
    kept in 'network'.

    Args:
        seeds: config['seeds'], {e: {'networks': [training yml paths]}}
    Returns:
        nuisances: [{'name': nuisance name, 'systematic': systematic in the data, 'method': method, 'term': 'signal'
            or the background process, 'data': training data directory of its networks}, with 'network': [per-era
            systematics of the shared network] for a nuisance of a list systematic] sorted by name; entry n is row n
            of f_hat
        networks: {e: [[down yml, up yml] for each nuisance n]}, direction index as in DIRECTIONS
    """
    samples = {}

    def term_of(path, data):
        sample = f'{data}/nominal/sample.yml'
        if sample not in samples:
            if not os.path.exists(sample):
                raise ValueError(f'{sample} not found: the term of {path} is unknown')
            with open(sample) as f:
                specs = yaml.safe_load(f)
            if specs.get('eft', True):
                samples[sample] = 'signal'
            elif 'process' in specs:
                samples[sample] = specs['process']
            else:
                raise ValueError(f'{sample} has eft: false and no process')
        return samples[sample]

    methods, datas, pairs = {}, {}, {}
    for seed, members in seeds.items():
        pairs[seed] = {}
        for path in members['networks']:
            with open(path) as f:
                config = yaml.safe_load(f)
            systematic, direction = config['systematic'], config['direction'].lower()
            if direction not in DIRECTIONS:
                raise ValueError(f'{path}: unknown direction {config["direction"]!r}; expected one of {DIRECTIONS}')
            # a network is keyed by its term and its systematic, the tuple of its names for a list of per-era names
            key = (term_of(path, config['data']), systematic if isinstance(systematic, str) else tuple(systematic))
            if methods.setdefault(key, config['method']) != config['method']:
                raise ValueError(
                    f'{key[0]} {systematic} is trained with both {methods[key]!r} and {config["method"]!r}'
                )
            if datas.setdefault(key, config['data']) != config['data']:
                raise ValueError(f'{key[0]} {systematic} is trained on both {datas[key]} and {config["data"]}')
            pair = pairs[seed].setdefault(key, [None, None])
            if pair[DIRECTIONS.index(direction)] is not None:
                raise ValueError(f'member {seed}: {key[0]} {systematic} {direction} is listed twice')
            pair[DIRECTIONS.index(direction)] = path
    # one row per nuisance name: the network it is trained in and the systematic it is stored as in the data
    rows = {}
    for key in methods:
        term, systematic = key
        for name in [systematic] if isinstance(systematic, str) else systematic:
            row = name if term == 'signal' else f'{name}_{term}'
            if row in rows:
                raise ValueError(f'{row} is trained in two networks, {rows[row][0]} and {key}')
            rows[row] = (key, name)
    names = sorted(rows)
    networks = {}
    for seed, pair in pairs.items():
        if set(pair) != set(methods):
            missing = sorted(set(methods) - set(pair), key=str)
            raise ValueError(f'member {seed} lacks nuisances {missing}')
        for key in methods:
            if None in pair[key]:
                raise ValueError(f'member {seed}: {key} has no {DIRECTIONS[pair[key].index(None)]} network')
        networks[seed] = [pair[rows[name][0]] for name in names]
    nuisances = []
    for name in names:
        key, stored = rows[name]
        term, systematic = key
        nuisance = {'name': name, 'systematic': stored, 'method': methods[key], 'term': term, 'data': datas[key]}
        if not isinstance(systematic, str):
            nuisance['network'] = list(systematic)
        nuisances.append(nuisance)
    return nuisances, networks


def reference_point(nuisances, networks):
    """
    The reference point theta_ref of the nuisance networks: reference_point of the sample.yml in each network's data
    directories (nominal, and <systematic><Up|Down> for kinematic_shift; a weight_shift reads its variation from
    nominal). Every network must see the same point.

    Args:
        nuisances, networks: as returned by nuisance_networks
    Returns:
        theta_ref: reference_point (SM-inclusive, theta_ref[0] = 1)
    """
    points = {}
    for pairs in networks.values():
        for nuisance, pair in zip(nuisances, pairs):
            for direction, path in zip(DIRECTIONS, pair):
                with open(path) as f:
                    data = yaml.safe_load(f)['data']
                names = ['nominal']
                if nuisance['method'] == 'kinematic_shift':
                    names.append(f'{nuisance["systematic"]}{direction.capitalize()}')
                for name in names:
                    sample = f'{data}/{name}/sample.yml'
                    if sample not in points:
                        if not os.path.exists(sample):
                            raise ValueError(f'{sample} not found: theta_ref of {path} is unknown')
                        with open(sample) as f:
                            points[sample] = yaml.safe_load(f).get('reference_point')
                        if points[sample] is None:
                            raise ValueError(f'{sample} has no reference_point')
    distinct = {tuple(point) for point in points.values()}
    if len(distinct) > 1:
        listing = ', '.join(f'{sample}: {point}' for sample, point in points.items())
        raise ValueError(f'the nuisance networks have different reference points: {listing}')
    return list(distinct.pop())


def nuisance_eras(nuisances, networks, year_loc=21):
    """
    The eras each nuisance acts in. A kinematic_shift variation may be produced only in some eras (JER per year);
    load_nuisance_data then trains on the nominal events of those eras only, and r(x, z; theta, nu) = 1 in the others.
    Read from the era index (year_int feature) of each variation's validation.p, which must be the same for Up and
    Down. A weight_shift nuisance of a shared network (one per-era name of a list systematic) acts in the eras where
    its syst_ratio column differs from 1 in either direction, read from the nominal validation.p; every other
    weight_shift nuisance acts in every era.

    Args:
        nuisances, networks: as returned by nuisance_networks
        year_loc: feature column of the era index
    Returns:
        [eras per nuisance n]: a sorted list of era indices, or None for every era (a weight_shift nuisance of its own
        network, or a variation present in every era of nominal)
    """

    def eras_of(path, layout):
        features = load_combined(path, layout, ('features',))[0]
        return sorted(torch.unique(features[:, year_loc].to(torch.int64)).tolist())

    def weight_eras(root, name):
        if root not in weights:
            columns = dict(
                zip(
                    NUISANCE_COLUMNS['nominal'],
                    load_combined(f'{root}/nominal/validation.p', 'nominal', ('features', 'syst_ratio')),
                )
            )
            weights[root] = (columns['features'][:, year_loc].to(torch.int64), columns['syst_ratio'])
        event_eras, syst_ratio = weights[root]
        shifted = torch.zeros(len(event_eras), dtype=torch.bool)
        for direction in DIRECTIONS:
            k, d = syst_ratio_column(root, name, direction)
            shifted |= syst_ratio[:, k, d] != 1
        return sorted(torch.unique(event_eras[shifted]).tolist())

    nominal, variations, weights = {}, {}, {}
    out = []
    for n, nuisance in enumerate(nuisances):
        if nuisance['method'] != 'kinematic_shift' and 'network' not in nuisance:
            out.append(None)
            continue
        eras = set()
        for pairs in networks.values():
            for direction, path in zip(DIRECTIONS, pairs[n]):
                with open(path) as f:
                    root = yaml.safe_load(f)['data']
                if nuisance['method'] != 'kinematic_shift':
                    eras.add(tuple(weight_eras(root, nuisance['systematic'])))
                    continue
                if root not in nominal:
                    nominal[root] = eras_of(f'{root}/nominal/validation.p', 'nominal')
                variation = f'{root}/{nuisance["systematic"]}{direction.capitalize()}/validation.p'
                if variation not in variations:
                    variations[variation] = eras_of(variation, 'variation')
                eras.add(None if variations[variation] == nominal[root] else tuple(variations[variation]))
        if len(eras) > 1:
            raise ValueError(f'{nuisance["name"]}: its networks were trained on different eras {sorted(eras, key=str)}')
        eras = eras.pop()
        out.append(None if eras is None else list(eras))
    return out


def nuisance_hashes(networks):
    """sha256 of the model.pt of every nuisance network: {e: [[down hash, up hash] for each nuisance n]}."""
    hashes = {}
    for seed, pairs in networks.items():
        hashes[seed] = []
        for pair in pairs:
            hashes[seed].append([])
            for network in pair:
                with open(network) as f:
                    name = yaml.safe_load(f)['name']
                with open(f'{name}/model.pt', 'rb') as f:
                    hashes[seed][-1].append(hashlib.sha256(f.read()).hexdigest())
    return hashes


def nuisance_seed(seed, networks, features, device, tmp):
    """
    One ensemble member's f_hat (through seed_map): every nuisance network evaluated on every event, written to
    {tmp}/seed_{seed} (f_hat.npy, filter.npy). A network shared by several nuisances (the per-era names of one
    network) is evaluated once and copied to each of their rows.

    Args:
        seed: the member e to evaluate
        networks: as returned by nuisance_networks, {e: [[down yml, up yml] for each nuisance n]}
        features: [N, n_features] non-normalized network inputs
        device: torch device to evaluate the networks on
        tmp: temporary cache directory (write_nuisance_cache moves it into place)
    Returns:
        finite: [N] events where every network of the member gives a finite f_hat
    """
    pairs = networks[seed]
    # filled row by row, preallocated so the heap does not fragment (as RatioFit.R)
    f_hat = torch.empty((len(pairs), len(DIRECTIONS), features.shape[0]))
    # the first row of each distinct pair of configs; later rows of the same network copy it
    first = {}
    for n, pair in tqdm.tqdm(enumerate(pairs), total=len(pairs)):
        key = tuple(pair)
        if key in first:
            f_hat[n] = f_hat[first[key]]
            continue
        first[key] = n
        for d, path in enumerate(pair):
            f_hat[n, d] = NetworkRatio(path, features.shape[1], device).log_ratio(features)
    finite = torch.isfinite(f_hat).all(dim=(0, 1))
    d = f'{tmp}/seed_{seed}'
    os.makedirs(d)
    # non-finite values kept as they are; filter marks the usable events
    np.save(f'{d}/f_hat.npy', f_hat.numpy())
    np.save(f'{d}/filter.npy', finite.numpy())
    return finite


def write_nuisance_cache(path, config, nuisances, networks, theta_ref, finites, event_eras):
    """
    Complete a nuisance cache and move it to path. nuisance_seed has written every member to path.tmp (f_hat.npy,
    filter.npy); this adds the member mean <f_hat>_e on the events every member keeps, 0 outside the nuisance's eras
    (f_hat_cache.npy, filter_cache.npy), then nuisance.yml last, so an interrupted write never looks complete. Every
    file and key is listed in docs/ensemble.md.

    Args:
        path: cache directory, replaced if it exists
        config: nuisance validation config (ensemble, data, seeds)
        nuisances, networks: as returned by nuisance_networks, each nuisance with its eras (nuisance_eras)
        theta_ref: as returned by reference_point
        finites: {e: [N] events member e keeps}, as returned by nuisance_seed
        event_eras: [N] era index of each event of config['data']
    """
    tmp = f'{path}.tmp'
    if not os.path.isdir(tmp):
        raise FileNotFoundError(f'{tmp} not found: evaluate the members with nuisance_seed first')
    seeds = list(networks)
    finite = torch.stack([finites[seed] for seed in seeds]).all(dim=0).numpy()
    f_hats = [np.load(f'{tmp}/seed_{seed}/f_hat.npy', mmap_mode='r') for seed in seeds]
    f_hat = np.empty((len(nuisances), len(DIRECTIONS), int(finite.sum())), dtype=np.float32)
    # one nuisance at a time, summed in float64 in member order, so memory stays at one [2, N] row per member
    for n in range(len(nuisances)):
        total = np.zeros((len(DIRECTIONS), int(finite.sum())))
        for member in f_hats:
            total += member[n][:, finite]
        f_hat[n] = total / len(seeds)
        # a network trained on some eras only describes those; r(x, z; theta, nu) = 1 in the others
        if nuisances[n].get('eras') is not None:
            f_hat[n][:, ~np.isin(event_eras[finite], nuisances[n]['eras'])] = 0
    np.save(f'{tmp}/f_hat_cache.npy', f_hat)
    np.save(f'{tmp}/filter_cache.npy', finite)
    manifest = {
        'ensemble': config['ensemble'],
        'data': config['data'],
        'theta_ref': theta_ref,
        'nuisances': nuisances,
        'seeds': config['seeds'],
        'model_hash': nuisance_hashes(networks),
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(f'{tmp}/nuisance.yml', 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def interpolate(f_hat, nu):
    """
    log r_hat(x; nu) = sum_n f_n(nu_n), Combine's interpolation of a template shift (shapeN, on log r_hat) with
    delta^+ = f_hat^Up, delta^- = f_hat^Down (log r_hat = 0 at nu = 0):
    f(nu) = 1/2 [(delta^+ - delta^-) nu + 1/8 (delta^+ + delta^-)(3 nu^6 - 10 nu^4 + 15 nu^2)] for |nu| <= 1,
    delta^+ nu for nu > 1 and delta^- |nu| for nu < -1; f and its first two derivatives are continuous
    (https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/settinguptheanalysis/).

    Args:
        f_hat: [n_n, 2, N] f_hat^Down, f_hat^Up per nuisance and event
        nu: [n_n] nuisance values, in the order of f_hat
    Returns:
        [N] log r_hat(x; nu), float64
    """
    f_hat = np.asarray(f_hat, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)[:, None]
    down, up = f_hat[:, 0], f_hat[:, 1]
    inside = ((up - down) * nu + (up + down) * (3 * nu**6 - 10 * nu**4 + 15 * nu**2) / 8) / 2
    outside = np.where(nu > 0, up, down) * np.abs(nu)
    return np.where(np.abs(nu) <= 1, inside, outside).sum(axis=0)


class NuisanceCache:
    def __init__(self, path, mmap=True):
        """
        Read a nuisance cache written by write_nuisance_cache. Large per-event arrays are memory-mapped unless mmap
        is False.

        Attributes:
            f_hat: [n_n, 2, N_finite] member-mean f_hat (direction 0 = Down, 1 = Up), 0 outside the nuisance's
                eras; finite: [N] events kept by every member; nuisances: [{'name', 'method', 'eras'}], row n of f_hat;
                theta_ref: the networks' reference point
            seed_f_hat, seed_finite: per member f_hat [n_n, 2, N] and kept events
        """
        if not os.path.exists(f'{path}/nuisance.yml'):
            raise FileNotFoundError(f'{path} is not a complete nuisance cache')
        with open(f'{path}/nuisance.yml') as f:
            self.manifest = yaml.safe_load(f)
        mode = 'r' if mmap else None
        self.path = path
        self.f_hat = np.load(f'{path}/f_hat_cache.npy', mmap_mode=mode)
        self.finite = np.load(f'{path}/filter_cache.npy')
        self.nuisances = self.manifest['nuisances']
        self.theta_ref = self.manifest['theta_ref']
        self.seeds = list(self.manifest['seeds'])
        self.seed_f_hat = {s: np.load(f'{path}/seed_{s}/f_hat.npy', mmap_mode=mode) for s in self.seeds}
        self.seed_finite = {s: np.load(f'{path}/seed_{s}/filter.npy') for s in self.seeds}

    def matches(self, config):
        """Whether the cache was built from the same EFT ensemble, data, networks and network weights as config."""
        if any(self.manifest[k] != config[k] for k in ('ensemble', 'data', 'seeds')):
            return False
        return self.manifest['model_hash'] == nuisance_hashes(nuisance_networks(config['seeds'])[1])

    def __call__(self, nu):
        """log r_hat(x; nu) for the events kept by every member; nu in the order of self.nuisances."""
        return interpolate(self.f_hat, nu)
