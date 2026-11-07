from topsbi.model.net import Net
from topsbi.tools.data import expansion, load_zero_terms, zero_term_mask

import concurrent.futures, multiprocessing, torch, tqdm, yaml

torch.set_num_threads(1)


class NetworkRatio:
    """One trained network as a function x -> r_hat(x; theta_1) (theta_1 = its config['c1']), or log r_hat with log_ratio."""

    def __init__(self, config, n_features, device='cpu'):
        """
        Load a trained network from its training config, on the device it is evaluated on (the training config's
        device is ignored).

        Args:
            config: path to yaml file used to configure the network training
            n_features: number of features used in training
            device: torch device to evaluate the network on
        """
        with open(config) as f:
            self.config = yaml.safe_load(f)
        if 'network' in self.config.keys():
            network = self.config['network']
        else:
            network = None
        with open(f'{self.config["name"]}/complete/performance.yml') as f:
            self.performance = yaml.safe_load(f)
        self.device = device
        self.model = Net(n_features, device, network)
        self.model.load_state_dict(torch.load(f'{self.config["name"]}/model.pt', map_location=torch.device(device)))
        self.model.eval()  # dropout off at inference

    def __call__(self, features: torch.tensor, network=None):
        """
        Learned ratio of a series of events.

        Args:
            features: non-normalized features x to be evaluated by the network
        Returns:
            r_hat: r_hat(x; theta_1) per event, on the CPU
        """
        with torch.no_grad():
            r_hat = self.model.ratio(features.to(self.device))
        return r_hat.cpu()

    def log_ratio(self, features: torch.tensor):
        """
        Learned log ratio of a series of events.

        Args:
            features: non-normalized features x to be evaluated by the network
        Returns:
            f_hat: log r_hat(x; theta_1) per event (log r_hat(x; theta, nu) for a nuisance network), on the CPU
        """
        with torch.no_grad():
            f_hat = self.model.log_ratio(features.to(self.device))
        return f_hat.cpu()


class RatioFit:
    """
    One seed's ensemble: the networks trained at points theta_1^(n) (and the SM network, or an exact row of ones when
    theta_0 is the SM) give the matrix of ratios R, one row r_hat(x; theta_1^(n)) per network, and Theta, one row
    T(theta_1^(n), mu^(n))[mask] per network (expansion; T(theta) without background processes). The per-event
    coefficients Gamma solve Theta Gamma = R in least squares, so r_hat(x; theta, mu) = T(theta, mu)[mask] Gamma.
    """

    def __init__(
        self,
        config: dict,
        features: torch.tensor,
    ):
        """
        Fit Gamma for one seed.

        Args:
            config: dictionary with 'sm' (SM network config, or None when the networks are trained against
                theta_0 = SM: then r_hat(x; theta_SM) = 1 exactly), 'networks' (network configs) and 'features'
            features: non-normalized features x to be evaluated by the networks
        Attributes:
            Theta: [n_networks, K] T(theta_1^(n), mu^(n)) of every network, SM last, kept columns only
            R: [n_networks, N] r_hat of every network for every event, SM last (ones when theta_0 is the SM)
            wcs, background_processes: WC names and background processes shared by every network
            n_wcs: number of WCs n; the background columns follow the (n + 1)(n + 2) / 2 columns of T(theta)
            mask: kept columns of T(theta, mu): not zero for every network, not a listed zero term, and for a
                background column, moved away from mu_p = 1 by at least one network (otherwise the SM column holds
                the background)
            finite: [N] events where every network's r_hat is finite
            Gamma: [K, N_finite] least-squares coefficients; residuals, rank, singular_values from lstsq
        """
        self.config = config
        self.Theta = []
        # filled row by row: no long-lived per-network tensors between the evaluation temporaries, so the heap does
        # not fragment (fitprobe, ensemble_validation diary)
        self.R = torch.empty((len(self.config['networks']) + 1, features.shape[0]))
        sm_reference = self.config.get('sm') is None
        if sm_reference:
            # theta_0 = SM: no SM network, its row of R is exactly one; WCs and zero terms from the first network
            with open(self.config['networks'][0]) as f:
                reference = yaml.safe_load(f)
            self.R[-1] = 1.0
        else:
            sm_network = NetworkRatio(self.config['sm'], len(self.config['features']), self.config['device'])
            reference = sm_network.config
            self.R[-1] = sm_network(features)
        self.wcs = reference['wcs']
        self.n_wcs = len(self.wcs)
        self.background_processes = list(reference.get('background_processes', []))
        zero_terms = load_zero_terms(reference)
        # (theta_SM, mu = 1): the SM with every background at its nominal yield
        sm = [1.0] + [0.0] * self.n_wcs + [1.0] * len(self.background_processes)
        if not sm_reference and [float(c) for c in reference['c1']] != sm:
            raise ValueError(f"{self.config['sm']} is not trained at theta_1 = {sm}")

        def same(terms):
            return {tuple(sorted(p)) for p in terms} == {tuple(sorted(p)) for p in zero_terms}

        for i, network_yaml in tqdm.tqdm(enumerate(self.config["networks"]), total=len(self.config['networks'])):
            network = NetworkRatio(network_yaml, len(self.config['features']), self.config['device'])
            if (
                network.config['wcs'] != self.wcs
                or not same(load_zero_terms(network.config))
                or list(network.config.get('background_processes', [])) != self.background_processes
            ):
                raise ValueError(
                    f'{network_yaml} uses different wcs, zero_terms or background_processes than the reference network'
                )
            if sm_reference and [float(c) for c in network.config['c0']] != sm:
                raise ValueError(f'{network_yaml} is not trained against theta_0 = SM, but the seed has no SM network')
            self.Theta += [expansion(network.config['c1'], self.n_wcs)]
            self.R[i] = network(features)
        self.Theta += [expansion(sm, self.n_wcs)]
        self.Theta = torch.vstack(self.Theta)
        self.mask = ~(self.Theta == 0).all(dim=0) & zero_term_mask(self.wcs, zero_terms, self.background_processes)
        # a background column at mu_p = 1 in every row equals the SM column: drop it, the SM column then holds b
        n_theta = self.Theta.shape[1] - len(self.background_processes)
        self.mask[n_theta:] &= (self.Theta[:, n_theta:] != 1).any(dim=0)
        self.Theta = self.Theta[:, self.mask]
        self.finite = ~torch.isinf(self.R).any(0)
        if sm_reference:
            # theta_0 = SM: r_hat(x; theta_SM) = T(theta_SM)[mask] Gamma = Gamma[0] (the SM term is the first kept
            # column) is exactly 1, so Gamma[0] is fixed to 1 and the other rows solve Theta[:, 1:] Gamma[1:] =
            # R - Theta[:, :1] in least squares; the SM row drops out (1 - 1 = 0 against a row of zeros)
            rest, self.residuals, rank, self.singular_values = torch.linalg.lstsq(
                self.Theta[:, 1:], self.R[:, self.finite] - self.Theta[:, :1], driver='gelsd'
            )
            # rank of the full Theta: the fixed SM column is independent of the others in the SM row
            self.rank = rank + 1
            self.Gamma = torch.vstack(
                [torch.ones((1, rest.shape[1]), dtype=rest.dtype), rest[: self.Theta.shape[1] - 1]]
            )
        else:
            Gamma, self.residuals, self.rank, self.singular_values = torch.linalg.lstsq(
                self.Theta, self.R[:, self.finite], driver='gelsd'
            )
            # lstsq returns the solution in an [n_networks, N_finite] buffer; keep only Gamma's rows
            self.Gamma = Gamma[: self.Theta.shape[1]].clone()

    def __call__(self, theta):
        """
        Learned ratio at a point: r_hat(x; theta, mu) = T(theta, mu)[mask] Gamma, for the finite events.

        Args:
            theta: SM-inclusive WC point, followed by the background strengths mu_p when the networks have
                background_processes
        Returns:
            r_hat(x; theta, mu) per finite event
        Raises:
            ValueError: a strength whose column no network moved is not 1 (the fit holds that background only at
                its nominal yield)
        """
        point = expansion(theta, self.n_wcs)
        n_theta = len(point) - len(self.background_processes)
        if (point[n_theta:][~self.mask[n_theta:]] != 1).any():
            raise ValueError(f'{theta}: a background column no network moved is fixed at mu_p = 1')
        return point[self.mask] @ self.Gamma


def seed_map(function, seeds, args, workers=1):
    """
    function(seed, *args) for every seed, in this process (workers 1) or in up to `workers` single-threaded spawn
    processes (fresh interpreters: fork is unsafe with torch / OpenMP threads; each imports this module, which sets
    one thread). Results come back in seed order, so whatever the caller sums adds in the same order either way.

    Returns:
        {seed: function(seed, *args)}
    """
    if workers <= 1:
        return {seed: function(seed, *args) for seed in seeds}
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(min(workers, len(seeds)), mp_context=context) as pool:
        futures = {seed: pool.submit(function, seed, *args) for seed in seeds}
        return {seed: future.result() for seed, future in futures.items()}


def fit_seed(seed, fit_config, seeds, features, per_seed=None):
    """
    One seed's RatioFit for Ensemble (through seed_map): the per-seed callback runs here (SeedCollector writes the
    seed's files), and only what Ensemble averages is returned, so the fit and its R are freed with the call.

    Args:
        seed: the seed to fit
        fit_config: settings shared by every seed (device, c0, features)
        seeds: config['seeds'], the sm and dedicated network configs of every seed
    Returns:
        mask, finite, Gamma and rank of the seed's fit, its number of WCs and its background processes
    """
    fit = RatioFit({**fit_config, 'sm': seeds[seed]['sm'], 'networks': seeds[seed]['networks']}, features)
    if per_seed is not None:
        per_seed(seed, fit)
    return fit.mask, fit.finite, fit.Gamma, int(fit.rank), fit.n_wcs, fit.background_processes


class Ensemble:
    """Gamma averaged over seeds: one RatioFit per seed, mean Gamma on the events every seed keeps."""

    def __init__(
        self,
        config: dict,
        features: torch.tensor,
        per_seed=None,
        workers=1,
    ):
        """
        Average the per-event coefficients Gamma of several seeds.

        Args:
            config: dictionary with the shared settings and, per seed, the sm and dedicated network configs
            features: non-normalized features x to be evaluated by the networks
            per_seed: optional callable per_seed(seed, fit) given each seed's RatioFit before it is discarded (with
                workers > 1 it runs in the worker, so it must be picklable and keep no state it needs later)
            workers: seeds fitted at once, one single-threaded process each (1: one after another in this process)
        Attributes:
            mask: kept columns of T(theta, mu), the same for every seed
            n_wcs, background_processes: number of WCs and background processes, the same for every seed
            finite: [N] events kept by every seed
            Gamma, Gamma2: [K, N_finite] seed mean of Gamma and of Gamma^2
            rank: {seed: rank of the seed's Theta}
        """
        fit_config = {
            'device': config['device'],
            'c0': config['c0'],
            'features': config['features'],
        }
        n_events = int(features.shape[0])
        self.mask = None
        self.n_wcs = None
        self.background_processes = None
        self.Gamma = None
        self.Gamma2 = None
        self.rank = {}
        # events kept by every seed; an event dropped by any seed has no complete average
        self.finite = torch.ones(n_events, dtype=torch.bool)
        seeds = list(config['seeds'])
        fits = seed_map(fit_seed, seeds, (fit_config, config['seeds'], features, per_seed), workers)
        for seed in seeds:
            mask, finite, Gamma, rank, n_wcs, background_processes = fits.pop(seed)
            if self.mask is None:
                self.mask = mask
                self.n_wcs = n_wcs
                self.background_processes = background_processes
                self.Gamma = torch.zeros((int(self.mask.sum()), n_events), dtype=Gamma.dtype)
                self.Gamma2 = torch.zeros((int(self.mask.sum()), n_events), dtype=Gamma.dtype)
            elif n_wcs != self.n_wcs or background_processes != self.background_processes:
                raise ValueError(f'seed {seed} uses different wcs or background_processes than the other seeds')
            elif not torch.equal(self.mask, mask):
                raise ValueError(f'seed {seed} was trained at a different set of WC points than the other seeds')
            # place each seed's coefficients at their own event indices
            self.Gamma[:, finite] += Gamma
            self.Gamma2[:, finite] += Gamma**2
            self.finite &= finite
            self.rank[seed] = rank
        self.Gamma = self.Gamma[:, self.finite] / len(config['seeds'])
        self.Gamma2 = self.Gamma2[:, self.finite] / len(config['seeds'])

    def __call__(self, theta: list):
        """
        r_hat(x; theta, mu) = T(theta, mu)[mask] Gamma for the events kept by every seed.

        Args:
            theta: SM-inclusive WC point, followed by the background strengths mu_p when the networks have
                background_processes
        Returns:
            r_hat(x; theta, mu) per kept event
        Raises:
            ValueError: a strength whose column no network moved is not 1 (the fit holds that background only at
                its nominal yield)
        """
        point = expansion(theta, self.n_wcs)
        n_theta = len(point) - len(self.background_processes)
        if (point[n_theta:][~self.mask[n_theta:]] != 1).any():
            raise ValueError(f'{theta}: a background column no network moved is fixed at mu_p = 1')
        return point[self.mask] @ self.Gamma
