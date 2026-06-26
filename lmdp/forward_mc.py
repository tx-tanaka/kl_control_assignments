"""Forward Monte Carlo simulation for KL control (Chapter 7, Section 7.1.6).

Instead of backward recursion, estimate desirability Z_k(x_k) and the
optimal policy Q*(u_k|x_k) by forward sampling under the reference policy R.

Key equations:
  r(i) = exp(-C_{k:T}(path_i) / alpha)                                (Eq. 7.37)
  r(x_k) = (1/N) sum_i r(i)  -->  Z_k(x_k)  as N -> inf              (Eq. 7.38)
  r(x_k, u_k) = (1/N) sum_{i in I(u_k)} r(i)                         (Eq. 7.40)
  Q*(u_k|x_k) ~= r(x_k, u_k) / r(x_k)                               (Eq. 7.43)
"""

import numpy as np


def _build_lookup_tables(grid):
    """Precompute transition and cost tables for vectorized simulation."""
    S, A = grid.n_states, grid.n_actions
    transition = np.zeros((S, A), dtype=int)
    cost_table = np.zeros((S, A))
    terminal = np.zeros(S)
    for s in range(S):
        for a in range(A):
            transition[s, a] = grid.step(s, a)
            cost_table[s, a] = grid.cost(s, a)
        terminal[s] = grid.terminal_cost(s)
    return transition, cost_table, terminal


def generate_sample_paths(grid, x0, k, T, N, rng=None):
    """Generate N sample paths from state x0 at time k under reference R.

    Each path runs from time k to time T, choosing actions according to
    the reference policy R(u|x) = uniform at each step.

    Parameters
    ----------
    grid : GridWorld
    x0 : int
        Starting state at time k.
    k : int
        Starting time step.
    T : int
        Terminal time step.
    N : int
        Number of sample paths.
    rng : numpy.random.Generator

    Returns
    -------
    states : array of shape (N, T-k+1)
        states[:, 0] = x0, states[:, t-k] = state at time t.
    actions : array of shape (N, T-k)
        actions[:, t-k] = action taken at time t.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not hasattr(grid, '_transition_table'):
        grid._transition_table, grid._cost_table, grid._terminal_table = \
            _build_lookup_tables(grid)

    transition = grid._transition_table
    horizon = T - k
    states = np.zeros((N, horizon + 1), dtype=int)
    actions = np.zeros((N, horizon), dtype=int)
    states[:, 0] = x0

    for t in range(horizon):
        actions[:, t] = rng.integers(0, grid.n_actions, size=N)
        states[:, t + 1] = transition[states[:, t], actions[:, t]]

    return states, actions


def compute_path_rewards(grid, states, actions, alpha):
    """Compute path rewards r(i) = exp(-C_{k:T}(path_i) / alpha).  (Eq. 7.37)

    Parameters
    ----------
    grid : GridWorld
    states : array of shape (N, horizon+1)
    actions : array of shape (N, horizon)
    alpha : float

    Returns
    -------
    rewards : array of shape (N,)
    C_min : float
        Minimum path cost, subtracted before exponentiating for stability.
    """
    if not hasattr(grid, '_cost_table'):
        grid._transition_table, grid._cost_table, grid._terminal_table = \
            _build_lookup_tables(grid)

    N, horizon = actions.shape
    path_costs = np.zeros(N)

    # ##########################################################
    # TODO: Compute path rewards for each sample path
    # (Eq. 7.37). Accumulate running cost and terminal cost
    # for each path, then convert to desirability scores via
    # r(i) = exp(-(C_i - C_min) / alpha).
    #
    # Lookup tables are available on the grid object (built by
    # generate_sample_paths):
    #   grid._cost_table[state, action] -> running cost
    #   grid._terminal_table[state]     -> terminal cost
    # Use these for vectorized computation over all N paths.
    #
    # ##########################################################
    # raise NotImplementedError("TODO: compute_path_rewards")

    cost_table = grid._cost_table
    terminal_table = grid._terminal_table
    for t in range(horizon):
        path_costs += cost_table[states[:, t], actions[:, t]]
    path_costs += terminal_table[states[:, -1]]

    # Subtract min for numerical stability; return offset so caller
    # can recover the true scale: Z = mean(rewards) * exp(-C_min/alpha)
    C_min = np.min(path_costs)
    rewards = np.exp(-(path_costs - C_min) / alpha)

    return rewards, C_min


def empirical_reward_per_action(rewards, first_actions, n_actions):
    """Compute r(x_k, u_k) for each action u_k.  (Eq. 7.40)

    r(x_k, u_k) = (1/N) * sum_{i in I(u_k)} r(i)

    where I(u_k) is the set of path indices whose first action is u_k.

    Parameters
    ----------
    rewards : array of shape (N,)
    first_actions : array of shape (N,)
        The action taken at the first time step of each path.
    n_actions : int

    Returns
    -------
    r_per_action : array of shape (n_actions,)
    """
    N = len(rewards)
    r_per_action = np.zeros(n_actions)

    # ##########################################################
    # TODO: Compute the average reward for paths starting
    # with each action (Eq. 7.40). For each action u, select
    # the paths whose first action equals u and average their
    # rewards over the total number of paths N.
    #
    # ##########################################################
    # raise NotImplementedError("TODO: empirical_reward_per_action")

    for u in range(n_actions):
        mask = first_actions == u
        r_per_action[u] = np.sum(rewards[mask]) / N

    return r_per_action


def approximate_policy(r_per_action):
    """Approximate optimal policy Q*(u|x) = r(x,u) / sum r(x,u').  (Eq. 7.43)

    Parameters
    ----------
    r_per_action : array of shape (n_actions,)

    Returns
    -------
    Q_approx : array of shape (n_actions,)
        Probability distribution over actions.
    """
    # ##########################################################
    # TODO: Normalize r_per_action to obtain Q*(u|x)
    # (Eq. 7.43). Divide each entry by the sum. If the sum
    # is near zero, fall back to uniform.
    #
    # ##########################################################
    # raise NotImplementedError("TODO: approximate_policy")

    total = np.sum(r_per_action)
    if total < 1e-300:
        return np.ones(len(r_per_action)) / len(r_per_action)
    return r_per_action / total


def sample_action(Q_approx, rng=None):
    """Sample an action from the approximate policy.  (Exercise 7.5 / Algorithm 10)

    Parameters
    ----------
    Q_approx : array of shape (n_actions,)
    rng : numpy.random.Generator

    Returns
    -------
    action : int
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(len(Q_approx), p=Q_approx)


def forward_mc_control(grid, x0, T, N, alpha, rng=None):
    """Run one episode of forward Monte Carlo KL control (Algorithm 13).

    At each time step k, generate N sample paths under R, compute the
    empirical policy, sample an action, and advance.

    Parameters
    ----------
    grid : GridWorld
    x0 : int
        Initial state.
    T : int
        Horizon.
    N : int
        Number of sample paths per step.
    alpha : float
    rng : numpy.random.Generator

    Returns
    -------
    trajectory : list of int
        States visited: [x_0, x_1, ..., x_T] or shorter if goal reached.
    actions_taken : list of int
        Actions taken at each step.
    Z_estimates : list of float
        Monte Carlo estimate of Z_k(x_k) at each step.
    """
    if rng is None:
        rng = np.random.default_rng()

    trajectory = [x0]
    actions_taken = []
    Z_estimates = []

    x = x0
    for k in range(T):
        states, actions = generate_sample_paths(grid, x, k, T, N, rng)
        rewards, C_min = compute_path_rewards(grid, states, actions, alpha)

        r_total = np.mean(rewards) * np.exp(-C_min / alpha)
        Z_estimates.append(r_total)

        first_actions = actions[:, 0]
        r_per_action = empirical_reward_per_action(rewards, first_actions, grid.n_actions)
        Q_approx = approximate_policy(r_per_action)
        a = sample_action(Q_approx, rng)

        x = grid.step(x, a)
        trajectory.append(x)
        actions_taken.append(a)

        r, c = grid.state_to_rc(x)
        if (r, c) == grid.goal:
            break

    return trajectory, actions_taken, Z_estimates
