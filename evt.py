import numpy as np
import matplotlib.pyplot as plt

def estimate_q_from_hill_plot_grimshaw(
    data: np.ndarray,
    spot_model,
    q_range: tuple[float, float] = (0.001, 0.1),
    step: float = 0.002,
    smooth_window: int = 5,
    alpha: float = 0.5,
    save=None
) -> float:
    """Select the tail probability with the most stable GPD estimates.
    
    Parameters:
    -----------
    data : np.ndarray
        One-dimensional array of training reconstruction errors.
    spot_model : object
        SPOT estimator implementing the _grimshaw(y) method.
    q_range : tuple
        Inclusive minimum and maximum candidate tail probabilities.
    step : float
        Grid-search increment.
    smooth_window : int
        Window used to smooth adjacent parameter differences.
    alpha : float
        Weight assigned to shape-parameter instability; must be in [0, 1].
    save : path-like, optional
        If provided, save the parameter-stability plot to this path.
    
    Returns:
    --------
    best_q : float
        Selected tail probability.
    """
    q_list = np.arange(q_range[0], q_range[1] + step, step)
    gamma_list, sigma_list = [], []

    for q in q_list:
        t = np.percentile(data, 100 * (1 - q))
        y = data[data > t] - t
        if len(y) < 5:
            n = np.count_nonzero(np.isfinite(data))
            min_required_q = 5 / n
            raise ValueError(
                            f"The lower q bound {q_range[0]:.6g} yields only "
                            f"{len(y)} exceedances; at least 5 are required. "
                            f"Use q >= {min_required_q:.6g}."
                            )
        try:
            gamma, sigma, _ = spot_model._grimshaw(y)
            gamma_list.append(gamma)
            sigma_list.append(sigma)
        except Exception:
            gamma_list.append(np.nan)
            sigma_list.append(np.nan)

    gamma_arr = np.array(gamma_list)
    sigma_arr = np.array(sigma_list)
    # Smooth the adjacent absolute parameter differences.
    def calc_smooth_diff(arr):
        diffs = np.abs(np.diff(arr))
        smooth_diffs = np.convolve(diffs, np.ones(smooth_window), 'valid') / smooth_window
        
        pad_len = len(arr) - len(smooth_diffs)
        pre_pad = pad_len // 2
        post_pad = pad_len - pre_pad
        # Replicate-pad both ends to preserve the candidate-list length.
        return np.pad(smooth_diffs, (pre_pad, post_pad), mode='edge')

    gamma_diff = calc_smooth_diff(gamma_arr)
    sigma_diff = calc_smooth_diff(sigma_arr)

    # Combine normalized shape and scale instability.
    gamma_diff_norm = (gamma_diff - np.mean(gamma_diff)) / (np.std(gamma_diff) + 1e-8)
    sigma_diff_norm = (sigma_diff - np.mean(sigma_diff)) / (np.std(sigma_diff) + 1e-8)

    instability = alpha * gamma_diff_norm + (1 - alpha) * sigma_diff_norm
    #instability = alpha * gamma_diff + (1 - alpha) * sigma_diff
    
    instability[np.isnan(instability)] = np.inf

    # Select the least-unstable tail probability.
    best_index = np.argmin(instability)
    best_q = q_list[best_index]

    fig, ax1 = plt.subplots(figsize=(6*0.7, 4*0.7))
    
    ax1.plot(q_list, gamma_arr, marker='o', color='blue', label='Estimated γ')
    ax1.set_xlabel(r"$t_q$")
    ax1.set_ylabel(r"Estimated $\gamma$", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(0, linestyle='--', color='gray')
    
    ax1.axvline(best_q, color='red', linestyle='--', linewidth=1.5, label=f'Chosen $t_q$ = {best_q:.3f}')
    ax1.text(best_q, ax1.get_ylim()[1], f' Chosen $t_q$ = {best_q:.3f}', color='red', ha='left', va='top')

    # Right axis: scale parameter.
    ax2 = ax1.twinx()
    ax2.plot(q_list, sigma_arr, marker='s', color='orange', label='Estimated σ')
    ax2.set_ylabel(r"Estimated $\sigma$", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    #plt.title("Grimshaw-based Estimation vs. Tail Ratio")
    fig.tight_layout()
    plt.grid(True)
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()
    
    return best_q



from scipy.optimize import minimize
class SPOT:
    def log_prob(self, y: np.ndarray, gamma: np.ndarray, sigma_recip: np.ndarray):
        """
        Evaluate the joint GPD log-likelihood for candidate parameters.

        :param y: array :math:`Y_t = {X_i - t | X_i > t}` with shape :math:`(N_t,)`;
        :param gamma: candidate shape parameters with shape (k,).
        :param sigma_recip: reciprocal candidate scales with shape (k,).

        :return: array with shape :math:`(k,)`.
        """
        sample_num = y.shape[0]
        temp = np.log(1 + (gamma * sigma_recip).reshape(-1, 1) *
                      y.reshape(1, -1)).sum(axis=1)
        sigma_recip[sigma_recip == 0] = 1e-6
        ret = sample_num * np.log(sigma_recip) - (1 + 1 / gamma) * temp
        return ret

    def _compute_threshold(self, t, gamma, sigma, anomaly_ratio, nt, n):
        """Compute the EVT decision threshold."""
        temp = (anomaly_ratio * n / nt) ** (-gamma)
        return t + sigma / gamma * (temp - 1)

    def _grimshaw(self, y: np.ndarray, k=10, x0=None):
        """
        :param y: array :math:`Y_t = {X_i - t | X_i > t}` with shape :math:`(N_t,)`;
        :param k: number of solver initialization points.
        :param x0: optional solver initializations with shape (k,).
        """
        def v(x: np.ndarray):
            return 1 + np.log(1 + x.reshape(-1, 1) * y.reshape(1, -1)).mean(axis=1)

        def optimize_func(x: np.ndarray):
            z = 1 + x.reshape(-1, 1) * y.reshape(1, -1)
            ux, vx = (1 / z).mean(axis=1), 1 + np.log(z).mean(axis=1)
            jac_u = -(y / np.square(z)).mean(axis=1)
            jac_v = (y / z).mean(axis=1)

            uv = ux * vx - 1
            target = np.square(uv).sum()
            jac = jac_u * vx + ux * jac_v
            return target, 2 * uv * jac

        if x0 is not None:
            assert isinstance(x0, np.ndarray) and x0.shape[0] == k
        else:
            x0 = np.zeros(k)

        low, high = -1 / y.max(), 2 * (y.mean() - y.min()) / np.square(y.min())
        mid = high * y.min() / y.mean()

        canditate_x = np.zeros(k)
        solution = minimize(
            optimize_func,
            x0=x0[:k // 2],
            method='L-BFGS-B', jac=True,
            bounds=np.array([low, 0]).reshape(1, -1).repeat(k // 2, axis=0)
        )
        canditate_x[:k // 2] = solution.x
        solution = minimize(
            optimize_func,
            x0=x0[-k // 2:],
            method='L-BFGS-B', jac=True,
            bounds=np.array([mid, high]).reshape(1, -1).repeat(k // 2, axis=0)
        )
        canditate_x[-k // 2:] = solution.x

        gamma = v(canditate_x) - 1
        gamma[gamma == 0] = 1e-6
        sigma_recip = canditate_x / gamma
        log_prob = self.log_prob(y, gamma, sigma_recip)

        target_index = np.argmax(log_prob)
        return gamma[target_index], 1 / sigma_recip[target_index], canditate_x

    def _pot(self, x: np.ndarray, anomaly_ratio: float, initial_thr_ratio=None, k=10):
        if initial_thr_ratio is None:
            initial_thr_ratio = anomaly_ratio * 3.5

        t = np.percentile(x, (1 - initial_thr_ratio) * 100, axis=0)
        t1 = np.percentile(x, (1 - anomaly_ratio) * 100, axis=0)
        y = x[x > t] - t
        gamma, sigma, x0 = self._grimshaw(y,k)
        print(f'gamma={gamma:.4g},sigma={sigma:.4g}')

        threshold = self._compute_threshold(
            t, gamma, sigma, anomaly_ratio, y.shape[0], x.shape[0])

        return threshold, t, x0, t1, gamma, sigma
