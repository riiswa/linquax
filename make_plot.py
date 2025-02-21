import os.path
from os import listdir
from os.path import isdir

import hydra
import numpy as np
from aquarel import load_theme
from matplotlib import pyplot as plt

from run_experiment import ExperimentConfig

import jax
jax.config.update("jax_enable_x64", True)

order = ["MED-LQ", "OFULQ", "StabL", "TS-LQR", "TSAC"]

def sort_controller(lst):
    order_dict = {key: index for index, key in enumerate(order)}
    return sorted(lst, key=lambda x: order_dict.get(x, len(order)))


def compute_stats(results):
    n = results.shape[0]
    sorted_results = np.sort(results, axis=0)
    q1_idx = n // 4
    median_idx = n // 2
    q3_idx = (3 * n) // 4

    q1 = sorted_results[q1_idx]
    median = sorted_results[median_idx]
    q3 = sorted_results[q3_idx]

    iqm = np.mean(sorted_results[q1_idx:(q3_idx + 1)], axis=0)

    return {
        'iqm': iqm,
        'q1': q1,
        'median': median,
        'q3': q3,
        'mean': results[:, -1].mean(),
        'std': results[:, -1].std(),
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def plot(cfg : ExperimentConfig) -> None:
    envs = [d for d in listdir(cfg.exp_name) if isdir(os.path.join(cfg.exp_name, d))]
    controllers = set([d for env in envs for d in listdir(os.path.join(cfg.exp_name, env)) if isdir(os.path.join(cfg.exp_name, env, d))])
    controllers = sort_controller(controllers)

    colors = ["#ff1f5b", "#00cd6c", "#009ade", "#af58ba"]
    markers = ["o", "s", "^", "x"]

    n = (len(envs)+ 3) // 4  # Compute rows needed for n*4 grid
    scale = 0.75

    with load_theme("scientific"):
        plt.rcParams["font.family"] = "Times New Roman"
        fig, axes = plt.subplots(n, 4, figsize=(13 *scale, (2.25 * n)*scale), dpi=300, sharex=True)
        axes = axes.flatten()  # Flatten for easy iteration

        for idx, env in enumerate(envs):
            ax = axes[idx]
            for i, controller in enumerate(controllers):
                d = os.path.join(cfg.exp_name, env, controller)
                data = [np.load(os.path.join(d, f), allow_pickle=True).item() for f in listdir(d) if f.endswith(".npy")]
                stats = compute_stats(np.cumsum(np.vstack([np.float64(d['regret']) for d in data]), axis=1))

                ax.fill_between(np.arange(cfg.horizon), stats['q1'], stats['q3'], alpha=0.2, color=colors[i], zorder=2)
                ax.plot(stats['iqm'], lw=2, marker=markers[i], markersize=5, label=controller, markevery=50,
                        color=colors[i], zorder=2)

            ax.set_title(env, fontsize=16)
            if idx % 4 == 0:
                ax.set_ylabel('Cumulative Regret', fontsize=12)
            if idx >= (n - 1) * 4:
                ax.set_xlabel('Time Steps', fontsize=12)

            ax.set_yscale('log')

        # Set shared labels
        #fig.text(0.5, 0.0, 'Time Steps', ha='center', fontsize=12)
        # for row in range(n):
        #     fig.text(0.0, 0.5 - ((row-1) / n), 'Cumulative Regret', va='center', rotation='vertical', fontsize=10)
        #fig.text(0.0, 0.5, 'Cumulative Regret', va='center', rotation='vertical', fontsize=12)

        # Remove empty subplots if envs are not a multiple of 4
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.25)
        for ax in axes:
            t = ax.yaxis.get_offset_text()
            t.set_visible(False)
            ax.text(0.05, 0.9, t.get_text(), fontsize=12, transform=ax.transAxes)


        plt.savefig(os.path.join(cfg.exp_name, 'envs.pdf'), dpi=600)
        plt.close()

    # Create a separate legend figure
    with load_theme("scientific"):
        plt.rcParams["font.family"] = "Times New Roman"
        fig_legend = plt.figure()
        fig_legend.legend(*axes[0].get_legend_handles_labels(), loc='center', frameon=False, ncol=len(controllers))
        fig_legend.savefig(os.path.join(cfg.exp_name, 'legend.pdf'), bbox_inches='tight')

if __name__ == "__main__":
    plot()