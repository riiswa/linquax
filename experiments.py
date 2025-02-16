import os
import time

import matplotlib.pyplot as plt
import psutil
from aquarel import load_theme
from jax.experimental import mesh_utils

import argparse
from functools import partial

import jax
import jax.numpy as jnp

from controllers import TS, OFULQ
from environments import make_env
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map


jax.config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={os.cpu_count()}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Adaptive LQR Experiment")

    # Define command-line arguments
    parser.add_argument('--env_id', type=str, default='ac1', help="ID of the environment")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--num_seeds', type=int, default=8, help="Number of seeds")
    parser.add_argument('--horizon', type=int, default=500, help="Horizon length for the LQR")
    parser.add_argument('--noise', type=float, default=1., help="Noise for the LQR")
    parser.add_argument('--warmup_steps', type=int, default=50, help="Warmup steps for the LQR")
    parser.add_argument('--improved_exploration_steps', type=int, default=0, help="Improved Exploration steps for the LQR")

    args = parser.parse_args()

    env = make_env(args.env_id)
    env.step_cov = args.noise

    rng = jax.random.PRNGKey(args.seed)

    controllers = [
        TS(env, warmup_steps=args.warmup_steps, improved_exploration_steps=args.improved_exploration_steps),
        OFULQ(env, warmup_steps=args.warmup_steps, improved_exploration_steps=args.improved_exploration_steps),
        #Optimal(env)
    ]

    rng, rng_init = jax.random.split(rng)
    states = jax.vmap(controllers[0].init, in_axes=(0, None))(
        jax.random.split(rng, args.num_seeds),
        args.horizon,
    )

    mesh = Mesh(jax.devices(), ('i',))

    def setup_experiment(controller):
        f = partial(env.simulate, policy=controller.policy_fn, on_completion=controller.on_completion_fn, num_steps=args.horizon)
        def experiment(rng, state):
            _, _, costs = jax.vmap(f, in_axes=(0, 0))(rng, state)
            return costs
        return jax.jit(jax.experimental.shard_map.shard_map(experiment, mesh, in_specs=(P('i'), P('i')), out_specs=(P('i')), check_rep=False))

    def compute_stats(results):
        sorted_results = jnp.sort(results, axis=0)
        q1_idx = args.num_seeds // 4
        median_idx = args.num_seeds // 2
        q3_idx = (3 * args.num_seeds) // 4

        q1 = sorted_results[q1_idx]
        median = sorted_results[median_idx]
        q3 = sorted_results[q3_idx]

        iqm = jnp.mean(sorted_results[q1_idx:(q3_idx + 1)], axis=0)

        return {
            'iqm': iqm,
            'q1': q1,
            'median': median,
            'q3': q3,
            'mean': results[:, -1].mean(),
            'std': results[:, -1].std(),
        }


    stats = {}

    for controller in controllers:
        start_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        process = psutil.Process()

        print(f"[{timestamp}] 🚀 Starting {controller.name} on environment {env.name}...")

        mem_before = process.memory_info().rss / (1024 ** 2)  # Convert to MB
        cpu_before = process.cpu_percent(interval=None)
        stats[controller.name] = jax.block_until_ready(
                compute_stats(
                    jnp.cumsum(
                        setup_experiment(controller)(
                        jax.random.split(rng, args.num_seeds),
                        states,
                        ) - controllers[0].cost_star, axis=1
                    )
            )
        )

        mem_after = process.memory_info().rss / (1024 ** 2)
        cpu_after = process.cpu_percent(interval=None)

        elapsed_time = time.time() - start_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Finished {controller.name} in {elapsed_time:.2f} seconds.")
        print(f"    🔹 Memory Used: {mem_after - mem_before:.2f} MB")
        print(f"    🔹 CPU Usage: {cpu_after:.2f}%")
        print(f"    🔹 Regret: {stats[controller.name]['mean']:.2f}±{stats[controller.name]['std']:.2f}\n")

    colors = ["#ff1f5b", "#00cd6c", "#009ade", "af58bqa"]
    markers = ["o", "s", "^", "x"]

    with load_theme("scientific"):
        plt.figure(figsize=(3.5, 2.125), dpi=300)
        for (controller_name, stat), color, marker in zip(stats.items(), colors, markers):
            plt.fill_between(jnp.arange(args.horizon), stat['q1'], stat['q3'], alpha=0.2, color=color)
        for (controller_name, stat), color, marker in zip(stats.items(), colors, markers):
            plt.plot(stat['iqm'], lw=2, marker=marker, markersize=5, label=controller_name, markevery=50, color=color)

        plt.title(env.name)
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Regret')
        plt.legend(frameon=False)
        plt.yscale('log')
        #plt.grid(False)
        plt.tight_layout()
        plt.show()
