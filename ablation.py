import argparse
import csv
import os
from environments import make_env
from controllers import *
import jax
import jax.numpy as jnp
from functools import partial
from time import time


def parse_args():
    parser = argparse.ArgumentParser(description='Run ablation study on controller')
    parser.add_argument('--env_id', type=str, default="inverted_pendulum",
                        help='Environment ID (default: inverted_pendulum)')
    parser.add_argument('--seeds', type=int, default=32,
                        help='Number of random seeds to run (default: 32)')
    parser.add_argument('--samples', type=str, default="4,16,64,128,256,512",
                        help='Comma-separated list of sample sizes to test (default: 4,16,64,128,256,512)')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Warmup steps for MED controller (default: 50)')
    parser.add_argument('--exploration', type=int, default=0,
                        help='Improved exploration steps for MED controller (default: 0)')
    parser.add_argument('--timesteps', type=int, default=500,
                        help='Simulation timesteps (default: 500)')
    parser.add_argument('--output_dir', type=str, default="./results",
                        help='Directory to save results (default: ./results)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse sample sizes
    ns = [int(n) for n in args.samples.split(',')]

    # Configure JAX
    jax.config.update("jax_enable_x64", True)
    rng = jax.random.PRNGKey(0)

    # Initialize environment and controller
    env = make_env(args.env_id)
    T = args.timesteps
    controller = MED(env,
                           warmup_steps=args.warmup,
                           improved_exploration_steps=args.exploration,
                           n_samples=1)
    controller_state = controller.init(rng, T)

    @partial(jax.jit, static_argnums=(1,))
    def run(rng, n_samples):
        controller.n_samples = n_samples
        _, _, costs = env.simulate(rng,
                                   controller_state,
                                   controller.policy_fn,
                                   controller.on_completion_fn,
                                   T)
        return jnp.cumsum(costs - controller.cost_star)


    # Prepare CSV file
    csv_filename = os.path.join(args.output_dir, f"ablation_{args.env_id}.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['sample_size', 'seed', 'runtime', 'regret']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Run experiments
        for n in ns:
            print(f"Running ablation with n_samples={n}")
            for i in range(args.seeds + 1):
                rng, _ = jax.random.split(rng)
                start = time()
                regret = jax.block_until_ready(run(rng, n)[-1])
                runtime = time() - start

                # Skip the first run (warmup)
                if i != 0:
                    print(f"n={n}, seed={i}, runtime={runtime:.4f}s, regret={float(regret):.6f}")
                    # Write to CSV
                    writer.writerow({'sample_size': n, 'seed': i, 'runtime': runtime, 'regret': float(regret)})
                    # Flush to ensure data is written even if interrupted
                    csvfile.flush()

    print(f"Results saved to {csv_filename}")