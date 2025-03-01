from tensorboardX import SummaryWriter
from linquax.environments import make_env
from linquax.controllers import MED, OFULQ, TS
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

if __name__ == '__main__':
    # Environment setup
    env = make_env("boeing747")
    writer = SummaryWriter(logdir="runs/boeing747_comparison")

    # Simulation parameters
    T = 500
    rng = jax.random.PRNGKey(4)

    # Dictionary to store results
    results = {}

    # Colors for plotting
    colors = {
        "MED": "#ffc61e",
        "OFULQ": "#ff5733",
        "TS": "#33b5ff"
    }

    # Run MED controller
    controller_med = MED(env, warmup_steps=50, improved_exploration_steps=0, n_samples=128, writer=writer)
    rng, rng_med = jax.random.split(rng)
    controller_med_state = controller_med.init(rng_med, T)
    _, _, costs_med = jax.block_until_ready(
        env.simulate(rng_med, controller_med_state, controller_med.policy_fn, controller_med.on_completion_fn, T)
    )
    results["MED"] = costs_med

    # Run OFULQ controller
    controller_ofulq = OFULQ(env, warmup_steps=50, improved_exploration_steps=0, writer=writer)
    rng, rng_ofulq = jax.random.split(rng)
    controller_ofulq_state = controller_ofulq.init(rng_ofulq, T)
    _, _, costs_ofulq = jax.block_until_ready(
        env.simulate(rng_ofulq, controller_ofulq_state, controller_ofulq.policy_fn, controller_ofulq.on_completion_fn,
                     T)
    )
    results["OFULQ"] = costs_ofulq

    # Run TS controller
    controller_ts = TS(env, warmup_steps=50, improved_exploration_steps=0, writer=writer)
    rng, rng_ts = jax.random.split(rng)
    controller_ts_state = controller_ts.init(rng_ts, T)
    _, _, costs_ts = jax.block_until_ready(
        env.simulate(rng_ts, controller_ts_state, controller_ts.policy_fn, controller_ts.on_completion_fn, T)
    )
    results["TS"] = costs_ts

    # Plotting
    plt.figure(figsize=(5, 3.5), dpi=300)

    for name, costs in results.items():
        cum_costs = jnp.cumsum(costs)
        plt.plot(
            cum_costs,
            lw=2,
            marker='*' if name == "MED" else 'o' if name == "OFULQ" else 'x',
            markersize=7 if name == "MED" else 6,
            label=name,
            markevery=50,
            color=colors[name]
        )

    plt.title(f"{env.name} Controller Comparison")
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Cost')
    plt.legend(frameon=False)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save the figure
    plt.show()

    # Report final cumulative costs
    for name, costs in results.items():
        final_cost = float(jnp.sum(costs))
        print(f"Final cumulative cost for {name}: {final_cost:.2f}")
        writer.add_scalar(f"final_cost/{name}", final_cost, 0)

    writer.close()