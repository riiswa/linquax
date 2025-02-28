from tensorboardX import SummaryWriter

from controllers import MED
import matplotlib.pyplot as plt
from environments import make_env

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(suppress=True )

if __name__ == '__main__':
    env = make_env("boeing747")

    writer = SummaryWriter(logdir="runs/boeing747")

    controller = MED(env, warmup_steps=50, improved_exploration_steps=0, n_samples=128, writer=writer)
    rng = jax.random.PRNGKey(4)
    rng, rng_init = jax.random.split(rng)
    T = 500
    controller_state = controller.init(rng_init, T)
    _, _, costs = jax.block_until_ready(env.simulate(rng, controller_state, controller.policy_fn, controller.on_completion_fn, T))

    plt.figure(figsize=(3.5, 2.125), dpi=300)
    plt.plot(jnp.cumsum(costs), lw=2, marker='*', markersize=7,  label="MED",markevery=50, color="#ffc61e")
    plt.title(env.name)
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Cost')
    plt.legend(frameon=False)
    plt.yscale('log')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
