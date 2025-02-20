from matplotlib.projections.polar import ThetaTick

from controllers import Optimal, OFULQ, TS, MED
from controllers.model_based import ModelBased
from environments import make_env

import jax
import jax.numpy as jnp

from utils import dare

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(suppress=True )

if __name__ == '__main__':
    env = make_env("boeing747")

    controller = OFULQ(env, warmup_steps=50, improved_exploration_steps=35, learning_rate=1e-3)
    print("Cost Star", controller.cost_star)
    J_star = controller.cost_star
    rng = jax.random.PRNGKey(4)
    rng, rng_init = jax.random.split(rng)
    T = 500
    controller_state = controller.init(rng_init, T)
    _, _, ofulq_costs = jax.block_until_ready(env.simulate(rng, controller_state, controller.policy_fn, controller.on_completion_fn, T))

    controller = TS(env, warmup_steps=50, improved_exploration_steps=0)
    controller_state = controller.init(rng_init, T)
    #_, _, ts_costs = jax.block_until_ready(env.simulate(rng, controller_state, controller.policy_fn, controller.on_completion_fn, T))

    controller = MED(env, warmup_steps=50, improved_exploration_steps=35, n_samples=128
                     )
    controller_state = controller.init(rng_init, T)
    _, _, med_costs = jax.block_until_ready(env.simulate(rng, controller_state, controller.policy_fn, controller.on_completion_fn, T))

    controller = ModelBased(env, warmup_steps=10, improved_exploration_steps=0)
    controller_state = controller.init(rng_init, T)
    _, _, mb_costs = jax.block_until_ready(
        env.simulate(rng, controller_state, controller.policy_fn, controller.on_completion_fn, T))

    import matplotlib.pyplot as plt
    from aquarel import load_theme

    with load_theme("scientific"):
        plt.figure(figsize=(3.5, 2.125), dpi=300)
        plt.plot(jnp.cumsum(ofulq_costs), lw=2, marker='o', markersize=5, label="OFULQ",markevery=50, color="#ff1f5b")
        #plt.plot(jnp.cumsum(ts_costs), lw=2, marker='s', markersize=5,  label="TS",markevery=50, color="#009ade")
        plt.plot(jnp.cumsum(med_costs), lw=2, marker='*', markersize=7,  label="MED",markevery=50, color="#ffc61e")
        #plt.plot(jnp.cumsum(mb_costs), lw=2, marker='*', markersize=7, label="Greedy", markevery=50, color="#000000")
        plt.title(env.name)
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Cost')
        plt.legend(frameon=False)
        plt.yscale('log')
        plt.grid(False)
        plt.tight_layout()
        plt.show()
