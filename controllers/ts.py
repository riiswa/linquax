from functools import partial

import jax
import jax.numpy as jnp

from controllers.model_based import ModelBasedState
from core import LinearQuadraticEnv
from controllers import OFULQ
from utils import inv_sqrt


class TS(OFULQ):
    @property
    def name(self) -> str:
        return "TS-LQ" if self.improved_exploration_steps == 0 else "TSAC"

    def __init__(self, env: LinearQuadraticEnv, warmup_steps: int, improved_exploration_steps: int, delta = 1e-4, excitation: float = 2.0):
        #delta = delta/(8*500)
        super().__init__(env, delta=delta, warmup_steps=warmup_steps, improved_exploration_steps=improved_exploration_steps, excitation=excitation)

    @partial(jax.jit, static_argnums=(0, 5, 6))
    def rejection_sampling(self, rng, Theta, V, P0, n_samples: int = 1, max_tries: int = 10000):
        W = inv_sqrt(V)[jnp.newaxis, :]
        beta = self.confidence_threshold(V)
        def step_fn(carry):
            rng, samples, accepted, i = carry
            rng, rng_noise = jax.random.split(rng)
            noise = jax.random.normal(rng, (n_samples, *Theta.shape))
            samples = jnp.where(accepted[:, jnp.newaxis, jnp.newaxis], samples, Theta[jnp.newaxis, :] + beta * noise @ W)
            accepted = jax.vmap(lambda T, a: jax.lax.cond(a, lambda _: True, partial(self.is_stabilizable, P0=P0), T), in_axes=(0, 0))(samples, accepted)
            # accepted = jnp.array([
            #     jax.lax.cond(a, lambda _: True, self.is_stabilizable, T)
            #     for T, a in zip(samples, accepted)
            # ])
            return rng, samples, accepted, i + 1

        def cond_fn(carry):
            rng, samples, accepted, i = carry
            return (~jnp.all(accepted) & (i < max_tries)) | (i == 0)

        _, samples, accepted, i = jax.lax.while_loop(cond_fn, step_fn, (rng, jnp.repeat(Theta[jnp.newaxis, :], n_samples, 0), jnp.zeros(n_samples, dtype=bool), 0))
        return samples, accepted


    @partial(jax.jit, static_argnums=(0,))
    def parameter_estimation(self, rng: jax.random.PRNGKey, controller_state: ModelBasedState):
        Theta_hat = jnp.hstack((controller_state.A, controller_state.B))

        Theta_hat = self.rejection_sampling(rng, Theta_hat, controller_state.V, controller_state.P, 1)[0][0]

        A = Theta_hat[:, :self.state_dim]
        B = Theta_hat[:, self.state_dim:]

        return controller_state.replace(A=A, B=B)

