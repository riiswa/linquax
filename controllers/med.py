from functools import partial

import jax
import jax.numpy as jnp

from controllers import TS
from controllers.model_based import ModelBasedState
from core import LinearQuadraticEnv
from utils import dlyap, dare


class MED(TS):
    @property
    def name(self) -> str:
        return "MED-LQ"

    def __init__(self, env: LinearQuadraticEnv, warmup_steps: int, improved_exploration_steps: int, delta = 1e-4, excitation: float = 2.0, n_samples: int = 32):
        super().__init__(env, delta=delta, warmup_steps=warmup_steps, improved_exploration_steps=improved_exploration_steps, excitation=excitation)
        self.n_samples = n_samples


    @partial(jax.jit, static_argnums=(0,))
    def parameter_estimation(self, rng: jax.random.PRNGKey, controller_state: ModelBasedState):
        Theta_hat = jnp.hstack((controller_state.A, controller_state.B))

        rng, rng_sample = jax.random.split(rng)
        candidates = self.rejection_sampling(rng_sample, Theta_hat, controller_state.V, controller_state.P, self.n_samples)[0]

        def closed_loop(Theta):
            A = Theta[:, :self.state_dim]
            B = Theta[:, self.state_dim:]

            P = dare(A, B, self.env.Q, self.env.R, controller_state.P)
            K = jnp.linalg.inv(self.env.B.T @ P @ self.env.B + self.env.R) @ (
                    self.env.B.T @ P @ self.env.A
            )
            return A - B @ K

        A_K_hat = closed_loop(Theta_hat)
        Omega = jnp.identity(self.state_dim) * self.env.step_cov
        Sigma = dlyap(A_K_hat, Omega)
        A_K_candidates = jax.vmap(closed_loop)(candidates)

        K = jax.vmap(lambda A_K: 0.5 * jnp.trace((A_K_hat - A_K).T @ jnp.linalg.inv(Omega) @ (A_K_hat - A_K) @ Sigma ))(A_K_candidates)
        U = jax.vmap(lambda Theta: jnp.trace((Theta_hat - Theta) @ jnp.linalg.inv(controller_state.V) @ (Theta_hat - Theta).T))(candidates)
        H = -K/(self.confidence_threshold(controller_state.V) * U)
        H = jnp.append(H, 0)
        candidates = jnp.append(candidates, Theta_hat[jnp.newaxis, :], axis=0)
        Theta_hat = candidates[jax.random.categorical(rng, H)]

        A = Theta_hat[:, :self.state_dim]
        B = Theta_hat[:, self.state_dim:]

        return controller_state.replace(A=A, B=B)

