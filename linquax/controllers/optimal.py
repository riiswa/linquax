from functools import partial

import jax
from jax import numpy as jnp

from linquax.core import Controller, ControllerState, LinearQuadraticEnv
from linquax.utils import dare

class Optimal(Controller):
    @property
    def name(self) -> str:
        return "Optimal"

    def __init__(self, env: LinearQuadraticEnv):
        super().__init__(env)
        P = dare(self.env.A, self.env.B, self.env.Q, self.env.R)
        self.K = jnp.linalg.inv(self.env.B.T @ P @ self.env.B + self.env.R) @ (
            self.env.B.T @ P @ self.env.A
        )

    def init(
        self,
        rng: jax.random.PRNGKey,
        num_steps: int,
        warmup_steps: int,
        excitation: float = 1.0,
    ) -> ControllerState:
        return None

    @partial(jax.jit, static_argnums=(0,))
    def policy_fn(
        self,
        rng: jax.random.PRNGKey,
        controller_state: ControllerState,
        state: jnp.ndarray,
    ) -> (ControllerState, jnp.ndarray):
        return controller_state, -self.K @ state

    def on_completion_fn(
        self,
        controller_state: ControllerState,
        state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray,
    ) -> ControllerState:
        return controller_state