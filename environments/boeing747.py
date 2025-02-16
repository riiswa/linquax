from core import LinearQuadraticEnv
import jax.numpy as jnp


class Boeing747(LinearQuadraticEnv):
    def __init__(self):
        A = jnp.array([
            [0.99, 0.03, -0.02, -0.32],
            [0.01, 0.47, 4.7, 0.0],
            [0.02, -0.06, 0.4, 0.0],
            [0.01, -0.04, 0.72, 0.99],
        ])

        B = jnp.array([
            [0.01, 0.99],
            [-3.44, 1.66],
            [-0.83, 0.44],
            [-0.47, 0.25],
        ])

        Q = jnp.eye(4)  # Identity matrix for Q
        R = jnp.eye(2)  # Identity matrix for R

        super().__init__(A, B, Q, R, True)

    @property
    def name(self) -> str:
        return "Boeing 747"