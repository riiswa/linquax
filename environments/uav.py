import jax.numpy as jnp

from core import LinearQuadraticEnv


class UAV(LinearQuadraticEnv):
    def __init__(self):
        A = jnp.array([
            [1, 0.5, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.5],
            [0, 0, 0, 1],
        ])

        B = jnp.array([
            [0.125, 0],
            [0.5, 0],
            [0, 0.125],
            [0, 0.5],
        ])

        Q = jnp.diag(jnp.array([1, 0.1, 2, 0.2]))
        R = jnp.eye(2)  # Identity matrix for R

        super().__init__(A, B, Q, R, True)

    @property
    def name(self) -> str:
        return "Unmanned Aerial Vehicle"