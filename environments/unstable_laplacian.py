import jax.numpy as jnp

from core import LinearQuadraticEnv


class UnstableLaplacian(LinearQuadraticEnv):
    def __init__(self):
        A = jnp.array([
            [1.01, 0.01, 0],
            [0.01, 1.01, 0.01],
            [0, 0.01, 1.01],
        ])

        B = jnp.eye(3)
        Q = jnp.eye(3)
        R = jnp.eye(3)

        super().__init__(A, B, Q, R, True)

    @property
    def name(self) -> str:
        return "Unstable Laplacian Dynamics"