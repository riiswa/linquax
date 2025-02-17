import jax.numpy as jnp

from core import LinearQuadraticEnv


class LargeTransient(LinearQuadraticEnv):
    def __init__(self):
        A = jnp.array([
            [1, 0, 0],
            [1.1, 1, 0],
            [0, 1.1, 1],
        ])

        B = jnp.eye(3)
        Q = jnp.eye(3)
        R = jnp.eye(3)

        super().__init__(A, B, Q, R, True)

    @property
    def name(self) -> str:
        return "Large Transient"
