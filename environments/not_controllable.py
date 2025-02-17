import jax.numpy as jnp

from core import LinearQuadraticEnv


class NotControllable(LinearQuadraticEnv):
    def __init__(self):
        A = jnp.array([
            [-2, 0, 1.1],
            [1.5, 0.9, 1.3],
            [0, 0, 0.5],
        ])

        B = jnp.array([
            [1., 0],
            [0, 1.],
            [0, 0],
        ])

        Q = jnp.eye(3)
        R = jnp.eye(2)

        super().__init__(A, B, Q, R, True)

    @property
    def name(self) -> str:
        return "Not Controllable"
