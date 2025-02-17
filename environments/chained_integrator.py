import jax.numpy as jnp

from core import LinearQuadraticEnv


class ChainedIntegrator(LinearQuadraticEnv):
    def __init__(self):
        A = jnp.array([
            [1, 0.1],
            [0, 1],
        ])

        B = jnp.eye(2)

        Q = R = jnp.eye(2)

        super().__init__(A, B, Q, R, True)

    @property
    def name(self) -> str:
        return "Chained Integrator"
