import jax

from core import LinearQuadraticEnv
import jax.numpy as jnp


class InvertedPendulum(LinearQuadraticEnv):
    def __init__(
        self,
        gravity: float = 9.81,
        cart_mass: float = 1.0,
        pendulum_mass: float = 0.1,
        pendulum_length: float = 0.5,
    ):
        A = jnp.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, pendulum_mass * gravity / cart_mass, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [
                    0.0,
                    0.0,
                    gravity
                    * (cart_mass + pendulum_mass)
                    / (pendulum_length * cart_mass),
                    0.0,
                ],
            ]
        )

        B = jnp.array(
            [[0.0], [1.0 / cart_mass], [0.0], [1.0 / (pendulum_length * cart_mass)]]
        )

        super().__init__(
            A, B, jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.0])), jnp.diag(jnp.array([1.0]))
        )

    @property
    def name(self) -> str:
        return "Inverted Pendulum"