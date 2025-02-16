import controlgym

from core import LinearQuadraticEnv
import jax.numpy as jnp


class ControlGymEnv(LinearQuadraticEnv):
    def __init__(self, id: str):
        self.id = id
        gym_env = controlgym.make(id)
        is_linear = gym_env.category == "linear" or gym_env.id in [
            "convection_diffusion_reaction",
            "wave",
            "schrodinger",
        ]
        assert is_linear and all(
            hasattr(gym_env, attr) for attr in ["A", "B2"]
        ), "The environment is not linear or system matrices do not exist. LQR is not applicable"

        A = jnp.array(gym_env.A)
        B = jnp.array(gym_env.B2)
        Q = jnp.array(gym_env.Q) if hasattr(gym_env, "Q") else jnp.identity(gym_env.n_state)
        R = jnp.array(gym_env.R) if hasattr(gym_env, "R") else jnp.identity(gym_env.n_action)

        super().__init__(A, B, Q, R, True)

    @property
    def name(self) -> str:
        return "controlgym/" + self.id