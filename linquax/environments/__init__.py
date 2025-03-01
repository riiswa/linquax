from linquax.environments.boeing747 import Boeing747
from linquax.environments.controlgym_wrapper import ControlGymEnv
from linquax.environments.inverted_pendulum import InvertedPendulum
from linquax.environments.uav import UAV
from linquax.environments.unstable_laplacian import UnstableLaplacian
from linquax.environments.large_transient import LargeTransient
from linquax.environments.chained_integrator import ChainedIntegrator
from linquax.environments.not_controllable import NotControllable


env_dict = {
    "inverted_pendulum": InvertedPendulum,
    "boeing747": Boeing747,
    "uav": UAV,
    "unstable_laplacian": UnstableLaplacian,
    "large_transient": LargeTransient,
    "not_controllable": NotControllable,
    "chained_integrator": ChainedIntegrator,
}

def make_env(id: str):
    return env_dict[id]() if id in env_dict else ControlGymEnv(id)
