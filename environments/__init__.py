from environments.boeing747 import Boeing747
from environments.controlgym_wrapper import ControlGymEnv
from environments.inverted_pendulum import InvertedPendulum
from environments.uav import UAV
from environments.unstable_laplacian import UnstableLaplacian
from environments.large_transient import LargeTransient
from environments.chained_integrator import ChainedIntegrator
from environments.not_controllable import NotControllable


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
