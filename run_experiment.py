import importlib
import os.path
import time
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig

import jax

from environments import make_env

@dataclass
class ExperimentConfig(DictConfig):
    env_id: str
    seed: int
    horizon: int
    noise: float
    warmup_steps: int
    improved_exploration_steps: int
    policy: str
    policy_kwargs: dict
    exp_name: str
    progress_file: str

@hydra.main(version_base=None, config_path="conf", config_name="config")
def experiment(cfg : ExperimentConfig) -> None:
    env = make_env(cfg.env_id)
    env.step_cov = cfg.noise
    controller = getattr(importlib.import_module('controllers'), cfg.policy)(
        env,
        warmup_steps=cfg.warmup_steps,
        improved_exploration_steps=cfg.improved_exploration_steps,
        **cfg.policy_kwargs
    )

    path = os.path.join(cfg.exp_name, env.name.replace("/", "_"), controller.name)
    file_path = os.path.join(path,
                             f"result__seed_{cfg.seed}__horizon_{cfg.horizon}__noise_{cfg.noise}__ws_{cfg.warmup_steps}__es_{cfg.improved_exploration_steps}")

    rng = jax.random.PRNGKey(cfg.seed)
    rng, rng_init = jax.random.split(rng)

    controller_state = controller.init(rng_init, cfg.horizon)

    print(f"[{file_path}] Starting experiment: {cfg.exp_name}")

    start = time.time()
    _, _, costs = jax.block_until_ready(
        env.simulate(rng, controller_state, controller.policy_fn, controller.on_completion_fn, cfg.horizon)
    )
    end = time.time()

    os.makedirs(path, exist_ok=True)

    jax.numpy.save(file_path, {'regret': costs - controller.cost_star, 'time': end - start})

    print(f"[{file_path}] Simulation completed in {end - start:.4f} seconds.")

    with open(cfg.progress_file, 'a+') as f:
        f.write(path + '\n')

if __name__ == "__main__":
    experiment()