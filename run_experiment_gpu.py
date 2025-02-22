import importlib
import os.path
import sys
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
    n_seeds: int

def make_experiment(cfg: ExperimentConfig):
    env = make_env(cfg.env_id)
    env.step_cov = cfg.noise
    controller = getattr(importlib.import_module('controllers'), cfg.policy)(
        env,
        warmup_steps=cfg.warmup_steps,
        improved_exploration_steps=cfg.improved_exploration_steps,
        **cfg.policy_kwargs
    )

    def f():
        rng = jax.random.PRNGKey(cfg.seed)
        rng, rng_init = jax.random.split(rng)
        states = jax.vmap(controller.init, in_axes=(0, None))(jax.random.split(rng_init, cfg.n_seeds), cfg.horizon)

        _, _, costs = jax.block_until_ready(
            jax.vmap(env.simulate, in_axes=(0, 0, None, None, None))(
                jax.random.split(rng, cfg.n_seeds),
                states,
                controller.policy_fn,
                controller.on_completion_fn,
                cfg.horizon
            )
        )

        return costs

    return f, env.name, controller.name, controller.cost_star


@hydra.main(version_base=None, config_path="conf", config_name="config")
def experiment(cfg : ExperimentConfig) -> None:
    f, env_name, controller_name, cost_star = make_experiment(cfg)
    f = jax.jit(f)

    path = os.path.join(cfg.exp_name, env_name, controller_name)

    print(f"[{path}] Compilation start.")
    start = time.time()
    _ = f.lower()
    end = time.time()
    print(f"[{path}] Compilation completed in {end - start:.4f} seconds.")



    def get_file_path(seed):
        return os.path.join(path,
                     f"result__seed_{seed}__horizon_{cfg.horizon}__noise_{cfg.noise}__ws_{cfg.warmup_steps}__es_{cfg.improved_exploration_steps}")

    print(f"[{path}] Simulation start.")

    start = time.time()
    _, _, costs = jax.block_until_ready(f())
    emd = time.time()

    os.makedirs(path, exist_ok=True)

    for seed in range(cfg.n_seeds):
        jax.numpy.save(get_file_path(seed), {"regret": costs[seed] - cost_star})
        jax.numpy.save(os.path.join(path, "time"), end - start)

    print(f"[{path}] Simulation completed in {end - start:.4f} seconds.")

if __name__ == "__main__":
    experiment()