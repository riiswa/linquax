from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp


class ControllerState(ABC):
    pass


class LinearQuadraticEnv(ABC):
    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        time_is_discrete: bool = False,
        dt: float = 0.1,
        step_cov: float = 1.,
    ):
        if time_is_discrete:
            self.A = A
            self.B = B
        else:
            self.A = jnp.eye(A.shape[0]) + A * dt
            self.B = B * dt
        self.Q = Q
        self.R = R
        self.step_cov = step_cov

    def reset(self, rng: jax.random.PRNGKey):
        return jnp.zeros(self.A.shape[0])

    @partial(jax.jit, static_argnums=(0,))
    def step_fn(
        self, rng_key: jax.random.PRNGKey, state: jnp.ndarray, action: jnp.ndarray
    ) -> (jnp.ndarray, jnp.ndarray):
        next_state = self.A @ state + self.B @ action
        noise = jax.random.normal(rng_key, next_state.shape) * self.step_cov

        state_cost = state.T @ self.Q @ state
        control_cost = action.T @ self.R @ action

        return next_state + noise, state_cost + control_cost

    #@partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def simulate(
        self,
        rng: jax.random.PRNGKey,
        controller_state: ControllerState,
        policy: Callable[
            [jax.random.PRNGKey, ControllerState, jnp.ndarray], jnp.ndarray
        ],
        on_completion: Callable[
            [ControllerState, jnp.ndarray, jnp.ndarray, jnp.ndarray], ControllerState
        ],
        num_steps,
    ) -> (jnp.ndarray, jnp.ndarray):
        initial_state = self.reset(rng)

        def step(carry, _) -> (jnp.ndarray, jnp.ndarray):
            rng, controller_state, state = carry
            rng, rng_policy = jax.random.split(rng)
            controller_state, action = policy(rng_policy, controller_state, state)
            next_state, cost = self.step_fn(rng, state, action)
            controller_state = on_completion(
                controller_state, state, action, next_state
            )
            return (rng, controller_state, next_state), (next_state, cost)

        (_, controller_state, _), (states, costs) = jax.lax.scan(
            step, (rng, controller_state, initial_state), length=num_steps
        )

        return controller_state, states, costs

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class Controller(ABC):
    def __init__(self, env: LinearQuadraticEnv):
        self.env = env
        self.state_dim, self.action_dim = self.env.B.shape

    @abstractmethod
    def init(
        self,
        rng: jax.random.PRNGKey,
        num_steps: int,
        warmup_steps: int,
        excitation: float = 1.0,
    ) -> ControllerState:
        pass

    @abstractmethod
    def policy_fn(
        self,
        rng: jax.random.PRNGKey,
        controller_state: ControllerState,
        state: jnp.ndarray,
    ) -> (ControllerState, jnp.ndarray):
        pass

    @abstractmethod
    def on_completion_fn(
        self,
        controller_state: ControllerState,
        state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray,
    ) -> ControllerState:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
