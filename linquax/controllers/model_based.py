from functools import partial
from typing import Optional

import jax
from flax.struct import dataclass
from jax import numpy as jnp
from tensorboardX import SummaryWriter

from linquax.controllers import Optimal
from linquax.core import Controller, ControllerState, LinearQuadraticEnv
from linquax.utils import dare


@dataclass
class ModelBasedState(ControllerState):
    t: int
    tau: int
    V: jnp.ndarray
    V_prev: jnp.ndarray

    states: jnp.ndarray
    actions: jnp.ndarray
    next_states: jnp.ndarray

    A: jnp.ndarray
    B: jnp.ndarray
    P: jnp.ndarray
    K: jnp.ndarray


class ModelBased(Controller):
    def __init__(self, env: LinearQuadraticEnv, warmup_steps: int, improved_exploration_steps: int, weight_decay: float = 1e-4, excitation: float = 2.0, writer: Optional[SummaryWriter] = None):

        super().__init__(env)
        self.weight_decay = weight_decay
        self.Theta_star = jnp.hstack((self.env.A, self.env.B))
        self.warmup_steps = warmup_steps
        self.improved_exploration_steps = improved_exploration_steps
        self.excitation = excitation
        self.writer = writer

    def log_scalar(self, tag, value, step):
        if self.writer is not None:
            jax.experimental.io_callback(partial(self.writer.add_scalar, tag=tag), None, scalar_value=value, global_step=step)

    @property
    def name(self) -> str:
        return "Nominal"

    @partial(jax.jit, static_argnums=(0,))
    def rls(self, controller_state: ModelBasedState):
        X = jnp.hstack((controller_state.states, controller_state.actions))
        Y = controller_state.next_states
        reg_I = jnp.eye(self.state_dim + self.action_dim) * self.weight_decay
        Theta = jnp.linalg.solve(X.T @ X + reg_I, X.T @ Y).T

        A = Theta[:, : self.state_dim]
        B = Theta[:, self.state_dim :]

        return A, B

    @partial(jax.jit, static_argnums=(0,))
    def update(self, rng: jax.random.PRNGKey, controller_state: ModelBasedState):
        A, B = self.rls(controller_state)

        self.log_scalar("loss/A", jnp.linalg.norm(A - self.env.A), controller_state.t - self.warmup_steps)
        self.log_scalar("loss/B", jnp.linalg.norm(B - self.env.B), controller_state.t - self.warmup_steps)

        controller_state = self.parameter_estimation(rng, controller_state.replace(A=A, B=B))
        A = controller_state.A
        B = controller_state.B
        P = dare(A, B, self.env.Q, self.env.R, controller_state.P)
        K = jnp.linalg.inv(B.T @ P @ B + self.env.R) @ (B.T @ P @ A)

        return controller_state.replace(P=P, K=K, V_prev=controller_state.V, tau=0)

    def init(
        self,
        rng: jax.random.PRNGKey,
        num_steps: int,
    ) -> ControllerState:
        controller_state = ModelBasedState(
            t=0,
            tau = 0,
            V=jnp.identity(self.state_dim + self.action_dim) * self.weight_decay,
            V_prev=jnp.identity(self.state_dim + self.action_dim) * self.weight_decay,
            states=jnp.zeros((self.warmup_steps + num_steps, self.state_dim)),
            actions=jnp.zeros((self.warmup_steps + num_steps, self.action_dim)),
            next_states=jnp.zeros((self.warmup_steps + num_steps, self.state_dim)),
            A=jnp.zeros((self.state_dim, self.state_dim)),
            B=jnp.zeros((self.state_dim, self.action_dim)),
            P=jnp.zeros((self.state_dim, self.state_dim)),
            K=jnp.zeros((self.action_dim, self.state_dim)),
        )

        optimal = Optimal(self.env)

        def policy_fn(
            _rng: jax.random.PRNGKey,
            controller_state: ModelBasedState,
            state: jnp.ndarray,
        ):
            _, action = optimal.policy_fn(_rng, None, state)
            return (
                controller_state,
                action + jax.random.normal(_rng, (self.action_dim,)) * self.excitation,
            )

        def on_completion_fn(
            controller_state: ModelBasedState,
            state: jnp.ndarray,
            action: jnp.ndarray,
            next_state: jnp.ndarray,
        ) -> ControllerState:
            return controller_state.replace(
                t=controller_state.t + 1,
                states=controller_state.states.at[controller_state.t].set(state),
                actions=controller_state.actions.at[controller_state.t].set(action),
                next_states=controller_state.next_states.at[controller_state.t].set(
                    next_state
                ),
            )

        rng_simulation, rng_update = jax.random.split(rng)

        controller_state, _, _ = self.env.simulate(
            rng_simulation, controller_state, policy_fn, on_completion_fn, self.warmup_steps
        )

        return controller_state

    @partial(jax.jit, static_argnums=(0,))
    def parameter_estimation(self, rng: jax.random.PRNGKey, controller_state: ModelBasedState):
        return controller_state

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, controller_state: ModelBasedState, state: jnp.ndarray):
        return -controller_state.K @ state

    @partial(jax.jit, static_argnums=(0,))
    def policy_fn(
        self,
        rng: jax.random.PRNGKey,
        controller_state: ModelBasedState,
        state: jnp.ndarray,
    ) -> (ControllerState, jnp.ndarray):

        rng_update, rng_action = jax.random.split(rng)

        t = controller_state.t - self.warmup_steps

        controller_state = jax.lax.cond(
            (jnp.linalg.det(controller_state.V)
            > 2 * jnp.linalg.det(controller_state.V_prev)) & (controller_state.tau >= 10),
            lambda cs: self.update(rng_update, cs),
            lambda cs: cs,
            controller_state,
        )
        action = self.get_action(controller_state, state) + jax.lax.select(t < self.improved_exploration_steps, jax.random.normal(rng_action) * self.excitation, 0.)

        return controller_state, action

    @partial(jax.jit, static_argnums=(0,))
    def on_completion_fn(
        self,
        controller_state: ModelBasedState,
        state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray,
    ) -> ControllerState:
        z = jnp.hstack((state, action))[:, None]
        return controller_state.replace(
            t=controller_state.t + 1,
            states=controller_state.states.at[controller_state.t].set(state),
            actions=controller_state.actions.at[controller_state.t].set(action),
            next_states=controller_state.next_states.at[controller_state.t].set(
                next_state
            ),
            V = controller_state.V + z @ z.T,
            tau=controller_state.tau + 1,
        )