from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jaxopt

from controllers.model_based import ModelBased, ModelBasedState
from core import LinearQuadraticEnv
from utils import dare, project_weighted_ball



class OFULQ(ModelBased):
    @property
    def name(self):
        return "OFULQ"

    def __init__(self, env: LinearQuadraticEnv, warmup_steps: int, improved_exploration_steps: int, delta = 1e-4, learning_rate: float = 1e-3, excitation: float = 2.0):
        super().__init__(env, warmup_steps=warmup_steps, improved_exploration_steps=improved_exploration_steps, excitation=excitation)
        self.delta = delta

        self.solver = jaxopt.ProjectedGradient(
            fun=self.cost,
            projection = lambda Theta, hyperparams: project_weighted_ball(Theta, *hyperparams),
            stepsize=learning_rate
        )

        self.S = jnp.trace(self.Theta_star @ self.Theta_star.T)

        self.cost_star = self.cost(self.Theta_star)
        self.D = self.cost_star * 20

    @partial(jax.jit, static_argnums=(0,))
    def confidence_threshold(self, V):
        # return jnp.trace((Theta_hat - self.Theta_star) @ V @ (Theta_hat - self.Theta_star).T)
        det_V = jnp.linalg.det(V)
        log_term = jnp.log(jnp.sqrt(det_V) / self.delta)
        scaling_factor = self.env.step_cov * self.state_dim * jnp.sqrt(2 * log_term)

        return (scaling_factor + jnp.sqrt(self.weight_decay) * self.S)

    @partial(jax.jit, static_argnums=(0,))
    def cost(self, Theta, P0: Optional[jnp.ndarray] = None):
        A = Theta[:, :self.state_dim]
        B = Theta[:, self.state_dim:]

        P = dare(A, B, self.env.Q, self.env.R, P0)
        return jnp.trace(P)

    @partial(jax.jit, static_argnums=(0,))
    def is_stabilizable(self, Theta, P0: Optional[jnp.ndarray] = None):
        c = self.cost(Theta, P0)
        return (c < self.D) & (c > 0)

    @partial(jax.jit, static_argnums=(0, 4))
    def pgd(self, Theta, hyperparams_proj, P0, max_steps: int = 500, patience: int = 5, tol: float = 1e-3):
        def step(carry):
            Theta, opt_state, best_Theta, best_cost, patience_count, step_count = carry

            new_Theta, opt_state = self.solver.update(Theta, opt_state, hyperparams_proj)
            current_cost = self.cost(new_Theta, P0)

            #jax.debug.print("{c}", c=current_cost)

            improved = current_cost < best_cost - tol

            new_best_Theta = jax.lax.select(improved, new_Theta, best_Theta)
            new_best_cost = jax.lax.select(improved, current_cost, best_cost)
            new_patience_count = jax.lax.select(improved, 0, patience_count + 1)

            return (new_Theta, opt_state, new_best_Theta, new_best_cost,
                    new_patience_count, step_count + 1)

        opt_state = self.solver.init_state(Theta, hyperparams_proj)

        # Define continuation condition
        def continue_condition(carry):
            Theta, _, _, _, patience_count, step_count = carry
            return (
                    ~jnp.isnan(jnp.sum(Theta)) &  # Check for NaN
                    self.is_stabilizable(Theta, P0) &
                    (patience_count < patience) &  # Haven't exceeded patience
                    (step_count < max_steps)  # Haven't exceeded max steps
            )

        initial_carry = (Theta, opt_state, Theta, jnp.inf, 0, 0)
        final_state = jax.lax.while_loop(
            continue_condition,
            step,
            initial_carry
        )

        return final_state[2]


    @partial(jax.jit, static_argnums=(0,))
    def parameter_estimation(self, rng: jax.random.PRNGKey, controller_state: ModelBasedState):
        Theta_hat = jnp.hstack((controller_state.A, controller_state.B))

        beta = self.confidence_threshold(controller_state.V)

        Theta_hat = self.pgd(Theta_hat, (Theta_hat, controller_state.V, beta), controller_state.P)

        A = Theta_hat[:, :self.state_dim]
        B = Theta_hat[:, self.state_dim:]

        return controller_state.replace(A=A, B=B)

