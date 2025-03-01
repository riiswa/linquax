from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from tensorboardX import SummaryWriter

from linquax.controllers.model_based import ModelBasedState
from linquax.core import LinearQuadraticEnv
from linquax.controllers import OFULQ
from linquax.utils import dare, dlyap


@jax.jit
def scalar_perturbation(rng, Theta, noise_scale):
    n, m = Theta.shape
    rng_i, rng_j, rng_noise = jax.random.split(rng, 3)
    u = jnp.zeros(n).at[jax.random.randint(rng_i, (), 0, n)].set(1)
    v = jnp.zeros(m).at[jax.random.randint(rng_j, (), 0, n)].set(1)
    noise = jax.random.uniform(rng_noise)
    return u, v, noise * noise_scale

@jax.jit
def create_perturbation_matrix(u, v, noise):
    return jnp.outer(u, v.T) * noise


class MED(OFULQ):
    @property
    def name(self) -> str:
        return "MED-LQ"

    def __init__(self, env: LinearQuadraticEnv, warmup_steps: int, improved_exploration_steps: int, delta = 1e-4, excitation: float = 2.0, n_samples: int = 128, check_stability: bool = True, writer: Optional[SummaryWriter] = None):
        super().__init__(env, delta=delta, warmup_steps=warmup_steps, improved_exploration_steps=improved_exploration_steps, excitation=excitation, writer=writer)
        self.n_samples = n_samples
        self.check_stability = check_stability


    @partial(jax.jit, static_argnums=(0,))
    def parameter_estimation(self, rng: jax.random.PRNGKey, controller_state: ModelBasedState):
        Theta_nom = jnp.hstack((controller_state.A, controller_state.B))

        perturbations = jax.vmap(scalar_perturbation, in_axes=(0, None, None))(jax.random.split(rng, self.n_samples), Theta_nom, 1)
        X = jax.vmap(create_perturbation_matrix, in_axes=(0, 0, 0))(*perturbations)

        candidates = Theta_nom[jnp.newaxis, :] + X

        def optimal_gain(Theta):
            A = Theta[:, :self.state_dim]
            B = Theta[:, self.state_dim:]

            P = dare(A, B, self.env.Q, self.env.R)
            return jnp.linalg.inv(B.T @ P @ B + self.env.R) @ (
                    B.T @ P @ A
            )

        def closed_loop(Theta, K = None):
            A = Theta[:, :self.state_dim]
            B = Theta[:, self.state_dim:]
            return A - B @ (K if K is not None else optimal_gain(Theta))

        def interp(t, Theta):
            return Theta_nom + t * (Theta - Theta_nom)


        def loss(t, Theta, K1, K2):
            Theta_t = interp(t, Theta)
            return jnp.trace(dlyap(closed_loop(Theta_t, K1).T, self.env.Q + K1.T @ self.env.R @ K1)) - jnp.trace(dlyap(closed_loop(Theta_t, K2).T, self.env.Q + K2.T @ self.env.R @ K2))

        K_nom = optimal_gain(Theta_nom)
        K_candidates = jax.vmap(optimal_gain)(candidates)

        def stability_check(Theta, K):
            # Compute closed-loop system matrices
            A_theta_K = closed_loop(Theta, K)
            A_theta_nom_K = closed_loop(Theta_nom, K)
            A_theta_K_nom = closed_loop(Theta, K_nom)
            A_theta_nom_K_nom = closed_loop(Theta_nom, K_nom)

            # Compute eigenvalues
            eig_A_theta_K = jnp.linalg.eigvals(A_theta_K)
            eig_A_theta_K_nom = jnp.linalg.eigvals(A_theta_K_nom @ A_theta_nom_K_nom)
            eig_A_theta_nom_K = jnp.linalg.eigvals(A_theta_K @ A_theta_nom_K)

            #jax.debug.print("{a} {b} {c}", a=jnp.all(jnp.abs(eig_A_theta_K) < 1), b= jnp.all(eig_A_theta_K_nom >= 0), c= jnp.all(eig_A_theta_nom_K >= 0) )

            # Stability conditions
            return (
                    jnp.all(jnp.abs(eig_A_theta_K) < 1) &  # Discrete-time stability (eigenvalues inside unit circle)
                    jnp.all(eig_A_theta_K_nom >= 0) &  # Non-negative eigenvalues condition 1
                    jnp.all(eig_A_theta_nom_K >= 0)  # Non-negative eigenvalues condition 2
            )


        ts = jnp.zeros(self.n_samples)
        fun = jax.vmap(jax.value_and_grad(loss), in_axes=(0, 0, None, 0))
        res0 = None
        for i in range(20):
            res = fun(ts, candidates, K_nom, K_candidates)
            if i == 0:
                res0 = res
            ts = ts - res[0] / res[1]
        if self.check_stability:
            mask = jax.vmap(stability_check, in_axes=(0, 0))(candidates, K_candidates) & (ts > 0.) & (ts < 1.) & (res0[0] < -0.1)
        else:
            mask = (ts > 0.) & (ts < 1.) & (res0[0] < -0.1)

        self.log_scalar("metrics/mask", mask.mean(), controller_state.t - self.warmup_steps)

        Theta_confusing = jnp.where(mask[:, jnp.newaxis, jnp.newaxis], jax.vmap(interp, in_axes=(0, 0))(ts, candidates),
                                    candidates)

        A_K_nom = closed_loop(Theta_nom)
        Omega = jnp.identity(self.state_dim) * self.env.step_cov
        Sigma = dlyap(A_K_nom, Omega)
        A_K_candidates = jax.vmap(closed_loop)(Theta_confusing)

        K = jax.vmap(lambda A_K: 0.5 * jnp.trace((A_K_nom - A_K).T @ jnp.linalg.inv(Omega) @ (A_K_nom - A_K) @ Sigma))(
            A_K_candidates)
        U = jax.vmap(
            lambda Theta: jnp.trace((Theta) @ jnp.linalg.inv(controller_state.V) @ (Theta).T))(
            candidates)

        H = -K/U
        w = jax.nn.softmax(H, where=mask & (H < 0))
        Theta_hat = Theta_nom + jnp.tensordot(w, X, axes=(0, 0))

        A = Theta_hat[:, :self.state_dim]
        B = Theta_hat[:, self.state_dim:]

        return controller_state.replace(A=A, B=B)

