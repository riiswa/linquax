from functools import partial

import jax
import jax.numpy as jnp

from controllers import TS
from controllers.model_based import ModelBasedState
from core import LinearQuadraticEnv
from utils import dlyap, dare


class MED(TS):
    @property
    def name(self) -> str:
        return "MED-LQ"

    def __init__(self, env: LinearQuadraticEnv, warmup_steps: int, improved_exploration_steps: int, delta = 1e-4, excitation: float = 2.0, n_samples: int = 32):
        super().__init__(env, delta=delta, warmup_steps=warmup_steps, improved_exploration_steps=improved_exploration_steps, excitation=excitation)
        self.n_samples = n_samples


    @partial(jax.jit, static_argnums=(0,))
    def parameter_estimation(self, rng: jax.random.PRNGKey, controller_state: ModelBasedState):
        Theta_hat = jnp.hstack((controller_state.A, controller_state.B))

        rng, rng_sample = jax.random.split(rng)
        candidates, accepted = self.rejection_sampling(rng_sample, Theta_hat, controller_state.V, controller_state.P, self.n_samples)

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
            return Theta_hat + t * (Theta - Theta_hat)


        def loss(t, Theta, K1, K2):
            Theta_t = interp(t, Theta)
            return jnp.trace(dlyap(closed_loop(Theta_t, K1).T, self.env.Q + K1.T @ self.env.R @ K1)) - jnp.trace(dlyap(closed_loop(Theta_t, K2).T, self.env.Q + K2.T @ self.env.R @ K2))

        K_hat = optimal_gain(Theta_hat)
        K_candidates = jax.vmap(optimal_gain)(candidates)

        ts = jnp.zeros(self.n_samples)
        fun = jax.vmap(jax.value_and_grad(loss), in_axes=(0, 0, None, 0))
        for i in range(20):
            res = fun(ts, candidates, K_hat, K_candidates)
            ts = ts - res[0] / res[1]
        mask = (ts > 0.) & (ts < 1.) & accepted & (fun(jnp.zeros(self.n_samples), candidates, K_hat, K_candidates)[0] <= 0)


        Theta_confusing = jnp.where(mask[:, jnp.newaxis, jnp.newaxis], jax.vmap(interp, in_axes=(0, 0))(ts, candidates), candidates)

        A_K_hat = closed_loop(Theta_hat)
        Omega = jnp.identity(self.state_dim) * self.env.step_cov
        Sigma = dlyap(A_K_hat, Omega)
        A_K_candidates = jax.vmap(closed_loop)(Theta_confusing)

        # jax.debug.print("{a}\n{x}\n{b}\n",
        #                 a=jnp.all(jnp.abs(jnp.linalg.eigvals(A_K_hat)) < 1),
        #                 x=accepted,
        #                 b=jax.vmap(lambda A_K: jnp.all(jnp.abs(jnp.linalg.eigvals(A_K)) < 1))(A_K_candidates)
        # )

        K = jax.vmap(lambda A_K: 0.5 * jnp.trace((A_K_hat - A_K).T @ jnp.linalg.inv(Omega) @ (A_K_hat - A_K) @ Sigma ))(A_K_candidates)
        U = jax.vmap(lambda Theta: jnp.trace((Theta_hat - Theta) @ jnp.linalg.inv(controller_state.V) @ (Theta_hat - Theta).T))(candidates)
        H = -K#/(self.confidence_threshold(controller_state.V) * U)
        H = jnp.append(H, 0)
        #jax.debug.print("{t} {H}, {D}, {i}\n", H=H, D=jax.nn.softmax(H), i=jax.random.categorical(rng, H), t=controller_state.t)
        candidates = jnp.append(candidates, Theta_hat[jnp.newaxis, :], axis=0)
        p = jax.nn.softmax(H, where=jnp.append(mask, True))

        i = jax.random.choice(rng, jnp.arange(self.n_samples + 1), p=p)
        #jax.debug.print("{t} {H} {p}, {i}\n", H=jnp.where(jnp.append(mask, True), H, jnp.nan), p=p , i=i, t=controller_state.t)
        #jax.random.categorical(rng, H)

        Theta_hat = candidates[jax.lax.select(jnp.any(H> 0.), self.n_samples, i)]

        A = Theta_hat[:, :self.state_dim]
        B = Theta_hat[:, self.state_dim:]

        return controller_state.replace(A=A, B=B)

