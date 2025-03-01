from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxopt import FixedPointIteration

@jax.jit
def dlyap(A: jnp.ndarray, Q: jnp.ndarray):
    n = A.shape[0]
    return (jnp.linalg.inv((jnp.eye(n ** 2) - jnp.kron(A, A))) @ Q.ravel()).reshape((n, n))


def dare_step(P: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray):
    return A.T @ P @ A - A.T @ P @ B @ jnp.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q

dare_solver = FixedPointIteration(fixed_point_fun=dare_step, maxiter=250, tol=0.01)


@jax.jit
def dare(A: jnp.ndarray, B: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray, P0: Optional[jnp.ndarray] = None):

    if P0 is None:
        P0 = Q
    result = dare_solver.run(P0, A, B, Q, R)
    return result.params


@partial(jax.jit, static_argnums=(0, 1))
def newton_method(f, max_steps: int = 1000, patience: int = 3):
    g = jax.grad(f)

    def step(carry):
        x, grad_x, grad_prev, i, patience_counter = carry

        # Compute new x and its gradient
        new_x = x - f(x) / (grad_x + 1e-9)
        new_grad = g(new_x)

        new_patience = patience_counter + (grad_x >= grad_prev)

        #jax.debug.print("{grad_x} {i}", i=i, grad_x=x)

        return new_x, new_grad, grad_x, i + 1, new_patience

    def cond_fun(carry):
        x, grad_x, grad_prev, i, patience_counter = carry
        return (patience_counter < patience) & (i < max_steps)

    x0 = 0.
    grad0 = g(x0)
    grad1 = g(-1.)

    final_state = jax.lax.while_loop(
        cond_fun,
        step,
        (x0, grad0, grad1, 0, 0)
    )



    return final_state[0]

# jax.debug.print("{x} {y} {z}, {i}", x=g(x - f(x) / g(x)), y=g(x), z= jnp.abs(g(x - f(x) / g(x)) - g(x)) ,i=i)

@jax.jit
def project_weighted_ball(Theta, Center, Cov, beta):
    Theta = Theta - Center

    def project():
        eigenvalues, eigenvectors = jnp.linalg.eigh(Cov)
        Theta_transformed = Theta @ eigenvectors
        term2 = jnp.diag(Theta_transformed.T @ Theta_transformed)

        def constraint(lam):
            term1 = eigenvalues / ((1 + lam * eigenvalues) ** 2)
            return beta - jnp.sum(term1 * term2)
        lam  = newton_method(constraint)

        return Theta_transformed @ (jnp.diag(1 / (1 + lam * eigenvalues))) @ eigenvectors

    return jax.lax.cond(jnp.trace(Theta @ Cov @ Theta.T) <= beta, lambda: Theta, project) + Center

def plot_cost_landscape(ofulq, A: jnp.ndarray, B: jnp.ndarray, V: jnp.ndarray):
    Theta_hat = jnp.hstack((A, B))

    beta = ofulq.state_dim * jnp.sqrt(
        2 * jnp.log(
            jnp.sqrt(jnp.linalg.det(V)) /
            (jnp.sqrt(jnp.linalg.det(jnp.eye(V.shape[0]) * ofulq.weight_decay)) * ofulq.delta)
        )
    ) + jnp.sqrt(ofulq.weight_decay)

    rng = jax.random.PRNGKey(42)
    rng1, rng2 = jax.random.split(rng)
    v1 = jax.random.normal(rng1, Theta_hat.shape)
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = jax.random.normal(rng2, Theta_hat.shape)
    v2 = v2 / jnp.linalg.norm(v2)
    v2 -= jnp.sum(v1 * v2) / jnp.sum(v1 * v1) * v1

    V1 = project_weighted_ball(Theta_hat + v1 * 100, Theta_hat, V, beta) - Theta_hat * 1.25
    V2 = project_weighted_ball(Theta_hat + v2 * 100, Theta_hat, V, beta) - Theta_hat * 1.25

    alpha_vals = jnp.linspace(-1, 1, 100)
    beta_vals = jnp.linspace(-1, 1, 100)

    # Define a function that computes loss for given alpha, beta
    def compute_loss(alpha, beta):
        c = ofulq.cost(Theta_hat + alpha * V1 + beta * V2)
        return jax.lax.cond(jnp.logical_and(c < 100000, c > 0), lambda: c, lambda: jnp.nan)

    # Vectorize over alpha and beta using vmap
    compute_loss_vmap = jax.vmap(jax.vmap(compute_loss, in_axes=(None, 0)), in_axes=(0, None))

    # Compute loss landscape efficiently
    loss_values = compute_loss_vmap(alpha_vals, beta_vals)

    def f(lv):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import numpy as np
        from aquarel import load_theme
        with load_theme("scientific"):
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)  # Create a figure and axis
            cmap = mpl.colormaps.get_cmap('inferno')
            cmap.set_bad(color='white')

            mat = ax.matshow(jnp.log(lv), cmap=cmap, extent=[-1, 1, -1, 1])
            ax.scatter([0], [0], color='black', marker='x', s=25)
            ax.text(0.05, -0.1, r"$\widetilde{\Theta}$", fontsize=14)

            fig.colorbar(mat, ax=ax, label="Log cost")  # Use fig.colorbar() instead of plt.colorbar()

            u = 0  # x-position of the center
            v = 0  # y-position of the center
            a = 0.75  # radius on the x-axis
            b = 0.75  # radius on the y-axis
            t = np.linspace(0, 2 * np.pi, 100)
            ax.plot(u + a * np.cos(t), v + b * np.sin(t), color="black", lw=2, ls="--",
                    label=r"$C(\delta)$")

            ax.set_xlabel(r"$\widetilde{\Theta} + \alpha \Delta_1$")
            ax.set_ylabel(r"$\widetilde{\Theta} + \beta \Delta_2$")
            ax.grid(False)
            ax.legend(frameon=False)

    jax.debug.callback(f, loss_values)


@jax.jit
def inv_sqrt(P):
    w, v = jnp.linalg.eigh(P)
    return v @ jnp.diag(1 / jnp.sqrt(w)) @ v.T

