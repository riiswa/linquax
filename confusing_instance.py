import jax
import jax.numpy as jnp

from environments import InvertedPendulum
from utils import dare, dlyap

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    env1 = InvertedPendulum(pendulum_mass=0.1, pendulum_length=0.4)
    #env2 = InvertedPendulum(pendulum_mass=0.3, pendulum_length=1.)
    env2 = InvertedPendulum(pendulum_mass=0.3, pendulum_length=1.)



    Q = env1.Q
    R = env1.R

    P1 = dare(env1.A, env1.B, Q, R)
    K1 = jnp.linalg.inv(env1.B.T @ P1 @ env1.B + R) @ (env1.B.T @ P1 @ env1.A)
    P2 = dare(env2.A, env2.B, Q, R)
    K2 = jnp.linalg.inv(env2.B.T @ P2 @ env2.B + R) @ (env2.B.T @ P2 @ env2.A)

    # Sigma1 = dlyap(env1.A + env1.B @ K1, jnp.eye(env1.A.shape[0]))
    # Sigma2 = dlyap(env2.A + env2.B @ K2, jnp.eye(env1.A.shape[0]))
    # kl_cov = 0.5 * (jnp.trace(jnp.linalg.inv(Sigma2) @ Sigma1) - Sigma1.shape[0] + jnp.log(jnp.linalg.det(Sigma2) / jnp.linalg.det(Sigma1)))
    # print(kl_cov)
    #
    #
    # exit()


    ts = jnp.linspace(0, 1, 100)

    At, Bt = jax.vmap(lambda t: (env1.A + t * (env2.A - env1.A), env1.B + t * (env2.B - env1.B)))(ts)

    def cost(A, B, K):
        return jnp.trace(dlyap((A - B @ K).T, Q + K.T @ R @ K))

    vcost = jax.vmap(cost, in_axes=(0, 0, None))
    K1_costs = vcost(At, Bt, K1)
    K2_costs = vcost(At, Bt, K2)

    def loss(t):
        A = env1.A + t * (env2.A - env1.A)
        B = env1.B + t * (env2.B - env1.B)
        return cost(A, B, K1) - cost(A, B, K2)

    t = 0.
    tx = [t]
    losses = [loss(t)]
    for i in range(10):
        t = t - (loss(t) / jax.grad(loss)(t))
        tx.append(t)
        losses.append(loss(t))

    import matplotlib.pyplot as plt
    from aquarel import load_theme
    import matplotlib.patches as patches

    with load_theme("scientific"):
        plt.figure(figsize=(3.5, 2.5), dpi=300)
        plt.plot(ts, K1_costs - K2_costs, c="black")
        plt.xticks((jnp.linspace(0, 1, 6) + 0.2)[:5])
        ax = plt.gca()
        ax.spines.bottom.set_position('zero')
        ax.spines[['top', 'right']].set_visible(False)
        plt.scatter(tx, losses, c="red", s=25, alpha=0.5, zorder=2)
        for i in range(len(tx)-1):
            if abs(losses[i] - losses[i + 1]) < 0.1:
                break
            arrow = patches.FancyArrowPatch((tx[i], losses[i]), (tx[i + 1], losses[i + 1]),
                             connectionstyle="arc3,rad=.5", color="red",
mutation_scale=4)
            ax.add_patch(arrow)
        # plt.plot(ts, K1_costs)
        # plt.plot(ts, K2_costs)
        # print(K1_costs)
        # print(K2_costs)
        plt.ylabel(r"$J_K(\Theta(\alpha)) - J_{K'}(\Theta(\alpha))$")
        plt.xlabel(r"$\alpha$")
        ax.xaxis.set_label_coords(0.5, 0.)
        plt.tight_layout()
        plt.savefig("objective_landscape.pdf")
        plt.show()

