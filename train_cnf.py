import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from functools import partial
import jax
from typing import Any, Callable, Sequence, Optional, NewType
from jax import lax, random, vmap, scipy, numpy as jnp
from jax.experimental.ode import odeint
import flax
from flax.training import train_state
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import serialization
import optax
from sklearn.datasets import make_circles


os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, t):
        # predict params
        blocksize = self.width * self.in_out_dim
        params = t.reshape((1, 1))
        params = nn.Dense(self.hidden_dim)(params)
        params = nn.tanh(params)
        params = nn.Dense(self.hidden_dim)(params)
        params = nn.tanh(params)
        params = nn.Dense(3 * blocksize + self.width)(params)

        # restructure
        params = params.reshape(-1)
        W = params[:blocksize].reshape((self.width, self.in_out_dim, 1))

        U = params[blocksize:2 * blocksize].reshape((self.width, 1, self.in_out_dim))

        G = params[2 * blocksize:3 * blocksize].reshape((self.width, 1, self.in_out_dim))
        U = U * nn.sigmoid(G)

        B = params[3 * blocksize:].reshape((self.width, 1, 1))
        return [W, B, U]


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz."""
    df_dz = jax.jacrev(f)(z)
    return jnp.diag(jnp.trace(df_dz, 0, 1, 3))


class DZDT(nn.Module):
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, z, t):
        W, B, U = HyperNetwork(self.in_out_dim, self.hidden_dim, self.width)(t)
        Z = jnp.expand_dims(z, 0)
        Z = jnp.repeat(Z, self.width, 0)
        h = nn.tanh(jnp.matmul(Z, W) + B)
        dz_dt = jnp.matmul(h, U).mean(0)

        return dz_dt


class CNF(nn.Module):
    """Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, t, states):
        z, logp_z = states

        func_dz_dt = lambda Z: DZDT(self.in_out_dim, self.hidden_dim, self.width)(Z, t)
        dz_dt = func_dz_dt(z)
        dlogp_z_dt = -trace_df_dz(func_dz_dt, z)

        return dz_dt, dlogp_z_dt.reshape((-1, 1))


class Neg_CNF(nn.Module):
    """Negative CNF for jax's odeint."""
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, t, states):
        dz_dt, dlogp_z_dt = CNF(self.in_out_dim, self.hidden_dim, self.width)(t, states)

        return -dz_dt, -dlogp_z_dt


def get_batch(num_samples):
    """Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = jnp.array(points, dtype=jnp.float32)
    logp_diff_t1 = jnp.zeros((num_samples, 1), dtype=jnp.float32)

    return x, logp_diff_t1


def create_train_state(rng, learning_rate, in_out_dim, hidden_dim, width):
    """Creates initial 'TrainState'."""
    z, logp_z = get_batch(1)
    neg_cnf = Neg_CNF(in_out_dim, hidden_dim, width)
    params = neg_cnf.init(rng, jnp.array(10.), (z, logp_z))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=neg_cnf.apply, params=params, tx=tx
    )

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def train_step(state, batch, in_out_dim, hidden_dim, width, t0, t1):
    x, logp_diff_t1 = batch
    p_z0 = lambda x: scipy.stats.multivariate_normal.logpdf(x,
                                                            mean=jnp.array([0., 0.]),
                                                            cov=jnp.array([[0.1, 0.], [0., 0.1]]))
    def loss_fn(params):
        func = lambda states, t: Neg_CNF(in_out_dim, hidden_dim, width).apply({'params': params}, -t, states)
        z_t, logp_diff_t = odeint(
            func,
            (x, logp_diff_t1),
            -jnp.array([t1, t0]),
            atol=1e-5,
            rtol=1e-5
        )
        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
        logp_x = p_z0(z_t0) - logp_diff_t0.reshape(-1)
        loss = -logp_x.mean(0)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def train(learning_rate, n_iters, batch_size, in_out_dim, hidden_dim, width, t0, t1, visual):
    """Train the model."""
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, learning_rate, in_out_dim, hidden_dim, width)

    for itr in range(1, n_iters+1):
        batch = get_batch(batch_size)
        state, loss = train_step(state, batch, in_out_dim, hidden_dim, width, t0, t1)
        print("iter: %d, loss: %.2f" % (itr, loss))

    if visual is True:
        # Convert Params of Neg_CNF to CNF
        neg_params = state.params
        neg_params = unfreeze(neg_params)
        # Get flattened-key: value list.
        neg_flat_params = {'/'.join(k): v for k, v in traverse_util.flatten_dict(neg_params).items()}
        pos_flat_params = {key[6:]: jnp.array(np.array(neg_flat_params[key])) for key in list(neg_flat_params.keys())}
        pos_unflat_params = traverse_util.unflatten_dict({tuple(k.split('/')): v for k, v in pos_flat_params.items()})
        pos_params = freeze(pos_unflat_params)
        output = viz(neg_params, pos_params, in_out_dim, hidden_dim, width, t0, t1)
        z_t_samples, z_t_density, logp_diff_t, viz_timesteps, target_sample, z_t1 = output
        create_plots(z_t_samples, z_t_density, logp_diff_t, t0, t1, viz_timesteps, target_sample, z_t1)


def solve_dynamics(dynamics_fn, initial_state, t):
    @partial(jax.jit, backend="cpu")
    def f(initial_state, t):
        return odeint(dynamics_fn, initial_state, t, atol=1e-5, rtol=1e-5)
    return f(initial_state, t)


@partial(jax.jit, backend="cpu", static_argnums=(2, 3, 4, 5, 6))
def viz(neg_params, pos_params, in_out_dim, hidden_dim, width, t0, t1):
    """Adapted from PyTorch """
    viz_samples = 10000
    viz_timesteps = 41
    target_sample, _ = get_batch(viz_samples)

    if not os.path.exists('results/'):
        os.makedirs('results/')

    z_t0 = jnp.array(np.random.multivariate_normal(mean=np.array([0., 0.]),
                                                   cov=np.array([[0.1, 0.], [0., 0.1]]),
                                                   size=viz_samples))
    logp_diff_t0 = jnp.zeros((viz_samples, 1), dtype=jnp.float32)

    func_pos = lambda states, t: CNF(in_out_dim, hidden_dim, width).apply({'params': pos_params}, t, states)
    z_t_samples, _ = solve_dynamics(func_pos, (z_t0, logp_diff_t0), jnp.linspace(t0, t1, viz_timesteps))
    # z_t_samples, _ = odeint(
    #     func_pos,
    #     (z_t0, logp_diff_t0),
    #     jnp.linspace(t0, t1, viz_timesteps),
    #     atol=1e-5,
    #     rtol=1e-5
    # )

    # Generate evolution of density
    x = jnp.linspace(-1.5, 1.5, 100)
    y = jnp.linspace(-1.5, 1.5, 100)
    points = jnp.vstack(jnp.meshgrid(x, y)).reshape([2, -1]).T

    z_t1 = jnp.array(points, dtype=jnp.float32)
    logp_diff_t1 = jnp.zeros((z_t1.shape[0], 1), dtype=jnp.float32)
    func_neg = lambda states, t: Neg_CNF(in_out_dim, hidden_dim, width).apply({'params': neg_params}, -t, states)
    z_t_density, logp_diff_t = solve_dynamics(func_neg, (z_t1, logp_diff_t1), -jnp.linspace(t1, t0, viz_timesteps))
    # z_t_density, logp_diff_t = odeint(
    #     func_neg,
    #     (z_t1, logp_diff_t1),
    #     -jnp.linspace(t1, t0, viz_timesteps),
    #     atol=1e-5,
    #     rtol=1e-5,
    # )

    return z_t_samples, z_t_density, logp_diff_t, viz_timesteps, target_sample, z_t1


def create_plots(z_t_samples, z_t_density, logp_diff_t, t0, t1, viz_timesteps, target_sample, z_t1):
    # Create plots for each timestep
    for (t, z_sample, z_density, logp_diff) in zip(
            np.linspace(t0, t1, viz_timesteps),
            z_t_samples, z_t_density, logp_diff_t
    ):
        fig = plt.figure(figsize=(12, 4), dpi=200)
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        fig.suptitle(f'{t:.2f}s')

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Target')
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Samples')
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Log Probability')
        ax3.get_xaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])

        cpus = jax.devices("cpu")
        ax1.hist2d(*jnp.transpose(target_sample), bins=300, density=True,
                   range=[[-1.5, 1.5], [-1.5, 1.5]])

        ax2.hist2d(*jnp.transpose(z_sample), bins=300, density=True,
                   range=[[-1.5, 1.5], [-1.5, 1.5]])
        p_z0 = lambda x: scipy.stats.multivariate_normal.logpdf(x,
                                                                mean=jnp.array([0., 0.]),
                                                                cov=jnp.array([[0.1, 0.], [0., 0.1]]))
        logp = p_z0(z_density) - logp_diff.reshape(-1)
        z_t1_copy = z_t1
        ax3.tricontourf(*jnp.transpose(z_t1_copy),
                        jnp.exp(logp), 200)

        plt.savefig(os.path.join('results/', f"cnf-viz-{int(t * 1000):05d}.jpg"),
                    pad_inches=0.2, bbox_inches='tight')
        plt.close()

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join('results/', f"cnf-viz-*.jpg")))]
    img.save(fp=os.path.join('results/', "cnf-viz.gif"), format='GIF', append_images=imgs,
             save_all=True, duration=250, loop=0)

    print('Saved visualization animation at {}'.format(os.path.join('results/', "cnf-viz.gif")))


if __name__ == '__main__':
    train(0.001, 10, 512, 2, 32, 64, 0., 10., True)
