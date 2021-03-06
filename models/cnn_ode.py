from functools import partial
import jax
from typing import Any, Callable, Sequence, Optional, NewType
from jax import lax, random, vmap, numpy as jnp
from jax.experimental.ode import odeint
import flax
from flax.training import train_state
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import serialization
import optax
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
import os


# Define Residual Block
class ResDownBlock(nn.Module):
    """Single ResBlock w/ downsample"""
    dim_out: Any = 64

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        f_x = nn.relu(nn.GroupNorm(self.dim_out)(x))
        x = nn.Conv(features=self.dim_out, kernel_size=(1, 1), strides=(2, 2))(x)
        f_x = nn.Conv(features=self.dim_out, kernel_size=(3, 3), strides=(2, 2))(f_x)
        f_x = nn.relu(nn.GroupNorm(self.dim_out)(f_x))
        f_x = nn.Conv(features=self.dim_out, kernel_size=(3, 3))(f_x)
        x = f_x + x
        return x


class ResBlock(nn.Module):
    """Single Resblock w/o downsample"""
    dim_out: Any = 64
    ksize: Any = 3

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        f_x = nn.relu(nn.GroupNorm(self.dim_out)(x))
        f_x = nn.Conv(features=self.dim_out, kernel_size=(self.ksize, self.ksize))(f_x)
        f_x = nn.relu(nn.GroupNorm(self.dim_out)(f_x))
        f_x = nn.Conv(features=self.dim_out, kernel_size=(self.ksize, self.ksize))(f_x)
        x = f_x + x
        return x


class ConcatConv2D(nn.Module):
    """Concat dynamics to hidden layer"""
    dim_out: Any = 64
    ksize: Any = 3

    @nn.compact
    def __call__(self, x, t):
        tt = jnp.ones_like(x[..., :1]) * t
        ttx = jnp.concatenate([tt, x], -1)
        return nn.Conv(features=self.dim_out, kernel_size=(self.ksize, self.ksize))(ttx)


# Define Model for Mnist example in Neural ODE
class SmallResNet(nn.Module):
    dim_out: Any = 64
    ksize: Any = 3

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = nn.Conv(features=self.dim_out, kernel_size=(self.ksize, self.ksize))(x)
        x = ResDownBlock()(x)
        x = ResDownBlock()(x)

        x = ResBlock()(x)
        x = ResBlock()(x)
        x = ResBlock()(x)
        x = ResBlock()(x)
        x = ResBlock()(x)
        x = ResBlock()(x)

        x = nn.GroupNorm(self.dim_out)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (1, 1))

        x = x.reshape((x.shape[0], -1))     # flatten

        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)

        return x


# Define Model for Mnist example in Neural ODE
class ODEfunc(nn.Module):
    """ODE function which replace ResNet"""
    dim_out: Any = 64
    ksize: Any = 3

    @nn.compact
    def __call__(self, inputs, t):
        # TODO Count number of function estimation
        # nfe_counter = NFEcounter()
        # nfe_counter()

        x = inputs
        out = nn.GroupNorm(self.dim_out)(x)
        out = nn.relu(out)
        out = ConcatConv2D(self.dim_out, self.ksize)(out, t)
        out = nn.GroupNorm(self.dim_out)(out)
        out = nn.relu(out)
        out = ConcatConv2D(self.dim_out, self.ksize)(out, t)
        out = nn.GroupNorm(self.dim_out)(out)

        return out


class NFEcounter(nn.Module):

    @nn.compact
    def __call__(self):
        is_initialized = self.has_variable('nfe', 'nfe')
        nfe = self.variable('nfe', 'nfe', jnp.array, [0])
        if is_initialized:
            nfe.value += 1


class ODEBlock(nn.Module):
    """ODE block which contains odeint"""
    tol: Any = 1.

    @nn.compact
    def __call__(self, x, params):
        ode_func = ODEfunc()
        init_state, final_state = odeint(partial(ode_func.apply, {'params': params}),
                                         x, jnp.array([0., 1.]),
                                         rtol=self.tol, atol=self.tol)
        return final_state


class ODEBlockVmap(nn.Module):
    """Apply vmap to ODEBlock"""
    tol: Any = 1.

    @nn.compact
    def __call__(self, x, params):
        vmap_odeblock = nn.vmap(ODEBlock,
                                variable_axes={'params': 0, 'nfe': None},
                                split_rngs={'params': True, 'nfe': False},
                                in_axes=(0, None))

        return vmap_odeblock(tol=self.tol, name='odeblock')(x, params)


class FullODENet(nn.Module):
    """Full ODE net which contains two downsampling layers, ODE block and linear classifier."""
    dim_out: Any = 64
    ksize: Any = 3
    tol: Any = 1.

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = nn.Conv(features=self.dim_out, kernel_size=(self.ksize, self.ksize))(x)
        x = ResDownBlock()(x)
        x = ResDownBlock()(x)

        ode_func = ODEfunc()
        init_fn = lambda rng, x: ode_func.init(random.split(rng)[-1], x, 0.)['params']
        ode_func_params = self.param('ode_func', init_fn, jnp.ones_like(x[0]))
        x = ODEBlockVmap(tol=self.tol)(x, ode_func_params)

        x = nn.GroupNorm(self.dim_out)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (1, 1))

        x = x.reshape((x.shape[0], -1))     # flatten

        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)

        return x


# Define loss
@jax.jit
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))


# Metric computation
@jax.jit
def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


def create_train_state(rng, learning_rate):
    """Creates initial 'TrainState'."""
    cnn = FullODENet()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx
    )


# Training step
@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = FullODENet().apply({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics


# Evaluation step
@jax.jit
def eval_step(params, batch):
    logits = FullODENet().apply({'params': params}, batch['image'])
    return compute_metrics(logits=logits, labels=batch['label'])


# Train function
def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch"""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]    # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in tqdm(perms):
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }
    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100
    ))

    return state


# Eval function
def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']


if __name__ == '__main__':
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # TODO Build learning rate decay as Neural ODE paper
    learning_rate = 0.0001

    state = create_train_state(init_rng, learning_rate)
    del init_rng  # Must not be used anymore.

    num_epochs = 1
    batch_size = 128

    for epoch in tqdm(range(1, num_epochs + 1)):
        rng, input_rng = jax.random.split(rng)
        state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
        test_loss, test_accuracy = eval_model(state.params, test_ds)
        print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, test_loss, test_accuracy * 100
        ))
