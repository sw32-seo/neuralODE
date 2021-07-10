from functools import partial
import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, vmap, numpy as jnp
from jax.experimental.ode import odeint
import flax
from flax.training import train_state
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import serialization
import optax
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm


# Define model
class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))     # flatten

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


# Define Residual Block
class ResDownBlock(nn.Module):
    """Single ResBlock w/ downsample"""

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        f_x = nn.relu(nn.GroupNorm(64)(x))
        x = nn.Conv(features=64, kernel_size=(1, 1), strides=(2, 2))(x)
        f_x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(f_x)
        f_x = nn.relu(nn.GroupNorm(64)(f_x))
        f_x = nn.Conv(features=64, kernel_size=(3, 3))(f_x)
        x = f_x + x
        return x


class ConcatConv2D(nn.Module):
    """Concat dynamics to hidden layer"""
    dim_out: Any = 64
    ksize: Any = 3
    strides: Any = 1

    @nn.compact
    def __call__(self, x, t):
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], -1)
        return nn.Conv(features=self.dim_out, kernel_size=self.ksize, strides=self.strides)(ttx)


# Define Model for Mnist example in Neural ODE
class ODEfunc(nn.Module):
    """ODE function which replace ResNet"""
    dim: Any = 64

    @nn.compact
    def __call__(self, inputs, t):
        x = inputs
        out = nn.GroupNorm(self.dim)(x)
        out = nn.relu(out)
        tt = jnp.ones_like(out[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, out], -1)
        out = nn.Conv(features=self.dim, kernel_size=(3, 3))(ttx)
        out = nn.GroupNorm(self.dim)(out)
        out = nn.relu(out)
        tt = jnp.ones_like(out[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, out], -1)
        out = nn.Conv(features=self.dim, kernel_size=(3, 3))(ttx)
        out = nn.GroupNorm(self.dim)(out)

        return out


class ODEBlock():
    """ODE block which contains odeint"""
    def __init__(self, odefunc):
        self.odefunc = odefunc
        self.integration_time = jnp.array([0., 1.])

    def __call__(self, x, params):
        ode_func_partial = partial(self.odefunc.apply, {'params': params})
        init_state, final_state = odeint(ode_func_partial, x, self.integration_time)
        return final_state


class SmallResNet(nn.Module):
    res_down1: Callable = ResDownBlock()
    res_down2: Callable = ResDownBlock()
    ode_func: Callable = ODEfunc(64)

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = self.res_down1(x)
        x = self.res_down2(x)
        _ = self.ode_func(x, 0)
        x = ODEBlock(self.ode_func)(x, self.ode_func.variables['params'])

        x = nn.GroupNorm(64)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (1, 1))

        x = x.reshape((x.shape[0], -1))     # flatten

        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)

        return x


# Define loss
def cross_entropy_loss(*, logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))


# Metric computation
def compute_metrics(*, logits, labels):
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
    cnn = SmallResNet()
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
        logits = vmap(SmallResNet()).apply({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics


# TODO Use vmap for speed up
# Training step w/ vmap
@jax.jit
def train_vmap_step(apply_fun, x_batch, y_batch, optimizer, state):
    def bathed_loss(params):
        def loss_fn(x, y):
            pred, updated_state = apply_fun({'params': params, **state},
                                            x, mutable= list(state.keys())
                                            )
            return cross_entropy_loss(pred, y)

        loss, updated_state = jax.vmap(
            loss_fn, out_axes=(0, None),
            axis_name='batch')(x_batch, y_batch)



# Evaluation step
@jax.jit
def eval_step(params, batch):
    logits = SmallResNet().apply({'params': params}, batch['image'])
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

    learning_rate = 0.0001

    state = create_train_state(init_rng, learning_rate)
    del init_rng  # Must not be used anymore.

    num_epochs = 10
    batch_size = 64

    for epoch in tqdm(range(1, num_epochs + 1)):
        rng, input_rng = jax.random.split(rng)
        state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
        test_loss, test_accuracy = eval_model(state.params, test_ds)
        print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, test_loss, test_accuracy * 100
        ))
