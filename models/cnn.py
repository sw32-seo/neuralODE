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
class ResBlock(nn.Module):
    """Single Resblock w/o downsample"""

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        f_x = nn.relu(nn.GroupNorm(64)(x))
        f_x = nn.Conv(features=64, kernel_size=(3, 3))(f_x)
        f_x = nn.relu(nn.GroupNorm(64)(f_x))
        f_x = nn.Conv(features=64, kernel_size=(3, 3))(f_x)
        x = f_x + x
        return x

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


# Define Model for Mnist example in Neural ODE
class SmallResNet(nn.Module):
    res_down1: Callable = ResDownBlock()
    res_down2: Callable = ResDownBlock()
    resblock1: Callable = ResBlock()
    resblock2: Callable = ResBlock()
    resblock3: Callable = ResBlock()
    resblock4: Callable = ResBlock()
    resblock5: Callable = ResBlock()
    resblock6: Callable = ResBlock()

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = self.res_down1(x)
        x = self.res_down2(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)

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
        logits = SmallResNet().apply({'params': params}, batch['image'])
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
    for perm in perms:
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

    num_epochs = 40
    batch_size = 128

    for epoch in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
        test_loss, test_accuracy = eval_model(state.params, test_ds)
        print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, test_loss, test_accuracy * 100
        ))
