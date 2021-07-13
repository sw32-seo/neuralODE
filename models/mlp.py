import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.training import train_state
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import serialization
import optax


class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x


class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features - 1):
                x = nn.relu(x)
        return x


if __name__ == '__main__':
    key1, key2 = random.split(random.PRNGKey(0), 2)

    # Set problem dimensions
    nsamples = 20
    xdim = 10
    ydim = 5

    # Generate true W and b
    W = random.normal(key1, (xdim, ydim))
    b = random.normal(key2, (ydim,))
    true_params = freeze({'params': {'bias': b, 'kernel': W}})

    # Generate samples with additional noise
    ksample, knoise = random.split(key1)
    x_samples = random.normal(ksample, (nsamples, xdim))
    y_samples = jnp.dot(x_samples, W) + b
    y_samples += 0.1 * random.normal(knoise, (nsamples, ydim))  # Adding noise
    print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)

    key_init, subkey = random.split(ksample, 2)
    model = ExplicitMLP(features=[5])
    params = model.init(subkey, x_samples)

    def make_mse_func(x_batched, y_batched):
        def mse(params):
            # Define the squared loss for a single pair (x,y)
            def squared_error(x, y):
                pred = model.apply(params, x)
                return jnp.inner(y - pred, y - pred) / 2.0

            # We vectorize the previous to compute the average of the loss on all samples.
            return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

        return jax.jit(mse)  # And finally we jit the result.

    # Get the sampled loss
    loss = make_mse_func(x_samples, y_samples)

    lr = 0.3
    tx = optax.sgd(learning_rate=lr)
    opt_state = tx.init(params)
    loss_grad_fn = jax.value_and_grad(loss)

    for i in range(101):
        loss_val, grads = loss_grad_fn(params)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if i % 10 == 0:
            print('Loss step {}: '.format(i), loss_val)

    # Serializing the result
    bytes_output = serialization.to_bytes(params)
    dict_output = serialization.to_state_dict(params)
    print('Dict output')
    print(dict_output)
    print('Bytes output')
    print(bytes_output)

    # Restore the parameter from the saved one
    saved_params = serialization.from_bytes(params, bytes_output)
    print(loss(saved_params))
    print(loss(params))
