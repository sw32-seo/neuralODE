# Neural ODE with Flax
***
This is the result of project ["Reproduce Neural ODE and SDE"][projectlink] in [HuggingFace Flax/JAX community week][comweeklink].

<code>main.py</code> will execute training of ResNet or OdeNet for MNIST dataset.

[projectlink]: https://discuss.huggingface.co/t/reproduce-neural-ode-and-neural-sde/7590

[comweeklink]: https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects#projects

## Dependency
***
###JAX and Flax
For JAX installation, please follow [here][jaxinstalllink].

or simply, type
```bash 
pip install jax jaxlib
```

For Flax installation,
```bash
pip install flax
```

[jaxinstalllink]: https://github.com/google/jax#installation


Tensorflow-datasets will download MNIST dataset to environment.

## How to run training
***
For (small) ResNet training,
```bash
python main.py --model=resnet --lr=1e-4 --n_epoch=20 --batch_size=64 
```

For Neural ODE training, 
```bash
python main.py --model=odenet --lr=1e-4 --n_epoch=20 --batch_size=64
```

