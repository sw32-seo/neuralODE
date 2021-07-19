# Neural ODE with Flax
This is the result of project ["Reproduce Neural ODE and SDE"][projectlink] in [HuggingFace Flax/JAX community week][comweeklink].

<code>main.py</code> will execute training of ResNet or OdeNet for MNIST dataset.

[projectlink]: https://discuss.huggingface.co/t/reproduce-neural-ode-and-neural-sde/7590

[comweeklink]: https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects#projects

## Dependency

### JAX and Flax

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

For (small) ResNet training,
```bash
python main.py --model=resnet --lr=1e-4 --n_epoch=20 --batch_size=64 
```

For Neural ODE training, 
```bash
python main.py --model=odenet --lr=1e-4 --n_epoch=20 --batch_size=64
```

For Continuous Normalizing Flow,
```bash
python main.py --model=cnf --sample_dataset=circles
```

# Sample Results

![cnf-viz](https://user-images.githubusercontent.com/72425253/126116823-a014f13a-1171-4309-898f-0b6aedd84649.gif)
![cnf-viz](https://user-images.githubusercontent.com/72425253/126117205-fa68c16b-fba1-48a0-a965-3ac6cb5e201c.gif)

