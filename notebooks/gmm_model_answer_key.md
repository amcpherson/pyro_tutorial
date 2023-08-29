---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python


import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete, Trace_ELBO
import torch

import seaborn as sns
import matplotlib.pyplot as plt

```


# Model

Below is a simple 1D gaussian mixture model with component specific means and global variance parameters.  We have used the @config_enumerate decorator to tell pyro to enumerate over, or marginalize, the discrete random variables (assignment).


```python

K = 2

@config_enumerate
def model(data=None, n_data=None):
    assert (data is None) != (n_data is None)
    if n_data is None:
        n_data = data.shape[0]

    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
    scale = pyro.sample('scale', dist.LogNormal(0., 0.5))

    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(0., 4.))

    with pyro.plate('data', n_data):
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

```


# Simulate data

Generate 100 data points from the model.  Print the simulated parameters and show a histogram of the simulated data.


```python

pyro.set_rng_seed(2022)
pyro.clear_param_store()

model_trace = pyro.poutine.trace(model)

samples = model_trace.get_trace(n_data=100)

for a in ('weights', 'scale', 'locs'):
    print(a, samples.nodes[a]['value'].detach().numpy())

sns.histplot(x=samples.nodes['obs']['value'].detach().numpy(), color='0.5')

data = samples.nodes['obs']['value'].detach()

```


# Inference

Use the TraceEnum_ELBO implementation of the elbo calculation that enumerates and marginalizes over the discrete hidden variables.  Use the AutoDelta guide for the remaining variables, which will use a delta function to approximate the posteriors of hidden variables producing point estimates of each variable.


```python

optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=1)

pyro.set_rng_seed(1)
pyro.clear_param_store()

global_guide = AutoDelta(
    pyro.poutine.block(model, expose=['weights', 'locs', 'scale']))

svi = SVI(model, global_guide, optim, loss=elbo)

```

```python

losses = []
for i in range(500):
    loss = svi.step(data)
    losses.append(loss)
    print('.' if i % 100 else '\n', end='')

plt.figure(figsize=(10,3), dpi=100).set_facecolor('white')
plt.plot(losses)
plt.xlabel('iters')
plt.ylabel('loss')
plt.yscale('log')
plt.title('Convergence of SVI');

```

```python

for k in pyro.get_param_store():
    print(k, pyro.get_param_store()[k].detach())

```


# Additional exercises

1. The inferred parameters were innaccurate, try generating a better initialization as described in the [gmm tutorial](https://pyro.ai/examples/gmm.html#Training-a-MAP-estimator)
2. Extend the GMM to a 2D model
3. Modify from global variance to component specific variance
4. Modify from global variance to dimension specific variance




# Answer key from here:


```python

def init_loc_fn(site):
    if site["name"] == "weights":
        # Initialize weights to uniform.
        return torch.ones(K) / K
    if site["name"] == "scale":
        return (data.var() / 2).sqrt()
    if site["name"] == "locs":
        return data[torch.multinomial(torch.ones(len(data)) / len(data), K)]
    raise ValueError(site["name"])

def initialize(seed):
    global global_guide, svi
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    global_guide = AutoDelta(
        pyro.poutine.block(model, expose=['weights', 'locs', 'scale']),
        init_loc_fn=init_loc_fn)
    svi = SVI(model, global_guide, optim, loss=elbo)
    return svi.loss(model, global_guide, data)

# Choose the best among 100 random initializations.
loss, seed = min((initialize(seed), seed) for seed in range(100))
initialize(seed)
print('seed = {}, initial_loss = {}'.format(seed, loss))

```

```python

losses = []
for i in range(500):
    loss = svi.step(data)
    losses.append(loss)
    print('.' if i % 100 else '\n', end='')

```

```python

plt.figure(figsize=(10,3), dpi=100).set_facecolor('white')
plt.plot(losses)
plt.xlabel('iters')
plt.ylabel('loss')
plt.yscale('log')
plt.title('Convergence of SVI');

```

```python

for k in pyro.get_param_store():
    print(k, pyro.get_param_store()[k].detach())

```


# 2D GMM


```python

K = 3
D = 2

@config_enumerate
def model(data=None, n_data=None):
    assert (data is None) != (n_data is None)
    if n_data is None:
        n_data = data.shape[0]

    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))

    scale = pyro.sample('scale', dist.LogNormal(0., 0.5))

    with pyro.plate('dims', D):
        with pyro.plate('components', K):
            locs = pyro.sample('locs', dist.Normal(0., 4.))

    with pyro.plate('data', n_data):
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        obs = pyro.sample('obs', dist.Normal(locs[assignment], scale).to_event(1), obs=data)

```

```python

trace = poutine.trace(model).get_trace(n_data=11)
trace.compute_log_prob()
print(trace.format_shapes())

```


# 2D GMM with component specific variance


```python

K = 3
D = 2

@config_enumerate
def model(data=None, n_data=None):
    assert (data is None) != (n_data is None)
    if n_data is None:
        n_data = data.shape[0]

    weights = pyro.sample('weights', dist.Dirichlet(1. * torch.ones(K)))

    dims = pyro.plate('dims', D, dim=-1)
    components = pyro.plate('components', K, dim=-2)

    with components:
        scale = pyro.sample('scale', dist.LogNormal(0., 0.5))

    with dims:
        with components:
            locs = pyro.sample('locs', dist.Normal(0., 4.))

    with pyro.plate('data', n_data):
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        obs = pyro.sample('obs', dist.Normal(locs[assignment], scale[assignment]).to_event(1), obs=data)

```


# 2D GMM with component specific variance (alternate)


```python

K = 3
D = 2

@config_enumerate
def model(data=None, n_data=None):
    assert (data is None) != (n_data is None)
    if n_data is None:
        n_data = data.shape[0]

    weights = pyro.sample('weights', dist.Dirichlet(1. * torch.ones(K)))

    dims = pyro.plate('dims', D, dim=-2)
    components = pyro.plate('components', K, dim=-1)

    with components:
        scale = pyro.sample('scale', dist.LogNormal(0., 0.5))
        with dims:
            locs = pyro.sample('locs', dist.Normal(0., 4.))

    with pyro.plate('data', n_data):
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        obs = pyro.sample('obs', dist.Normal(locs.swapaxes(-1, -2)[assignment], scale[assignment, None]).to_event(1), obs=data)

```


# 2D GMM with dimension specific variance


```python

K = 3
D = 2

@config_enumerate
def model(data=None, n_data=None):
    assert (data is None) != (n_data is None)
    if n_data is None:
        n_data = data.shape[0]

    weights = pyro.sample('weights', dist.Dirichlet(1. * torch.ones(K)))

    dims = pyro.plate('dims', D, dim=-2)
    components = pyro.plate('components', K, dim=-1)

    with dims:
        scale = pyro.sample('scale', dist.LogNormal(0., 0.5))

    with components:
        with dims:
            locs = pyro.sample('locs', dist.Normal(0., 4.))

    with pyro.plate('data', n_data):
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        obs = pyro.sample('obs', dist.Normal(locs.swapaxes(-1, -2)[assignment], scale.swapaxes(-1, -2)).to_event(1), obs=data)

```

```python

trace = poutine.trace(model).get_trace(n_data=11)
trace.compute_log_prob()
print(trace.format_shapes())

```

```python

trace = poutine.trace(poutine.enum(model, first_available_dim=-2)).get_trace(n_data=11)
trace.compute_log_prob()
print(trace.format_shapes())

```

```python

pyro.set_rng_seed(59)
pyro.clear_param_store()

model_trace = pyro.poutine.trace(model)

samples = model_trace.get_trace(n_data=100)

for a in ('weights', 'scale', 'locs'):
    print(a, samples.nodes[a]['value'].detach().numpy())

data = samples.nodes['obs']['value'].detach()

sns.scatterplot(x=data[:, 0], y=data[:, 1])
sns.despine(trim=True)

```

```python

optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=2)

pyro.set_rng_seed(1)
pyro.clear_param_store()

global_guide = AutoDelta(
    pyro.poutine.block(model, expose=['weights', 'locs', 'scale']))

svi = SVI(model, global_guide, optim, loss=elbo)

losses = []
for i in range(100):
    loss = svi.step(data)
    losses.append(loss)
    print('.' if i % 100 else '\n', end='')

```

```python

for k in pyro.get_param_store():
    print(k, pyro.get_param_store()[k].detach())

```

```python

```
