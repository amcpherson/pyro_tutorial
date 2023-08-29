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
from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete, Trace_ELBO
import torch
from torch.distributions import constraints

import seaborn as sns
import matplotlib.pyplot as plt


```

```python
d = dist.Bernoulli(0.5 * torch.ones(3,4)).to_event(1)
assert d.batch_shape == (3,) and d.event_shape == (4,)

```

```python

d = dist.MultivariateNormal(torch.zeros(3), torch.eye(3, 3))
d.batch_shape, d.event_shape

```


# Model

A poisson model with a somewhat complex prior on expected counts.


```python

def model():
    A = pyro.sample('A', dist.Normal(0, 1))
    B = pyro.sample('B', dist.LogNormal(0, 1))
    C = pyro.sample('C', dist.LogNormal(A, B))
    X = pyro.sample('X', dist.Poisson(C))

```

```python

pyro.render_model(model, model_args=())

```


# Samples

Generate samples using poutine.trace


```python

pyro.set_rng_seed(2022)
pyro.clear_param_store()

model_trace = pyro.poutine.trace(model)

samples = model_trace.get_trace()

for a in ('A', 'B', 'C', 'X'):
    print(a, samples.nodes[a]['value'].detach().numpy())

```


# Model 2

Modify the model to generate multiple independent observations given the same prior over expected counts using pyro.plate.


```python

def model(n_data):
    A = pyro.sample('A', dist.Normal(0, 1))
    B = pyro.sample('B', dist.LogNormal(0, 1))
    C = pyro.sample('C', dist.LogNormal(A, B))
    with pyro.plate('N', n_data):
        X = pyro.sample('X', dist.Poisson(C))

```

```python

pyro.render_model(model, model_kwargs={'n_data': 1})

```

```python

pyro.set_rng_seed(2022)
pyro.clear_param_store()

model_trace = pyro.poutine.trace(model)

samples = model_trace.get_trace(n_data=100)

for a in ('A', 'B', 'C'):
    print(a, samples.nodes[a]['value'].detach().numpy())

sns.countplot(x=samples.nodes['X']['value'].numpy(), color='0.5')

```


# Conditioning using poutine

We can use poutine to modify our model, conditioning on any of the random variables, using poutine.condition.


```python

pyro.set_rng_seed(2022)
pyro.clear_param_store()

conditioned_model = pyro.poutine.condition(
    model,
    data={
        'A': torch.tensor([0.1]),
    })

model_trace = pyro.poutine.trace(conditioned_model)

samples = model_trace.get_trace(n_data=100)

for a in ('A', 'B', 'C'):
    print(a, samples.nodes[a]['value'].detach().numpy())

sns.countplot(x=samples.nodes['X']['value'].detach().numpy(), color='0.5')

```


Lets use data sampled from the model to explore inference techniques


```python

data = samples.nodes['X']['value'].detach()

```


# Model 3

Modify the model to be useful for sampling and inference by taking either number of samples to generate or the observed data.


```python

def model(data=None, n_data=None):
    assert (data is None) != (n_data is None)
    if n_data is None:
        n_data = data.shape[0]
    A = pyro.sample('A', dist.Normal(0, 1))
    B = pyro.sample('B', dist.LogNormal(0, 1))
    C = pyro.sample('C', dist.LogNormal(A, B))
    with pyro.plate('N', n_data):
        X = pyro.sample('X', dist.Poisson(C), obs=data)

```


# Inference

Use the Adam optimizer, Trace_ELBO based SVI and an AutoNormal guide to infer A, B, C from our data


```python

optim = pyro.optim.Adam({'lr': 0.01, 'betas': [0.8, 0.99]})
elbo = Trace_ELBO(max_plate_nesting=2)
guide = AutoNormal(pyro.poutine.block(model, expose=['A', 'B', 'C']))
svi = SVI(model, guide, optim, loss=elbo)

```

```python

pyro.clear_param_store()

losses = []
for i in range(300):
    loss = svi.step(data=data)
    losses.append(loss)
    print('.' if i % 80 else '\n', end='')

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


# Model evaluation

We can evaluate our model by generating samples from the posterior and comparing with the known hidden variables and observed data.  Use pyro.infer.Predictive to generate 1000 samples of 100 data points each, summarize, and compare.


```python

import pandas as pd
from pyro.infer import Predictive


num_samples = 1000
predictive = Predictive(model, guide=guide, num_samples=num_samples)(n_data=100)

svi_samples = pd.DataFrame({k: v.reshape(num_samples).detach().cpu().numpy()
               for k, v in predictive.items()
               if k != 'X'})
svi_samples.describe().T

```

```python

svi_samples = pd.DataFrame([pd.Series(a).value_counts() for a in predictive['X']])
svi_samples = svi_samples.stack().reset_index()
svi_samples.columns = ['sample', 'X', 'count']

sns.barplot(x='X', y='count', data=svi_samples, color='r', alpha=0.5, label='posterior')
sns.countplot(x=data.detach().numpy(), color='b', alpha=0.5, label='data')
plt.legend()
sns.despine()

```


# Writing a custom guide

Below is a very simple guide using a mean field approach.  Each latent variable is modeled by an independent distribution.  Marginal distributions over the latent variables are parameterized by pyro params.  During inference, the params of each marginal are inferred using SVI.


```python

def custom_guide(data=None, n_data=None):
    A_loc = pyro.param('A_loc', lambda: torch.tensor(0.))
    A_scale = pyro.param('A_scale', lambda: torch.tensor(1.), constraint=constraints.positive)
    A = pyro.sample('A', dist.Normal(A_loc, A_scale))
    
    B_loc = pyro.param('B_loc', lambda: torch.tensor(0.))
    B_scale = pyro.param('B_scale', lambda: torch.tensor(1.), constraint=constraints.positive)
    B = pyro.sample('B', dist.LogNormal(B_loc, B_scale))

    C_loc = pyro.param('C_loc', lambda: torch.tensor(0.))
    C_scale = pyro.param('C_scale', lambda: torch.tensor(1.), constraint=constraints.positive)
    C = pyro.sample('C', dist.LogNormal(C_loc, C_scale))

```

```python

optim = pyro.optim.Adam({'lr': 0.01, 'betas': [0.8, 0.99]})
elbo = Trace_ELBO(max_plate_nesting=2)
svi = SVI(model, custom_guide, optim, loss=elbo)

```

```python

pyro.clear_param_store()

losses = []
for i in range(300):
    loss = svi.step(data=data)
    losses.append(loss)
    print('.' if i % 80 else '\n', end='')

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

1. Evaluate the performance of the custom guide
2. Improve the simple mean field guide suggested above.


```python

```
