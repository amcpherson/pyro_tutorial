---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro.infer import Predictive
from torch.distributions import constraints

import pandas as pd
import matplotlib.pyplot as plt

```

```python

n_categories = 3

@config_enumerate
def model(data=None, n_data=None):
    assert (data is None) != (n_data is None)
    if n_data is None:
        n_data = data.shape[0]

    # Prior over categories
    probs = pyro.sample("probs", dist.Dirichlet(torch.ones([n_categories])))

    # Conditional distribution of the observed data
    means = torch.tensor([-2.0, 0.0, 2.0])  # Mean for each category
    scale = torch.tensor(0.25)  # Fixed standard deviation
    with pyro.plate("data", n_data):
        category = pyro.sample("category", dist.Categorical(probs))
        pyro.sample("obs", dist.Normal(means[category], scale), obs=data)

```

```python

pyro.set_rng_seed(2022)
pyro.clear_param_store()

conditioned_model = pyro.poutine.condition(
    model,
    data={
        'probs': torch.tensor([0.1, 0.5, 0.4]),
    })

model_trace = pyro.poutine.trace(conditioned_model)

samples = model_trace.get_trace(n_data=100)

print(samples.nodes['probs']['value'].detach())

data = samples.nodes['obs']['value'].detach()

pd.Series(data).hist(bins=100)

```


# Inference with an AutoNormal guide


```python

optim = pyro.optim.Adam({'lr': 0.01, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=2)
guide = AutoNormal(pyro.poutine.block(model, expose=['probs']))
svi = SVI(model, guide, optim, loss=elbo)

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

num_samples = 1000
predictive = Predictive(model, guide=guide, num_samples=num_samples)(n_data=100)
predictive['probs'].mean(axis=0)

```

```python

def custom_guide(data=None, n_data=None):
    probs_posterior = pyro.param(
        'probs_posterior',
        lambda: torch.ones(n_categories),
        # constraint=constraints.greater_than(0.5),
    )

    # Learnable parameter for the probabilities of the categorical distribution
    probs = pyro.sample('probs', dist.Dirichlet(probs_posterior))

```

```python

optim = pyro.optim.Adam({'lr': 0.01, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=2)
svi = SVI(model, custom_guide, optim, loss=elbo)

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

num_samples = 1000
predictive = Predictive(model, guide=custom_guide, num_samples=num_samples)(n_data=100)
predictive['probs'].mean(axis=0)

```

```python

```
