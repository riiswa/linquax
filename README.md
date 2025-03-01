# LinQuax - Online linear quadratic control in JAX.

## Overview

LinQuAx is a flexible, efficient library for implementing and experimenting with online linear quadratic control algorithms. Built on JAX for accelerated computation, LinQuAx provides implementations of state-of-the-art control ex[;ptation strategy for LQR including MED (Minimum Empirical Divergence), OFULQ (Optimism in the Face of Uncertainty), and Thompson Sampling (TS).

## Features

- Fast, vectorized implementations of modern control algorithms
- Built on JAX for GPU/TPU acceleration and automatic differentiation
- Integration with tensorboardX for experiment tracking
- Compatible with standard control environments
- Easy comparison between different control strategies

## Installation

### Prerequisites

Make sure you have Python 3.9+ installed.

### Install Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/linquax.git
cd linquax
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install controlgym --no-deps
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium
```

## Quick Start

```python
from linquax.environments import make_env
from linquax.controllers import MED
import jax
import matplotlib.pyplot as plt

# Setup environment
env = make_env("boeing747")

# Create a controller
controller = MED(env, warmup_steps=50, n_samples=128)

# Simulate
rng = jax.random.PRNGKey(42)
T = 500  # Time horizon
controller_state = controller.init(rng, T)
_, _, costs = env.simulate(rng, controller_state, controller.policy_fn, 
                           controller.on_completion_fn, T)

# Plot results
plt.figure()
plt.plot(jax.numpy.cumsum(costs))
plt.xlabel('Time steps')
plt.ylabel('Cumulative cost')
plt.show()
```

## Available Environments

LinQuax integrates with [controlgym](https://github.com/xiangyuan-zhang/controlgym) and provides several pre-configured environments:

- `boeing747`: Boeing 747 aircraft control model
- `inverted_pendulum`: Classic inverted pendulum stabilization problem
- `large_transient`: Environment for large transient system control
- `not_controllable`: Test environment for uncontrollable systems
- `uav`: Unmanned Aerial Vehicle control environment
- `unstable_laplacian`: Control of unstable systems with Laplacian dynamics
- `chained_integrator`: Multi-stage chained integrator system

These environments provide diverse challenges for testing control algorithms across different domains, from aircraft navigation to theoretical control problems.

## Implemented Controllers

- `MED`: Minimum Empirical Divergence
- `OFULQ`: Optimism in the Face of Uncertainty
- `TS`: Thompson Sampling
- More coming soon...
