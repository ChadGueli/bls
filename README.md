# Bootstrapped Least Squares

## A Parallelized Linear Regression Bootstrap Ideal For Experimental Data

This module provides a bootstrapped linear regression algorithm accelerated by parallelism.
The program efficiently distributes computations over your CPU's cores.
The result is a parallelized bootstrap algorithm that is delightfully quick, even on a modern laptop.

The test file, which is 1.2 Mb and gets subsampled 100,000 times, runs in less than 30 minutes on modern laptops, and under 10 minutes on high-end laptops. These observations were made on MacOS. With a 32-core processor in an HPC, the computation takes under 5 minutes. Using a for-loop and statsmodels, the same computation would take about 4 hours on a laptop.

## Usage

Use on datasets with between 10,000 and 1,000,000 observations.

```python
import pandas as pd

import bls

# have your data ready in an array-like; for example
data = pd.read_csv("data.csv")

# Use this condition, else problems with parallelization may occur.
if __name__ == "__main__":
    # initialize an OLSBootstrap instance on the data
    olsb = bls.OLSBootstrap(data)

    # fit and optionally choose your confidence level
    # default is alpha=0.05 for basic 95% CIs with bootstrap SE
    olsb.fit(alpha=0.10)

    # If you want to construct more advanced confidence
    # bound estimators, you can access the coefficients
    # and their unbiased bootstrap standard deviations
    # after fitting via the corresponding methods.
    olsb.coef
    olsb.std
```

## For Optimal Performance:
- It may be necessary to tweak the number of cores used.
  - Why?
    - When performing parallel computations, speed is a non-monotone function of the number of cores; that is, more cores doesn't guarantee more speed. 
  - By default, half of the cores are used. This heuristic has worked well in testing.

- If you want to do something else while running the algorithm, start the app before you start the bootstrap.
  - Why?
    - Often, the algorithm uses as much memory on your CPU as is available at initialization. If you run more jobs on the CPU after initialization, then they will cause memory overflow. While not a huge problem, it will slow your computation.
  - In particular, the memory used is the minimum of available memory, and the amount of memory used by our processes if total memory were evenly distributed among all processes.
  
## Advantages
- Vectorized
  - OLS multiplications are sped up with [Einstein summation](https://mathworld.wolfram.com/EinsteinSummation.html). np.einsum performs matmul on a stack of subsamples with one call. This pushes the heavy loops necessary to repeatedly compute coefficients onto NumPy's highly-optimized, C-level implementation.
- Scalable
  - Reliance on the MapReduce paradigm and Python's built-in multiprocessing module makes transitioning from your laptop to a cluster simple.

## Dependencies

NumPy >1.14, SciPy, Pandas, psutil

## Areas For Improvement (Collaboration?)
- Integration with the Scikit API?
  - I'm not entirely convinced this makes sense, if you have thoughts please share.
- Better algorithm for determining optimal CPU usage and block size.
- Test file that is more thorough.
- Add checkpointing and progress.
  - I believe this is challenging because of the pool executor.
- Add arg for number of bootstrap samples to take.
- Add implementations of multiple CIs
