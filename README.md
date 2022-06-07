# Bootstrapped Least Squares

## A Parallelized Linear Regression Bootstrap Ideal for Experimental Data

This module provides a bootstrapped linear regression algorithm accelerated by parallelism.
The program efficiently distributes computations over your CPU's cores.
The result is a parallelized bootstrap algorithm that is delightfully quick, even on a laptop.

The test file, which is 9.6 Mb and gets subsampled 100,000 times, runs in less than 30 minutes on modern laptops, and in under 10 minutes on high-end laptops. These observations were made on MacOS. With a 32-core processor in an HPC, the computation takes under 5 minutes. Using a for-loop and statsmodels, the same computation would take about 4 hours on a laptop.

This project was originally created as a homework assignment for SBU AMS 598 with Dr. Song Wu. The goal was to use MapReduce to implement a linear regression bootstrap on data with 100,000 observations and 20 explanatory variables that ran in under 10 minutes on a SeaWulf cluster. Suffice it to say that we have overcome the task. In publishing this, I generalized the code a little and optimized it to run on a laptop. As such, I have stayed true to the assignment. I know that using joblib would make this run even faster, but multiprocessing is just fine. NOTE WELL, Prof. Wu nows that this repo is here and if you submit my code, you will know fear.

## Usage

Use on datasets with between 10,000 and 1,000,000 observations.

```python
import pandas as pd

import bls

# Have your data ready in an array-like; for example
data = pd.read_csv("data.csv")

# Use this condition, else problems with parallelization
# may occur.
if __name__ == "__main__":
    # initialize
    olsb = bls.OLSBootstrap(data)

    # Fit and optionally choose your confidence level.
    # The default is alpha=0.05 for basic 95% CIs with
    # bootstrap SE.
    olsb.fit(alpha=0.10)

    # If you want to construct more advanced confidence
    # bound estimators, you can access the coefficients.
    olsb.coefs
```

Please note, the default confidence intervals have form $\hat{\beta}_i^* \pm z_{\alpha/2}\hat{\sigma}_i$ where $\hat{\beta}_i^*$ and $\hat{\sigma}$ are respectively the mean and sample standard deviation of the subsampled coefficients. There is NO division by $\sqrt{n}$. Normally, we use $s$ to approximate the standard deviation of the sample, and divide by $\sqrt{n}$ to approximate the standard deviation of the distribution of sample means. Here, we are subsampling to approximate the distribution of sample coefficients, and directly taking the sample standard deviation of that coefficient distribution. As such, dividing $\hat{\sigma}_i$ by $\sqrt{n}$ would produce intervals with confidence far below the nominal value.

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
  - OLS computations are sped up by stacking matrices. `np.matmul` performs matrix multiplication on a stack of subsamples with one call; e.g. if `A.shape==[2, 3, 4, 5]` and `B.shape==[2, 3, 5, 6]` then `np.matmul(A, B).shape==[2, 3, 4, 6]`. This pushes the heavy loops necessary to repeatedly compute coefficients onto NumPy's highly-optimized, C-level implementation.
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
