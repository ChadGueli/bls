import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import norm
import psutil

class OLSBootstrap(object):

    def __init__(self, data, y_col=0, use_intercept=True,
                 n_workers=None, entropy=None):
        """ An implementation of bootstrapped ordinary least
        squares regression. Takes advantage of the MapReduce
        paradigm to parallelize computations.

        Args:
            data: The data to perform regression on. Should
                contain the y variable.
            y_col: Which column in the data to use as the
                response variable.
            use_intercept: Whether to include an intercept
                term. Intercept is included if True.
            n_workers: The number of processes to
                parallelize the implementation on. Defaults
                to half of the available CPU cores.
            entropy: The entropy used to initialize the
            random number generators.
        """
        self.entropy = entropy
        self._β = None
        self._σ = None
        self.data = np.asfarray(data)
        
        if isinstance(data, pd.DataFrame):
            self.cols = np.delete(data.columns.to_numpy(), y_col)
            self.cols = np.insert(self.cols, 0, "Inter")
        else:
            count = self.data.shape[1]
            cols = (f"β{i}" for i in range(count))
            self.cols = np.fromiter(cols, dtype='U8', count=count)

        if y_col != 0:
            y_col += 1
            self.data[:, :y_col] = np.roll(self.data[:, :y_col], 1, axis=1)

        if use_intercept == True:
            self.data = np.insert(self.data, 1, 1., axis=1)
        else:
            self.cols = np.delete(self.cols, 0)

        # process size determination
        cpus = psutil.cpu_count()
        if n_workers is None:
            self.n_workers = cpus // 2
        else:
            self.n_workers = n_workers

        # mem size determination
        mem = psutil.virtual_memory()
        mem_use = min(mem.available, (mem.total * self.n_workers) // cpus)
        max_size = mem_use // (self.data.nbytes * self.n_workers)

        n2 = max_size - (self.data.shape[0] % max_size)
        n1 = (self.data.shape[0] // max_size) + 1 - n2
        self.stack_sizes = [max_size]*n1 + [max_size-1]*n2

    def _subsample_regress(self, stack_size, seq):
        rng = np.random.default_rng(seq)
        chunk = rng.choice(self.data, size=(stack_size, self.data.shape[0]))

        # regress using vectorized least squares
        ys = chunk[..., 0]
        Xs = chunk[..., 1:]
        Xᵀys = np.einsum("ijk,ij->ik", Xs, ys)
        invXᵀXs = np.linalg.pinv(np.einsum("ijk,ijl->ikl", Xs, Xs),
                                hermitian=True)
        return np.einsum("ijk,ij->ik", invXᵀXs, Xᵀys)
        
    def _get_stats(self, key, βs):
        return key, np.mean(βs), np.std(βs)

    def fit(self, alpha=0.05):
        """Performs the actual bootstrapping, and produces
        100(1-α)% confidence intervals.
        """
        α = alpha

        # SeedSequence is used to produce parallel rngs
        # that are independent with high probability.
        seed_seq = np.random.SeedSequence(entropy=self.entropy,
                                          pool_size=8)
        seqs = seed_seq.spawn(len(self.stack_sizes))
        with mp.Pool(processes=self.n_workers) as pool:
            # map
            β_stacks = pool.starmap(
                self._subsample_regress,
                zip(self.stack_sizes, seqs),
                len(self.stack_sizes) // self.n_workers)
            βs = np.concatenate(β_stacks)

            # reduce (column index is key)
            stats = np.array(
                pool.starmap(self._get_stats, enumerate(βs.T)),
                dtype=[("key", "i4"), ("β", "f8"), ("σ", "f8")])

        self._β = stats["β"]
        self._σ = stats["σ"]

        z = norm.ppf(1-α/2) / np.sqrt(self.data.shape[0])
        per = 100*(1-α)
        for key, β, σ in stats:
            print(f"{self.cols[key]} has bootstrap coef {β:.8f} with "
                  f"bootstrapped {per}% CI ({β-z*σ:.8f}, {β+z*σ:.8f}).")
    
    @property
    def coef(self):
        """The bootstrap mean of the coefficients."""
        return self._β

    @property
    def std(self):
        """The bootstrap unbiased standard deviation of the coefficients."""
        return self._σ
    
    
    

    


    
    
    
    
    
    