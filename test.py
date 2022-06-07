import pandas as pd
from datetime import datetime

import bls

def test():
    data = pd.read_csv("data.csv")
    # β = [ 4  2  0 12  9  0  0  1 15  3 12  0]
    
    t0 = datetime.now()
    olsb = bls.OLSBootstrap(data)
    print(f"Starting the bootstrap on {olsb.n_workers} processes "
          f"with blocks of {olsb.stack_sizes[0]} matrices.")

    olsb.fit()
    t1 = datetime.now()
    print(f"The MapReduce bootstrap took {t1-t0} to complete.")
    
    print(olsb.coefs)

if __name__ == "__main__":
    test()