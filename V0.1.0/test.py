import pandas as pd
from datetime import datetime

import bls

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    # β = [ 4  2  0 12  9  0  0  1 15  3 12  0]
    
    start = datetime.now()
    olsb = bls.OLSBootstrap(data)
    print(f"Starting the bootstrap on {olsb.n_workers} processes "
          f"with blocks of {olsb.stack_sizes[0]} matrices.")

    olsb.fit()
    print("The MapReduce bootstrap took " 
          f"{datetime.now() - start} to complete.")
    
    print(olsb.coefs)