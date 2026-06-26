
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble_3_reservoir_long_short_dist_Gabor_lsm import long_short_Gabor_ensemble_lsm
from ensemble_lsm import simple_ensemble_lsm
from ensemble_3_reservoir_long_short_dist_lsm import long_short_ensemble_lsm

if __name__ == "__main__":

    in_conn_simple = 0.15

    results = [0.7221731448763251, 0.7592756183745583, 0.7075971731448764, 0.7367491166077739, 0.7217314487632509]
    plt.boxplot(results)
    plt.scatter(1, 0.778, color="green", zorder=3)
    plt.ylabel("Score")
    plt.show()
    
    print(np.mean(results))
    print(np.std(results))
    print(np.min(results))
    print(np.max(results))
    for i in range(5):
        result = simple_ensemble_lsm(in_conn_simple, num_res=1, num_partitions=6, Nz=30, dataset = 'SHD', tauV=40.0, tauI=20.0)
        results.append(result)

    print(results)
    print(np.mean(results))
    print(np.std(results))
    print(np.min(results))
    print(np.max(results))
