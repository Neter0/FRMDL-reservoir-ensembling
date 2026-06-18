
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble_3_reservoir_long_short_dist_Gabor_lsm import long_short_Gabor_ensemble_lsm
from ensemble_lsm import simple_ensemble_lsm
from ensemble_3_reservoir_long_short_dist_lsm import long_short_ensemble_lsm

if __name__ == "__main__":

    in_conn_gabor = 0.05
    in_conn_simple = 0.15

    params_gabor = []
    params_simple = []

    scores_gabor = []
    scores_simple = []

    Nzs = [12, 24, 36]

    long_dist1 = 4
    long_dist2 = 6

    score_lsm = []
    score_lsm_ensemble = []
    score_MuLRE = []
    score_TEPRE = []
    
    for Nz in Nzs:
        score_lsm.append(simple_ensemble_lsm(in_conn_simple, num_res=1, Nz=Nz))
        score_lsm_ensemble.append(simple_ensemble_lsm(in_conn_simple, num_res=3, Nz=Nz))
        score_MuLRE.append(long_short_ensemble_lsm(in_conn_simple, long_dist1=long_dist1, long_dist2=long_dist2, Nz=Nz))
        score_TEPRE.append(simple_ensemble_lsm(in_conn_simple, num_res=1, num_partitions=3, Nz=Nz))

    # Plot results
    neurons = [Nz * 100 for Nz in Nzs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(neurons, score_lsm, 'o-', color='blue', label='LSM', linewidth=2, markersize=8)
    plt.plot(neurons, score_lsm_ensemble, 's-', color='green', label='Vanilla Ensemble', linewidth=2, markersize=8)
    plt.plot(neurons, score_MuLRE, '^-', color='red', label='MuLRE', linewidth=2, markersize=8)
    plt.plot(neurons, score_TEPRE, 'd-', color='purple', label='TEPRE', linewidth=2, markersize=8)
    
    plt.xlabel('Neurons', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('Ensemble Method Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
