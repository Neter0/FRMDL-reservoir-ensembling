
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

    partitions = [2, 3, 4]

    long_dist1 = 4
    long_dist2 = 6

    score_TEPRE_24 = []
    score_TEPRE_36 = []
    
    for partition in partitions:
        score_TEPRE_24.append(simple_ensemble_lsm(in_conn_simple, num_res=1, num_partitions=partition, Nz=24))
        score_TEPRE_36.append(simple_ensemble_lsm(in_conn_simple, num_res=1, num_partitions=partition, Nz=36))

    # Plot results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for partition in partitions:
        print(f"Partition: {partition}, TEPRE (2400 neurons) Accuracy: {score_TEPRE_24[partitions.index(partition)]}")
        print(f"Partition: {partition}, TEPRE (3600 neurons) Accuracy: {score_TEPRE_36[partitions.index(partition)]}")

    plt.figure(figsize=(10, 6))
    plt.plot(partitions, score_TEPRE_24, 'd-', color='purple', label='TEPRE (2400 neurons)', linewidth=2, markersize=8)
    plt.plot(partitions, score_TEPRE_36, 'd-', color='orange', label='TEPRE (3600 neurons)', linewidth=2, markersize=8)

    plt.xlabel('Partitions', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('Partition Number Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'tepre_partition_study.png', dpi=300, bbox_inches='tight')
    plt.show()
