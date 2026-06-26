
import json
from json import JSONDecodeError
import random
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble_lsm import simple_ensemble_lsm
from ensemble_3_reservoir_long_short_dist_lsm import long_short_ensemble_lsm

SCORES_PATH = Path(__file__).parent / "outputs" / "IF_comparison_scores.json"

global_seeds = [42, 123, 456]  # List of seeds for reproducibility

def configured_models(seeds = global_seeds[0], mulre_period_ranges = False, tepre_period_ranges = False):
    # Take the simple (non-seeded, default resonance range) configurations, extend them with each given seed 
    # and with the specified resonance period ranges for MuLRE and TEPRE models.
    # This can be easily set up to recreate all of my results, as follows:
    # For resonance period study, let seeds default to 42, uncomment the mulre_period_ranges 
    # and tepre_period_ranges lists, and run the script.
    # For the general model comparison (and don't want the resonance study), 
    # set mulre_period_ranges and tepre_period_ranges to the best 
    # options ((30, 300) for MuLRE and (100, 100) for TEPRE), and run the script with seeds = global_seeds.
    # Keep in mind every run persists the scores, without overriting any of the old ones!

    
    raw_configs = [model_config.copy() for model_config in MODEL_CONFIGS]
    extended_configs = []

    refs = raw_configs
    for ref in refs:
        for seed in seeds:
            temp = ref.copy()
            temp['seed'] = seed
            extended_configs.append(temp)
    
    if mulre_period_ranges:
        mulre_configs = [config for config in extended_configs if "MuLRE RF" in config["name"]]
        for mulre_config in mulre_configs:
            mulre_config["T_min"] = mulre_period_ranges[0][0]
            mulre_config["T_max"] = mulre_period_ranges[0][1]
            for T_min, T_max in mulre_period_ranges[1:]: 
                model_config = mulre_config.copy()
                model_config["T_min"] = T_min
                model_config["T_max"] = T_max
                extended_configs.append(model_config)

    if tepre_period_ranges:
        for tepre_config in [config for config in extended_configs if "TEPRE RF" in config["name"]]:
            tepre_config["T_min"] = tepre_period_ranges[0][0]
            tepre_config["T_max"] = tepre_period_ranges[0][1]
            for T_min, T_max in tepre_period_ranges[1:]: 
                model_config = tepre_config.copy()
                model_config["T_min"] = T_min
                model_config["T_max"] = T_max
                extended_configs.append(model_config)

    return extended_configs


def persist_score(model_name, Nz, score, params=None, scores_path=SCORES_PATH):
    # Take the computes score and append it to the scores.json file under the correct model

    scores_path.parent.mkdir(parents=True, exist_ok=True)

    if not scores_path.exists() or scores_path.stat().st_size == 0:
        scores = {}
    else:
        try:
            with scores_path.open("r", encoding="utf-8") as f:
                scores = json.load(f)
        except JSONDecodeError:
            backup_path = scores_path.with_suffix(
                f".invalid-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
            )
            scores_path.replace(backup_path)
            print(f"Invalid scores JSON moved to {backup_path}")
            scores = {}

    Nz_key = str(Nz)
    scores.setdefault(model_name, {}).setdefault(Nz_key, []).append({
        "score": float(score),
        "Nz": Nz,
        "neurons": Nz * 100,
        "params": params,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })

    with scores_path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")


def run_model(model_config, Nz):
    # Take the model_config and run the correct predefined script
    # Biases are set to false for maintaining reliable reproductivity: implementation 
    # generates random linear NNs and overwrides their weights, but NOT the biases, 
    # so I set them to False

    neuron_dynamics = model_config.get("neuron_dynamics", "LIF")

    if model_config["model"] == "MuLRE":
        return long_short_ensemble_lsm(
            model_config["in_conn"],
            long_dist1=model_config["long_dist1"],
            long_dist2=model_config["long_dist2"],
            Nz=Nz,
            beta_LIF=model_config.get("beta", 0.99),
            neuron_dynamics=neuron_dynamics,
            T_min=model_config.get("T_min", 3),
            T_max=model_config.get("T_max", 100),
            seed=model_config.get("seed", global_seeds[0]),
            bias=False
        )

    if model_config["model"] == "TEPRE":
        return simple_ensemble_lsm(
            model_config["in_conn"],
            num_res=model_config["num_res"],
            num_partitions=model_config["num_partitions"],
            Nz=Nz,
            beta_LIF=model_config.get("beta", 0.99),
            neuron_dynamics=neuron_dynamics,
            T_min=model_config.get("T_min", 3),
            T_max=model_config.get("T_max", 100),
            seed=model_config.get("seed", global_seeds[0]),
            bias=False
        )

    raise ValueError(f"Unknown model type: {model_config['model']}")


def persisted_params(model_config):
    return {key: value for key, value in model_config.items() if key != "name"}


def plot_run_results(results, Nz_values):
    all_neurons = [Nz * 100 for Nz in Nz_values]

    plt.figure(figsize=(10, 6))
    for model_name, scores in results.items():
        neurons = all_neurons[:len(scores)]
        if not scores:
            continue
        plt.plot(neurons, scores, marker="o", linewidth=2, markersize=8, label=model_name)

    plt.xlabel("Neurons", fontsize=12)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.title("Ensemble Method Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


NZ_VALUES = [12]


# Set simple configs: no period ranges, no seeds: just the model and the wanted neuron dynamics
MODEL_CONFIGS = [
    {
        "name": "MuLRE LIF",
        "model": "MuLRE",
        "in_conn": 0.125,
        "long_dist1": 4,
        "long_dist2": 6,
        "num_res": 1,
        "beta": 0.995,
        "num_partitions": 3,
        "neuron_dynamics": "LIF",
    },
    {
        "name": "TEPRE LIF",
        "model": "TEPRE",
        "in_conn": 0.15,
        "num_res": 1,
        "beta": 0.98,
        "num_partitions": 3,
        "neuron_dynamics": "LIF",
    },
]

mulre_period_ranges = [
    #(3, 100),
    #(5, 150),
    #(10, 150),
    #(10, 300),
    #(20, 300),
    (30, 300),
    #(100, 300),
    #(175,300),
    #(300,300)
]

tepre_period_ranges = [
    #(5, 75),
    #(10, 75),
    #(10, 100),
    #(20, 100),
    #(30, 100)
    #(75,100),
    (100,100)
]


if __name__ == "__main__":
    model_configs = configured_models(seeds = global_seeds, 
                          mulre_period_ranges = mulre_period_ranges, 
                          tepre_period_ranges = tepre_period_ranges)
    results = {model_config["name"]: [] for model_config in model_configs}

    for model_config in model_configs:
        print(model_config)
    
    
    for Nz in NZ_VALUES:
        for model_config in model_configs:
            model_name = model_config["name"]
            score = run_model(model_config, Nz)
            results[model_name].append(score)
            persist_score(model_name, 
                          Nz, 
                          score, 
                          persisted_params(model_config))
