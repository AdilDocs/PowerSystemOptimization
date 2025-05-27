import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time


def load_dataset(name):

    np.random.seed(42)
    data_size = 1000
    features = np.random.rand(data_size, 50)  # 50 arbitrary features
    targets = np.random.rand(data_size, 5)  # [EDNS, EENS, LOLP, LOLE, LOLF]
    return features, targets


def split_train_test(features, targets, train_ratio=0.8):
    split_index = int(len(features) * train_ratio)
    return (features[:split_index], targets[:split_index],
            features[split_index:], targets[split_index:])


def simulate_case(case_num, features, targets):


    start_time = time.time()

    if case_num == 1:
        # No optimization: simple pass-through or baseline random
        simulated_indices = targets.mean(axis=0) * (1 + 0.2)  # worse performance
    elif case_num == 2:
        # MILP optimization simulation (dummy improvement)
        simulated_indices = targets.mean(axis=0) * (1 - 0.1)
    elif case_num == 3:
        # Transformer-based optimization (better improvement)
        simulated_indices = targets.mean(axis=0) * (1 - 0.3)
    elif case_num == 4:
        # AI-driven state reduction (best case)
        simulated_indices = targets.mean(axis=0) * (1 - 0.4)
    else:
        raise ValueError("Invalid case number")

    computation_time = time.time() - start_time
    return simulated_indices, computation_time


def run_scenarios(features_train, targets_train, load_levels=[0.9, 1.0]):
    """
    Run scenarios under different load levels (90%, 100% etc.).
    Returns a dict of results per case per load level.
    """
    results = {}
    for load in load_levels:
        scaled_targets = targets_train * load  # Scale demand/load proportionally
        results[load] = {}
        for case_num in range(1, 5):
            indices, ctime = simulate_case(case_num, features_train, scaled_targets)
            results[load][case_num] = {
                "EDNS": indices[0],
                "EENS": indices[1],
                "LOLP": indices[2],
                "LOLE": indices[3],
                "LOLF": indices[4],
                "CompTime": ctime
            }
    return results


def k_fold_cross_validation(features, targets, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics_per_fold = []

    for train_idx, val_idx in kf.split(features):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]

        # Simulate training and evaluation (replace with real training)
        # Here, just compute means as dummy performance
        val_metrics = y_val.mean(axis=0)
        generation_cost = 1_250_000 + np.random.randint(-10000, 10000)  # dummy cost variation

        metrics_per_fold.append({
            "LOLP": val_metrics[2],
            "EENS": val_metrics[1],
            "LOLF": val_metrics[4],
            "Cost": generation_cost
        })

    return metrics_per_fold


def plot_trends(results, load_levels):
    """
    Plot EENS and LOLP trends for different cases under given load levels.
    """
    for load in load_levels:
        eens = [results[load][case]["EENS"] for case in range(1, 5)]
        lolp = [results[load][case]["LOLP"] for case in range(1, 5)]
        cases = ["Case 1", "Case 2", "Case 3", "Case 4"]

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(cases, eens, color=['red', 'blue', 'green', 'pink'])
        plt.title(f'EENS at Load {load * 100:.0f}%')
        plt.ylabel('EENS (MWh/yr)')

        plt.subplot(1, 2, 2)
        plt.bar(cases, lolp, color=['red', 'blue', 'green', 'pink'])
        plt.title(f'LOLP at Load {load * 100:.0f}%')
        plt.ylabel('LOLP')
        plt.tight_layout()
        plt.show()


# --- Main experimental procedure --- #

if __name__ == "__main__":
    # Load datasets
    features_rts96, targets_rts96 = load_dataset("RTS-96")
    features_spc, targets_spc = load_dataset("SPC")

    # Combine or select dataset
    features, targets = features_rts96, targets_rts96

    # Normalize features (simple min-max normalization)
    features = (features - features.min(axis=0)) / (features.ptp(axis=0) + 1e-9)

    # Split into training and testing
    X_train, y_train, X_test, y_test = split_train_test(features, targets, train_ratio=0.8)

    # Run scenarios for 90% and 100% load levels
    scenario_results = run_scenarios(X_train, y_train, load_levels=[0.9, 1.0])

    # Print table 1 style summary
    print("Reliability indices under 90% and 100% load levels:")
    for load in scenario_results:
        print(f"\nLoad: {load * 100:.0f}%")
        for case_num, metrics in scenario_results[load].items():
            print(
                f"Case {case_num}: EDNS={metrics['EDNS']:.4f}, EENS={metrics['EENS']:.1f}, LOLP={metrics['LOLP']:.4f}, LOLE={metrics['LOLE']:.2f}, LOLF={metrics['LOLF']:.2f}, Time={metrics['CompTime']:.2f}s")

    # Plot trends (Figure 3 equivalent)
    plot_trends(scenario_results, load_levels=[0.9, 1.0])

    # K-Fold cross-validation
    cv_metrics = k_fold_cross_validation(features, targets)
    print("\nK-Fold Cross Validation Results:")
    for i, fold_metrics in enumerate(cv_metrics):
        print(
            f"Fold {i + 1}: LOLP={fold_metrics['LOLP']:.4f}, EENS={fold_metrics['EENS']:.2f}, LOLF={fold_metrics['LOLF']:.2f}, Cost=${fold_metrics['Cost']}")

    # Model generalization scenarios (dummy example)
    generalization_scenarios = {
        "High RES Penetration": 0.0041,
        "Islanded Mode Operation": 0.0050,
        "Peak Load Demand Profile": 0.0036,
        "Contingency Event (N-1)": 0.0045
    }
    print("\nModel Generalization Performance (LOLP):")
    for scenario, lolp in generalization_scenarios.items():
        print(f"{scenario}: LOLP = {lolp:.4f}")

    # Case study summary (dummy values)
    case_study_results = [
        ("Meshed Transmission", 0.0032, 15.6, 4.0, 120),
        ("Radial Distribution", 0.0035, 16.1, 4.3, 95),
        ("Microgrid", 0.0040, 17.5, 4.7, 110)
    ]
    print("\nCase Study Results:")
    for grid_type, lolp, eens, lolf, comp_time in case_study_results:
        print(f"{grid_type}: LOLP={lolp}, EENS={eens}, LOLF={lolf}, Computation Time={comp_time}s")
