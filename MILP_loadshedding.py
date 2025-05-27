import pulp
import numpy as np


def milp_load_shedding(C_i, P_min, P_max, L_j, P_load, w_j, B_ij, P_ij_max, V_min, V_max, incidence_matrix):


    n_gen = len(C_i)
    n_load = len(L_j)
    n_nodes = len(V_min)
    n_lines = len(P_ij_max)

    # Create the MILP problem instance
    prob = pulp.LpProblem("Optimal_Load_Shedding", pulp.LpMinimize)

    # Decision variables
    P_i = [pulp.LpVariable(f"P_gen_{i}", lowBound=P_min[i], upBound=P_max[i]) for i in range(n_gen)]
    S_j = [pulp.LpVariable(f"Load_Shed_{j}", lowBound=0, upBound=P_load[j]) for j in range(n_load)]
    theta = [pulp.LpVariable(f"Voltage_Angle_{k}", lowBound=V_min[k], upBound=V_max[k]) for k in range(n_nodes)]

    # Objective function: minimize generation cost + penalty cost of unserved load weighted by importance
    prob += (pulp.lpSum([C_i[i] * P_i[i] for i in range(n_gen)]) +
             pulp.lpSum([w_j[j] * L_j[j] * S_j[j] for j in range(n_load)])), "Total_Cost"

    # Constraint 1: Power balance - generation equals load minus load shedding
    # Sum of generation = sum of (load - load_shed)
    prob += (pulp.lpSum(P_i) == pulp.lpSum([P_load[j] - S_j[j] for j in range(n_load)])), "Power_Balance"

    # Constraint 2: DC power flow line limits
    # P_ij = B_ij * (theta_i - theta_j), and |P_ij| <= P_ij_max
    # Using incidence matrix, line power flow for each line l: P_ij = B_l * sum over nodes (incidence[l][k] * theta_k)
    # For each line:
    for l in range(n_lines):
        # Calculate power flow on line l using voltage angles
        flow_expr = pulp.lpSum([incidence_matrix[l][k] * theta[k] for k in range(n_nodes)])
        flow_expr_scaled = B_ij[l] * flow_expr
        prob += flow_expr_scaled <= P_ij_max[l], f"LineLimit_Pos_{l}"
        prob += flow_expr_scaled >= -P_ij_max[l], f"LineLimit_Neg_{l}"

    # Constraint 3: Voltage angle limits at each node (already bounded in variable definitions)

    # Solve the problem using default solver
    prob.solve()

    # Prepare results dictionary
    results = {
        "Status": pulp.LpStatus[prob.status],
        "Generation": np.array([pulp.value(var) for var in P_i]),
        "Load_Shedding": np.array([pulp.value(var) for var in S_j]),
        "Voltage_Angles": np.array([pulp.value(var) for var in theta]),
    }

    # Compute line flows based on optimized angles
    line_flows = np.zeros(n_lines)
    for l in range(n_lines):
        flow_sum = 0
        for k in range(n_nodes):
            flow_sum += incidence_matrix[l][k] * results["Voltage_Angles"][k]
        line_flows[l] = B_ij[l] * flow_sum
    results["Line_Flows"] = line_flows

    # Check constraints violations (optional)
    violations = {
        "Line_Flow_Violations": np.maximum(np.abs(line_flows) - P_ij_max, 0)
    }
    results["Violations"] = violations

    return results


# Example usage (if run as script)
if __name__ == "__main__":
    # Dummy example parameters (replace with actual data loading)
    n_gen = 3
    n_load = 4
    n_nodes = 4
    n_lines = 5

    C_i = np.array([20, 25, 30])  # $/MWh
    P_min = np.array([10, 15, 10])
    P_max = np.array([100, 90, 80])

    L_j = np.array([1000, 1200, 1500, 1100])  # Penalty cost for unserved load
    P_load = np.array([50, 60, 40, 30])

    w_j = np.array([1, 1, 2, 1])  # Importance weights

    B_ij = np.array([0.1, 0.2, 0.15, 0.12, 0.18])  # Line susceptance
    P_ij_max = np.array([40, 30, 35, 25, 30])

    V_min = np.array([-0.1, -0.1, -0.1, -0.1])
    V_max = np.array([0.1, 0.1, 0.1, 0.1])

    incidence_matrix = np.array([
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])

    results = milp_load_shedding(C_i, P_min, P_max, L_j, P_load, w_j, B_ij, P_ij_max, V_min, V_max, incidence_matrix)

    print("MILP Load Shedding Optimization Results:")
    print("Status:", results["Status"])
    print("Generation Outputs:", results["Generation"])
    print("Load Shedding:", results["Load_Shedding"])
    print("Voltage Angles:", results["Voltage_Angles"])
    print("Line Flows:", results["Line_Flows"])
    print("Line Flow Violations:", results["Violations"]["Line_Flow_Violations"])
