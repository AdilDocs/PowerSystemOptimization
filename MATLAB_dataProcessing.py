import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math

# Check if MATLAB and matlab.engine are installed
try:
    import matlab.engine

    print("MATLAB is available and the required version of Python is installed.")
except ImportError:
    print(
        "Error: MATLAB or the required version of Python is not installed. Please install MATLAB and the required Python package 'matlab.engine' to run this code.")
    exit(1)  # Exit the script if MATLAB is not available


# --- Load IEEE RTS-96 Dataset from .mat file ---
def load_rts96_data(mat_file_path):
    """
    Function to load IEEE RTS-96 dataset from .mat file

    """
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)

    # Extract the relevant fields (adjust according to the dataset structure)
    # For example, assuming 'generation_cost', 'load_demand', 'line_capacity' are stored in the mat file
    generation_cost = mat_data['generation_cost'].flatten()  # Flatten in case it's a 2D array
    load_demand = mat_data['load_demand'].flatten()
    line_limits = mat_data['line_limits'].flatten()

    return generation_cost, load_demand, line_limits


# --- Objective function for optimization ---
def objective_function(x, generation_cost, load_demand):

    P_i = x[0]  # Generator output in MW
    S_j = x[1]  # Load shedding in MW

    # Minimize the total cost: generation cost + penalty cost for unserved load
    cost = np.sum(generation_cost * P_i) + np.sum(load_demand * S_j)
    return cost


# --- Nonlinear constraint function ---
def nonlin_constraint(x, line_limits):

    P_i = x[0]  # Generator output
    S_j = x[1]  # Load shedding

    power_balance = np.sum(P_i) - np.sum(load_demand + S_j)

    line_capacity_constraint = np.sum(np.abs(P_i)) - line_limits

    return power_balance, line_capacity_constraint


# --- Function to run fmincon solver inside Python ---
def run_fmincon(mat_file_path):
    # Load the dataset from .mat file
    generation_cost, load_demand, line_limits = load_rts96_data(mat_file_path)

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Initial guess for the optimization problem
    x0 = np.array([100, 10])  # Example: Initial values for generator output and load shedding
    lb = np.array([0, 0])  # Lower bounds: P_i >= 0, S_j >= 0
    ub = np.array([200, 100])  # Upper bounds: P_i <= 200 MW, S_j <= 100 MW

    # Convert Python arrays to MATLAB-compatible format
    x0_matlab = matlab.double(x0.tolist())
    lb_matlab = matlab.double(lb.tolist())
    ub_matlab = matlab.double(ub.tolist())

    # Define the objective function for fmincon
    matlab_func = matlab.function_handle(objective_function)

    # Define the constraint function
    constraint_func = matlab.function_handle(nonlin_constraint)

    # Set the options for the optimization solver
    options = eng.optimset('Display', 'iter', 'Algorithm', 'sqp')  # Sequential Quadratic Programming (SQP)

    # Run the fmincon solver to minimize the objective function
    result = eng.fmincon(matlab_func, x0_matlab, [], [], [], [], lb_matlab, ub_matlab, constraint_func, options)

    # Stop the MATLAB engine
    eng.quit()

    # Return the optimization result
    return result


def dc_power_flow(P_gen, B_ij, theta, theta_min, theta_max):

    # Ensure voltage angle limits are respected
    theta = np.clip(theta, theta_min, theta_max)

    # Calculate the line power flows
    P_ij = np.dot(B_ij, (theta - theta.T))  # Matrix multiplication: B_ij * (theta_i - theta_j)

    return P_ij
def power_flow_constraints(P_ij, line_limits):

    # Ensure that power flows do not exceed the line limits
    constraint_violation = np.abs(P_ij) - line_limits

    # Any violation is a constraint violation
    constraint_violation = np.maximum(constraint_violation, 0)

    return constraint_violation

def dc_power_flow_optimization(mat_file_path, P_gen, theta):


    generation_cost, load_demand, line_limits, B_ij, theta_min, theta_max = load_rts96_data(mat_file_path)

    # Calculate line power flows using DC Power Flow model
    P_ij = dc_power_flow(P_gen, B_ij, theta, theta_min, theta_max)

    # Check if the power flow exceeds line capacities (constraints)
    constraints_violations = power_flow_constraints(P_ij, line_limits)

    mat_file_path = 'path_to_your_dataset/IEEE_RTS96.mat'

    # Example generator output (P_gen) and voltage angles (theta)
    P_gen = np.random.rand(96) * 200  # Assuming 96 nodes in IEEE RTS-96
    theta = np.random.rand(96) * 2 * np.pi  # Random voltage angles (0 to 2pi)

    # Perform DC power flow optimization
    P_ij, constraints_violations = dc_power_flow_optimization(mat_file_path, P_gen, theta)

    print("Calculated Line Power Flows (P_ij):")
    print(P_ij)

    print("Constraints Violations (if any):")
    print(constraints_violations)

    # Optional: Visualize the results of the power flow and violations
    plt.figure(figsize=(10, 6))
    plt.plot(P_ij, label='Line Power Flow (P_ij)')
    plt.plot(constraints_violations, label='Constraint Violations')
    plt.xlabel('Line Index')
    plt.ylabel('Power Flow / Violations')
    plt.title('Line Power Flow and Constraint Violations')
    plt.legend()
    plt.show()

    return P_ij, constraints_violations