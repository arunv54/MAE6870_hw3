import numpy as np
import rps.robotarium as robotarium
from rps.utilities.graph import topological_neighbors
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary
from rps.utilities.misc import generate_random_poses

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 8
iterations = 3000
formation_control_gain = 10

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
# Initializing with the updated Width/Height/Spacing from MATLAB
initial_conditions = generate_random_poses(
    N, 
    width=2, 
    height=1, 
    spacing=0.5
)

r = robotarium.Robotarium(
    number_of_robots=N, 
    show_figure=True, 
    initial_conditions=initial_conditions,
    sim_in_real_time=True
)

# =========================================================
# GRAPH TOPOLOGY
# =========================================================
# Adjacency Matrix
adjacency_mat = np.zeros((N,N))
for i in range(-1,len(adjacency_mat)-1):
    adjacency_mat[i,i+1] = 1
adjacency_mat[0,6] = 1
adjacency_mat[1,5] = 1; adjacency_mat[1,6] = 1
adjacency_mat[2,4] = 1; adjacency_mat[2,5] = 1
adjacency_mat[3,7] = 1
adjacency_mat = adjacency_mat + adjacency_mat.transpose()

# Laplacian for a 2D-rigid smile formation
L = L = np.diag(np.sum(adjacency_mat, axis=1)) - adjacency_mat

# =========================================================
# FORMATION GEOMETRY
# =========================================================
desired_positions = 1.5*np.array([[0.5,0,-0.5,-0.75,-0.375,0,0.375,0.75],
                              [0.4,0,0.4,-0.25,-0.4,-0.45,-0.4,-0.25]])

# Weight matrix for desired inter-agent distances
weights = np.zeros((N,N))
for i in range(adjacency_mat.shape[0]):
    for j in range(i,adjacency_mat.shape[1]):
        if adjacency_mat[i,j] == 1:
            weights[i,j] = np.linalg.norm(desired_positions[:,i]-desired_positions[:,j])


# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
# Single-integrator barrier certificate for safety
si_barrier_cert = create_si_barrier_certificate_with_boundary()

# Mapping with specific MATLAB gains
si_to_uni_dyn = create_si_to_uni_dynamics(
    linear_velocity_gain=0.5, 
    angular_velocity_limit=np.pi/2
)

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for t in range(iterations):
    x = r.get_poses()
    dxi = np.zeros((2, N))

    # Formation control algorithm using edge tension energy
    for i in range(N):
        neighbors = topological_neighbors(L, i)
        for j in neighbors:
            # gradient = gain * (||actual||^2 - ||desired||^2) * (pos_j - pos_i)
            dist_sq = np.linalg.norm(x[:2, i] - x[:2, j])**2
            dxi[:, i] += formation_control_gain * (dist_sq - weights[i, j]**2) * (x[:2, j] - x[:2, i])

    # ---------------------------------------------------------
    # Velocity thresholding (3/4 of max linear velocity)
    # ---------------------------------------------------------
    norms = np.linalg.norm(dxi, axis=0)
    threshold = 0.75 * r.MAX_LINEAR_VELOCITY
    to_thresh = norms > threshold
    
    if np.any(to_thresh):
        dxi[:, to_thresh] *= threshold / norms[to_thresh]

    # ---------------------------------------------------------
    # Safety and Dynamics conversion
    # ---------------------------------------------------------
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

# Final debug report
r.debug()