import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import matplotlib.animation as animation

def find_neighbours(positions, cutoff = 2, debug=False):
    N = len(positions)
    neighbours = {}
    for i in range(N):
        distances = np.linalg.norm(positions - positions[i], axis=1)
        if debug:
            print(f'For particle {i}, distances are:')
            print(distances)
        
        neighbour_indices = np.where(distances < cutoff)[0]
        neighbour_indices = neighbour_indices[neighbour_indices != i]
        neighbours[i] = list(neighbour_indices)
    return neighbours

def calculate_interaction(ri, rj, k, epsilon):
    forces = np.zeros((N, 2))
    rij = rj - ri
    r = np.linalg.norm(rij)
    # Combined radius of both particles (assume unit radii for now)
    bij = 2.0  
    if r < bij*(1 + epsilon):
        force = k*(r - bij)*rij/r  
    elif r < bij*(1 + 2*epsilon):
        force = -k*(r - bij - 2*epsilon)*rij/r
    else:
        force = 0.0
    return force

def boundary_effects(ri, fi, boundary, strength = 1):
    x, y = ri
    fx, fy = fi
    reverse_angle = 0
    if x > boundary or x < -boundary:
        fx = -strength*fx
        reverse_angle = 1
    if y > boundary or y < -boundary:
        fy = -strength*fy
        reverse_angle = 1

    return np.array([fx, fy]), reverse_angle

def animate(output, interactions, boundary):
    fig, ax = plt.subplots()
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    # Draw boundary box
    boundary_box = patches.Rectangle((-boundary, -boundary), 2*boundary, 2*boundary, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(boundary_box)

    # Draw particles
    circles = [patches.Circle((0, 0), radius=1, fill=True, alpha=0.5) for _ in range(len(output[0]))]
    for circle in circles:
        ax.add_patch(circle)

    def update(frame):
        for i, circle in enumerate(circles):
            circle.center = output[frame][i]
            circle.set_facecolor('red' if interactions[frame][i] else 'blue')

    ani = animation.FuncAnimation(fig, update, frames=range(len(output)))
    plt.show()

def plot_output(output):
    T, N, _ = output.shape

    plt.figure(figsize=(5, 5))
    for t in range(0, T, 10):  # Plot every 10th time step for clarity
        color = str(1 - t / T)  # From light (near 1) to dark (near 0)
        plt.scatter(output[t, :, 0], output[t, :, 1], c=color, s=20)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Particle Spread Over Time')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5) 
    plt.show()

def main(N, params, boundary, T = 100, dt = 0.01, seed = 20, debug=False):
    # Parameters
    v0, tau, k, epsilon = params[0], params[1], params[2], params[3] 
    print(f"v0:{v0}, tau:{tau}, k:{k}, epsilon:{epsilon}")
    # Diffusion strength
    D = 1 / (2 * tau) if tau > 0 else 0

    # Initialize
    np.random.seed(seed)
    R = 10
    # Initial angles for the particles
    starting_angle = np.random.rand(N) * 2 * np.pi
    # Positions — distributed in a circle of radius R
    positions = np.column_stack((R * np.cos(starting_angle), R * np.sin(starting_angle))) 
    angles = np.random.rand(N) * 2 * np.pi
    forces = np.zeros((N, 2))

    output = np.zeros((T, N, 2))
    interactions = np.zeros((T, N))
    output[0] = positions
    # Time evolution
    for t in tqdm(range(T)):
        neighbours = find_neighbours(positions)

        # Update positions
        for i in range(N):
            ri = positions[i]

            # Calculate active forces
            ni = np.array([np.cos(angles[i]), np.sin(angles[i])])
            active_force = v0 * ni

            # Calculate interaction forces
            interaction_force = 0
            if debug:
                print(f"{t}: particle {i} has position {positions[i]}")
            if len(neighbours[i]) > 0:                
                interactions[t, i] = 1
            for j in neighbours[i]:
                rj = positions[j]
                if debug:
                    print(f"{t} : particle {i} has neighbours {j} at position {rj}")
                interaction_force += calculate_interaction(ri, rj, k, epsilon) 
                interactions[t, j] = 1
            total_force = active_force + interaction_force

            # Apply boundary effects
            corrected_force, reverse_angle = boundary_effects(positions[i], total_force, boundary=boundary)

            # Update positions
            positions[i] += dt * corrected_force
            angles[i] += reverse_angle*np.pi
        if debug:
            print(f'Interactions after timestep {t}')
            print(interactions[t])

        # Update noise term in angles
        # noise = np.random.normal(0, np.sqrt(2 * D * dt), N)
        # angle += noise    
        output[t] = positions

    return output, interactions

if __name__ == "__main__":
    # Main parameters
    N = 10
    v0 = 50 # Active force
    k = 10.  # Stiffness of interaction
    boundary = 20

    # Assume 0 for now
    epsilon = 0.  # Attractive component of interaction
    tau = 0.0  # Noise term
    params = v0, tau, k, epsilon
    output, interactions = main(N, params, boundary)
    animate(output, interactions, boundary)
    
    