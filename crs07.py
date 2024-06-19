import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 20
C = 1.0
v = 0.001
r = 0.045
P = 0.015
time_steps = 500

# Initialization
positions = np.random.uniform(0, C, N)
directions = np.random.choice([-1, 1], N)  # -1 for left, +1 for right

# Function to update positions and directions
def update_positions(positions, directions):
    new_positions = (positions + v * directions) % C
    return new_positions

def update_directions(positions, directions):
    new_directions = np.copy(directions)
    for i in range(N):
        # Find locusts within the perception range
        distances = np.abs(positions - positions[i])
        distances = np.minimum(distances, C - distances)  # considering the ring
        within_range = (distances < r)
        
        if np.random.rand() < P:
            new_directions[i] *= -1
        else:
            # Check the majority direction within the perception range
            if np.sum(directions[within_range]) < 0:
                new_directions[i] = -1
            elif np.sum(directions[within_range]) > 0:
                new_directions[i] = 1

    return new_directions

# Simulation
left_going_count = []

for t in range(time_steps):
    positions = update_positions(positions, directions)
    directions = update_directions(positions, directions)
    left_going_count.append(np.sum(directions == -1))


# Plotting
plt.plot(left_going_count)
plt.xlabel('Time Step')
plt.ylabel('Number of Left-going Locusts')
plt.title('a)Number of Left-going Locusts Over Time')
filename = 'a)num_left_locusts.png'
plt.savefig(filename)
plt.grid(True)
plt.show()

###================(b)====================###

# Initialize transition count array
A = np.zeros((N + 1, N + 1), dtype=int)

# Simulation for histogram
num_runs = 1000

for run in range(num_runs):
    positions = np.random.uniform(0, C, N)
    directions = np.random.choice([-1, 1], N)
    
    for t in range(time_steps):
        L_t = np.sum(directions == -1)
        positions = update_positions(positions, directions)
        directions = update_directions(positions, directions)
        L_t1 = np.sum(directions == -1)
        A[L_t, L_t1] += 1

# Plotting the histogram
plt.imshow(A, cmap='hot', interpolation='nearest')
plt.colorbar(label='Frequency')
plt.xlabel('L(t)')
plt.ylabel('L(t+1)')
plt.title('b)Histogram of Transitions L(t) -> L(t+1)')
filename = 'b)histogram_transitions.png'
plt.savefig(filename)
plt.grid(True)
plt.show()


###================(c)====================###

# Count occurrences of each state L
M = np.sum(A, axis=1)

# Normalize the transition array to get probabilities
P = np.zeros_like(A, dtype=float)
for i in range(N + 1):
    if M[i] > 0:
        P[i] = A[i] / M[i]

# Sample evolution of L_t using the transition probabilities
def sample_L_trajectory(P, initial_L, time_steps):
    L = initial_L
    trajectory = [L]
    
    for t in range(time_steps):
        L = np.random.choice(np.arange(N + 1), p=P[L])
        trajectory.append(L)
    
    return trajectory

# Initial state L_0
initial_L = np.random.randint(0, N + 1)

# Sample one trajectory
trajectory = sample_L_trajectory(P, initial_L, time_steps)

# Plot the sampled trajectory
plt.plot(trajectory)
plt.xlabel('Time Step')
plt.ylabel('Number of Left-going Locusts')
plt.title('c)Sampled Trajectory of L over Time')
filename = 'c)sampled_trajectory.png'
plt.savefig(filename)
plt.show()

# Compare with initial plot
