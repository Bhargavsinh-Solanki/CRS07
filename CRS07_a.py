import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
C = 1.0  # circumference of the ring
v = 0.001  # speed of the locusts
r = 0.045  # perception range
P = 0.015  # probability of spontaneous switching
N = 20  # number of locusts
time_steps = 500  # number of time steps

# Initialize locust positions and directions
positions = np.random.uniform(0, C, N)
directions = np.random.choice([-1, 1], N)

# Function to update the positions and directions of locusts
def update_locusts(positions, directions):
    new_directions = directions.copy()
    
    for i in range(N):
        # Find locusts within the perception range
        locusts_in_range = np.abs((positions - positions[i] + C/2) % C - C/2) <= r
        opposite_directions = directions[locusts_in_range] == -directions[i]
        
        # Rule 1: Majority of locusts in perception range have opposite direction
        if np.sum(opposite_directions) > np.sum(locusts_in_range) / 2:
            new_directions[i] = -directions[i]
        
        # Rule 2: Spontaneous switching
        if np.random.rand() < P:
            new_directions[i] = -directions[i]
    
    # Update positions based on new directions
    positions += new_directions * v
    positions %= C  # Ensure positions stay within the ring
    
    return positions, new_directions

# Simulation loop
left_going_counts = []

for t in range(time_steps):
    positions, directions = update_locusts(positions, directions)
    left_going_counts.append(np.sum(directions == -1))

# Plotting the number of left-going locusts over time
plt.figure(figsize=(10, 6))
plt.plot(left_going_counts, label='Left-going locusts')
plt.xlabel('Time step')
plt.ylabel('Number of left-going locusts')
plt.title('Number of left-going locusts over time')
plt.legend()
plt.grid(True)
plt.show()
