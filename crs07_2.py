import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
C = 1.0
v = 0.001
r = 0.045
P = 0.015
time_steps = 5000
swarm_sizes = np.arange(20, 151, 10)
num_runs_per_size = 10

# Function to update positions and directions (same as in Task 7.1)
def update_positions(positions, directions, C, v):
    new_positions = (positions + v * directions) % C
    return new_positions

def update_directions(positions, directions, r, P, C):
    new_directions = np.copy(directions)
    N = len(positions)
    for i in range(N):
        distances = np.abs(positions - positions[i])
        distances = np.minimum(distances, C - distances)
        within_range = (distances < r)
        if np.random.rand() < P:
            new_directions[i] *= -1
        else:
            if np.sum(directions[within_range]) < 0:
                new_directions[i] = -1
            elif np.sum(directions[within_range]) > 0:
                new_directions[i] = 1
    return new_directions

# Function to run a single simulation
def run_simulation(N, time_steps, C, v, r, P):
    positions = np.random.uniform(0, C, N)
    directions = np.random.choice([-1, 1], N)
    switch_times = []
    counter = 0
    current_zone = 'B'
    previous_zone = 'B'  # Initialize previous_zone

    for t in tqdm(range(time_steps), desc=f"Simulating N={N}"):
        positions = update_positions(positions, directions, C, v)
        directions = update_directions(positions, directions, r, P, C)
        L = np.sum(directions == -1)

        if L > 0.7 * N:
            zone = 'A'
        elif L < 0.3 * N:
            zone = 'C'
        else:
            zone = 'B'

        if zone == 'B':
            counter += 1
        else:
            if current_zone == 'B' and zone in ['A', 'C'] and zone != previous_zone:
                switch_times.append(counter)
                counter = 0
            current_zone = zone
            previous_zone = zone  # Update previous_zone

    return switch_times

# Collect data for different swarm sizes
average_switch_times = []
number_of_switches = []

for N in swarm_sizes:
    all_switch_times = []
    for _ in range(num_runs_per_size):
        switch_times = run_simulation(N, time_steps, C, v, r, P)
        all_switch_times.extend(switch_times)
    average_switch_times.append(np.mean(all_switch_times) if all_switch_times else float('inf'))
    number_of_switches.append(len(all_switch_times))

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(swarm_sizes, average_switch_times, marker='o')
plt.xlabel('Swarm Size (N)')
plt.ylabel('Average Global Switch Time')
plt.title('Average Global Switch Time vs. Swarm Size')

plt.subplot(1, 2, 2)
plt.plot(swarm_sizes, number_of_switches, marker='o')
plt.xlabel('Swarm Size (N)')
plt.ylabel('Number of Global Switches')
plt.title('Number of Global Switches vs. Swarm Size')

plt.tight_layout()
plt.show()
