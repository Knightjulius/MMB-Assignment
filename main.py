from IPython.display import HTML
import numpy as np 
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib import colors
import matplotlib.pyplot as plt
import os
import random
# Constants
GRID_SIZE = (50, 50)  # Grid size (100x100)
MELANOPHORE = 1  # Black stripe cells (melanophores)
XANTHOPHORE = 2  # Yellow interstripe cells (xanthophores)
EMPTY = 0  # Empty cells
STEPS = 100  # Number of simulation steps
# Length scales for cellular interactions
DMM = 50  # Average distance between melanophores
DXX = 36  # Average distance between xanthophores
DXM = 82  # Average distance between melanophores and xanthophores at stripe/interstripe boundaries

# Morse potential parameters (example values)
RMM = 0.1  # Repulsion strength for melanophores
RXX = 0.1  # Repulsion strength for xanthophores
RXM = 0.1  # Repulsion strength for melanophore and xanthophore
AMM = 0.05  # Attraction strength for melanophores
AXX = 0.05  # Attraction strength for xanthophores
AXM = 0.05  # Attraction strength for melanophore and xanthophore

# Initialize grid with empty cells
grid = np.zeros(GRID_SIZE, dtype=int)

# Function to initialize cells (randomly place melanophores and xanthophores)
def initialize_cells(grid):
    # Top and bottom rows are melanophores with empty spaces between each cell
    grid[0, ::2] = MELANOPHORE  # Set every other cell in the top row to melanophores
    grid[GRID_SIZE[0] - 1, ::2] = MELANOPHORE  # Set every other cell in the bottom row to melanophores
    
    # Middle row is xanthophores
    middle_row = GRID_SIZE[0] // 2
    grid[middle_row, :] = XANTHOPHORE  # Set the middle row to xanthophores
    grid[middle_row - 20, :] = XANTHOPHORE  # Set the middle row to xanthophores
    grid[middle_row + 20, :] = XANTHOPHORE  # Set the middle row to xanthophores
    
    # Rows directly above and below the middle row are melanophores with gaps
    grid[middle_row - 5, ::2] = MELANOPHORE  # Set every other cell in the row above middle row to melanophores
    grid[middle_row + 5, ::2] = MELANOPHORE  # Set every other cell in the row below middle row to melanophores
    return grid

# Function to calculate the number of adjacent melanophores and xanthophores
def count_neighbors(grid, x, y):
    melanophores = 0
    xanthophores = 0
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            # Use modulo for wrapping
            ni = i % GRID_SIZE[0]  # Wrap row
            nj = j % GRID_SIZE[1]  # Wrap column
            if (ni != x or nj != y):  # Exclude the current cell
                if grid[ni, nj] == MELANOPHORE:
                    melanophores += 1
                elif grid[ni, nj] == XANTHOPHORE:
                    xanthophores += 1
    return melanophores, xanthophores

# Morse potential function to calculate the interaction between two cells
def morse_potential(x1, y1, x2, y2, type1, type2):
    # Calculate wrapped distance
    dx = min(abs(x2 - x1), GRID_SIZE[0] - abs(x2 - x1))
    dy = min(abs(y2 - y1), GRID_SIZE[1] - abs(y2 - y1))
    distance = np.sqrt(dx ** 2 + dy ** 2)
    
    if distance == 0:
        return 0  # No interaction if cells are at the same position
    
    # Define the Morse potential parameters based on the cell types
    if type1 == MELANOPHORE and type2 == MELANOPHORE:
        R, A, r, a = RMM, AMM, DMM, DMM
    elif type1 == XANTHOPHORE and type2 == XANTHOPHORE:
        R, A, r, a = RXX, AXX, DXX, DXX
    else:
        R, A, r, a = RXM, AXM, DXM, DXM
    
    # Morse potential formula
    return R * np.exp(-distance / r) - A * np.exp(-distance / a)



# Function to update the grid based on interaction rules
def update_grid(grid, differentiation_neighborhood_size=3):
    new_grid = grid.copy()
    
    # Loop through all cells on the grid
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            if grid[x, y] != EMPTY:
                # Death condition for melanophores: majority of direct neighbors are xanthophores
                if grid[x, y] == MELANOPHORE:
                    # Get direct neighbors (1 step in all directions)
                    direct_neighbors = grid[max(0, x - 1):min(GRID_SIZE[0], x + 2), max(0, y - 1):min(GRID_SIZE[1], y + 2)]
                    # Count the number of xanthophores in the direct neighborhood
                    nearby_xanthophores = np.sum(direct_neighbors == XANTHOPHORE) - (direct_neighbors[1, 1] == XANTHOPHORE)  # Exclude the current cell
                    # If the majority of direct neighbors are xanthophores, the melanophore dies
                    if nearby_xanthophores > 4:  # Threshold for melanophore death (more than 4 xanthophores in the 8 neighbors)
                        new_grid[x, y] = EMPTY
                
                # Death condition for xanthophores: majority of direct neighbors are melanophores
                elif grid[x, y] == XANTHOPHORE:
                    # Get direct neighbors (1 step in all directions)
                    direct_neighbors = grid[max(0, x - 1):min(GRID_SIZE[0], x + 2), max(0, y - 1):min(GRID_SIZE[1], y + 2)]
                    # Count the number of melanophores in the direct neighborhood
                    nearby_melanophores = np.sum(direct_neighbors == MELANOPHORE) - (direct_neighbors[1, 1] == MELANOPHORE)  # Exclude the current cell
                    # If the majority of direct neighbors are melanophores, the xanthophore dies
                    if nearby_melanophores > 4:  # Threshold for xanthophore death (more than 4 melanophores in the 8 neighbors)
                        new_grid[x, y] = EMPTY
                
                # Only move cells if they are not dead
                if new_grid[x, y] != EMPTY:
                    # Calculate the net force on each cell based on the Morse potential
                    force_x = 0
                    force_y = 0
                    for i in range(GRID_SIZE[0]):
                        for j in range(GRID_SIZE[1]):
                            if (i != x or j != y) and grid[i, j] != EMPTY:
                                potential = morse_potential(x, y, i, j, grid[x, y], grid[i, j])
                                force_x += potential * (i - x) / np.sqrt((i - x)**2 + (j - y)**2)
                                force_y += potential * (j - y) / np.sqrt((i - x)**2 + (j - y)**2)
                    
                    # Move the cell based on the calculated forces (scaled for simplicity)
                    new_x = min(max(0, x + int(force_x * 0.1)), GRID_SIZE[0] - 1)
                    new_y = min(max(0, y + int(force_y * 0.1)), GRID_SIZE[1] - 1)
                    
                    # If the cell moves to a new location, update the grid
                    if new_grid[new_x, new_y] == EMPTY:
                        new_grid[new_x, new_y] = grid[x, y]
                        new_grid[x, y] = EMPTY
                
            # Differentiation rule: check for empty spaces, excluding direct neighbors
            if grid[x, y] == EMPTY:
                melanophores, xanthophores = count_neighbors(grid, x, y)
                 # Empty cell differentiation (may become melanophore or xanthophore)
                if melanophores > xanthophores:
                    if random.random() < 0.05:  # Random chance for differentiation into melanophore
                        new_grid[x, y] = MELANOPHORE
                elif xanthophores > melanophores:
                    if random.random() < 0.05:  # Random chance for differentiation into xanthophore
                        new_grid[x, y] = XANTHOPHORE
    return new_grid


# Function to visualize the grid
def plot_grid(grid):
    cmap = colors.ListedColormap(['white', 'black', 'yellow'])
    return plt.imshow(grid, cmap=cmap)

# Initialize grid with cells
grid = initialize_cells(grid)  # Initialize cells

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(grid, cmap=colors.ListedColormap(['white', 'black', 'yellow']))

# Update function for animation
def update(frame):
    global grid
    grid = update_grid(grid)  # Update grid at each step
    cax.set_array(grid)  # Update the grid visualization
    return [cax]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=200, blit=True)

# Saving the animation
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
os.chdir(script_dir)  # Change to the script's directory

# Save the animation
writer = FFMpegWriter(fps=30, metadata={'artist': 'Benthe & Julius'}, bitrate=1800)
file_path = os.path.join(script_dir, 'zebrafish_stripe_formation.mp4')
ani.save(file_path, writer=writer)
