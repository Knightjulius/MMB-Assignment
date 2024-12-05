import numpy as np  
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation

# Constants
GRID_SIZE = (100, 100)  # Grid size (100x100)
MELANOPHORE = 1  # Black stripe cells (melanophores)
XANTHOPHORE = 2  # Yellow interstripe cells (xanthophores)
EMPTY = 0  # Empty cells
STEPS = 100  # Number of simulation steps

# Length scales for cellular interactions
DMM = 50  # Average distance between melanophores
DXX = 36  # Average distance between xanthophores
DXM = 82  # Average distance between melanophores and xanthophores at stripe/interstripe boundaries

# Morse potential parameters (example values)
RMM = 1  # Repulsion strength for melanophores
RXX = 1  # Repulsion strength for xanthophores
RXM = 1  # Repulsion strength for melanophore and xanthophore
AMM = 0.5  # Attraction strength for melanophores
AXX = 0.5  # Attraction strength for xanthophores
AXM = 0.5  # Attraction strength for melanophore and xanthophore

# Initialize grid with empty cells
grid = np.zeros(GRID_SIZE, dtype=int)

def initialize_cells(grid):
    # Top and bottom rows are melanophores with empty spaces between each cell
    grid[0, ::2] = MELANOPHORE  # Set every other cell in the top row to melanophores
    grid[GRID_SIZE[0] - 1, ::2] = MELANOPHORE  # Set every other cell in the bottom row to melanophores
    
    # Middle row is xanthophores
    middle_row = GRID_SIZE[0] // 2
    grid[middle_row, :] = XANTHOPHORE  # Set the middle row to xanthophores
    
    # Rows directly above and below the middle row are melanophores with gaps
    grid[middle_row - 5, ::2] = MELANOPHORE  # Set every other cell in the row above middle row to melanophores
    grid[middle_row + 5, ::2] = MELANOPHORE  # Set every other cell in the row below middle row to melanophores
    return grid

# Function to initialize cells (randomly place melanophores and xanthophores)
def initialize_cells(grid, num_melanophores=200, num_xanthophores=200):
    for _ in range(num_melanophores):
        x, y = random.randint(0, GRID_SIZE[0]-1), random.randint(0, GRID_SIZE[1]-1)
        grid[x, y] = MELANOPHORE
    for _ in range(num_xanthophores):
        x, y = random.randint(0, GRID_SIZE[0]-1), random.randint(0, GRID_SIZE[1]-1)
        grid[x, y] = XANTHOPHORE
    return grid

# Morse potential function to calculate the interaction between two cells
def morse_potential(x1, y1, x2, y2, type1, type2):
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
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
                # Check if the empty cell is in contact with another cell (not empty)
                neighboring_cells = grid[max(0, x - 1):min(GRID_SIZE[0], x + 2), max(0, y - 1):min(GRID_SIZE[1], y + 2)]
                if np.any(neighboring_cells != EMPTY):  # Only consider the empty cell if it's in contact with another cell
                    # Select a neighborhood excluding direct neighbors and check for majority
                    neighborhood = grid[max(0, x - differentiation_neighborhood_size):min(GRID_SIZE[0], x + differentiation_neighborhood_size + 1),
                                        max(0, y - differentiation_neighborhood_size):min(GRID_SIZE[1], y + differentiation_neighborhood_size + 1)]
                    # Exclude direct neighbors (1 cell distance)
                    direct_neighbors = neighborhood[1:-1, 1:-1]  # Exclude immediate neighbors
                    
                    # Count the number of xanthophores and melanophores in the neighborhood (excluding direct neighbors)
                    nearby_xanthophores = np.sum(direct_neighbors == XANTHOPHORE)
                    nearby_melanophores = np.sum(direct_neighbors == MELANOPHORE)

                    # If the majority are xanthophores, the empty cell becomes a melanophore, and vice versa
                    if nearby_xanthophores > nearby_melanophores:
                        new_grid[x, y] = MELANOPHORE
                    else:
                        new_grid[x, y] = XANTHOPHORE
                
    return new_grid


# Function to visualize the grid
def plot_grid(grid):
    cmap = colors.ListedColormap(['white', 'black', 'yellow'])
    return plt.imshow(grid, cmap=cmap)

# Initialize grid with cells based on the new initialization function
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

# Display the animation
plt.title("Zebrafish Stripe Formation")
plt.show()