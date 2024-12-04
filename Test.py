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
RMM = 0.1  # Repulsion strength for melanophores
RXX = 0.1  # Repulsion strength for xanthophores
RXM = 0.1  # Repulsion strength for melanophore and xanthophore
AMM = 0.05  # Attraction strength for melanophores
AXX = 0.05  # Attraction strength for xanthophores
AXM = 0.05  # Attraction strength for melanophore and xanthophore

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

def update_grid(grid, neighborhood_size=2):
    new_grid = grid.copy()
    
    # Loop through all cells on the grid
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            if grid[x, y] != EMPTY:
                # Step 1: Check if the cell dies by looking at the majority of opposite type cells in the direct neighborhood
                if grid[x, y] == MELANOPHORE:
                    # Define the neighborhood for the current cell based on neighborhood_size
                    nearby_xanthophores = np.sum(grid[max(0, x - neighborhood_size):min(GRID_SIZE[0], x + neighborhood_size + 1), 
                                                     max(0, y - neighborhood_size):min(GRID_SIZE[1], y + neighborhood_size + 1)] == XANTHOPHORE)
                    if nearby_xanthophores > (2 * neighborhood_size + 1) ** 2 / 2:  # Majority of neighbors are xanthophores
                        new_grid[x, y] = EMPTY
                elif grid[x, y] == XANTHOPHORE:
                    # Define the neighborhood for the current cell based on neighborhood_size
                    nearby_melanophores = np.sum(grid[max(0, x - neighborhood_size):min(GRID_SIZE[0], x + neighborhood_size + 1), 
                                                      max(0, y - neighborhood_size):min(GRID_SIZE[1], y + neighborhood_size + 1)] == MELANOPHORE)
                    if nearby_melanophores > (2 * neighborhood_size + 1) ** 2 / 2:  # Majority of neighbors are melanophores
                        new_grid[x, y] = EMPTY
                
                # Step 2: If the cell does not die, check for movement (based on forces)
                if new_grid[x, y] != EMPTY:  # Only move non-empty cells
                    force_x = 0
                    force_y = 0
                    for i in range(GRID_SIZE[0]):
                        for j in range(GRID_SIZE[1]):
                            if (i != x or j != y) and grid[i, j] != EMPTY:
                                potential = morse_potential(x, y, i, j, grid[x, y], grid[i, j])
                                force_x += potential * (i - x) / np.sqrt((i - x)**2 + (j - y)**2)
                                force_y += potential * (j - y) / np.sqrt((i - x)**2 + (j - y)**2)
                    
                    # Move the cell based on the forces
                    new_x = min(max(0, x + int(force_x * 0.1)), GRID_SIZE[0] - 1)
                    new_y = min(max(0, y + int(force_y * 0.1)), GRID_SIZE[1] - 1)
                    
                    # If the cell moves to a new location, update the grid
                    if new_grid[new_x, new_y] == EMPTY:
                        new_grid[new_x, new_y] = grid[x, y]
                        new_grid[x, y] = EMPTY

            # Step 3: Check empty spaces for differentiation
            if grid[x, y] == EMPTY:
                # Define the neighborhood for the current cell based on neighborhood_size
                nearby_cells = grid[max(0, x - neighborhood_size):min(GRID_SIZE[0], x + neighborhood_size + 1), 
                                    max(0, y - neighborhood_size):min(GRID_SIZE[1], y + neighborhood_size + 1)]
                # Condition for melanophore differentiation (more than 2 xanthophores nearby)
                if np.sum(nearby_cells == XANTHOPHORE) > (2 * neighborhood_size + 1) ** 2 / 2:
                    new_grid[x, y] = MELANOPHORE
                # Condition for xanthophore differentiation (more than 2 melanophores nearby)
                elif np.sum(nearby_cells == MELANOPHORE) > (2 * neighborhood_size + 1) ** 2 / 2:
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