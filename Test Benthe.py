from IPython.display import HTML
import numpy as np 
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib import colors
import matplotlib.pyplot as plt
import os

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
RMM = 1.0  # Repulsion strength for melanophores
RXX = 1.0  # Repulsion strength for xanthophores
RXM = 1.0  # Repulsion strength for melanophore and xanthophore
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
def update_grid(grid):
    new_grid = grid.copy()
    
    # Loop through all cells on the grid
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            if grid[x, y] != EMPTY:
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
                
                # Implement differentiation and death rules here
                if grid[x, y] == MELANOPHORE:
                    # Cell death condition for melanophores: too many xanthophores nearby
                    nearby_xanthophores = np.sum(grid[max(0, x - 1):min(GRID_SIZE[0], x + 2), max(0, y - 1):min(GRID_SIZE[1], y + 2)] == XANTHOPHORE)
                    if nearby_xanthophores > 3:  # Threshold for melanophore death
                        new_grid[x, y] = EMPTY
                
                elif grid[x, y] == XANTHOPHORE:
                    # Cell death condition for xanthophores: too many melanophores nearby
                    nearby_melanophores = np.sum(grid[max(0, x - 1):min(GRID_SIZE[0], x + 2), max(0, y - 1):min(GRID_SIZE[1], y + 2)] == MELANOPHORE)
                    if nearby_melanophores > 3:  # Threshold for xanthophore death
                        new_grid[x, y] = EMPTY
                
                # Differentiation rule: check for empty spaces
                if grid[x, y] == EMPTY:
                    # Randomly select a nearby region and check for conditions for differentiation
                    nearby_cells = grid[max(0, x - 1):min(GRID_SIZE[0], x + 2), max(0, y - 1):min(GRID_SIZE[1], y + 2)]
                    if np.random.rand() < 0.1:  # Probability of differentiation
                        if np.sum(nearby_cells == MELANOPHORE) > np.sum(nearby_cells == XANTHOPHORE):  # More melanophores nearby
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
    """
    global grid
    
    # Create a new grid with increased size
    new_size = (grid.shape[0] + 1, grid.shape[1] + 1)
    new_grid = np.zeros(new_size, dtype=int)
    
    # Copy the existing grid into the new grid
    new_grid[:grid.shape[0], :grid.shape[1]] = grid
    
    # Initialize new row and column
    # Here, you can decide how to initialize these new cells.
    # For simplicity, we fill the new row and column with empty cells.
    # You can add patterns if needed (e.g., random cells or based on rules).
    new_grid[grid.shape[0], :] = EMPTY  # New row
    new_grid[:, grid.shape[1]] = EMPTY  # New column
    
    # Assign the expanded grid back to `grid`
    grid = new_grid
    
    # Update the grid using the existing update_grid function
    grid = update_grid(grid)  # Update grid at each step
    
    # Update the plot with the new grid size
    cax.set_array(grid)
    cax.set_extent([0, grid.shape[1], 0, grid.shape[0]])
    
    #grid = update_grid(grid)  # Update grid at each step
    #cax.set_array(grid)  # Update the grid visualization
    return [cax]
    """

    global grid
    p = 0.5  # Probability of increasing grid size
    
    # Determine whether to expand the grid
    if np.random.rand() < p:
        new_size = (grid.shape[0] + 2, grid.shape[1] + 2)
        new_grid = np.zeros(new_size, dtype=int)
        
        # Calculate offsets to center the existing grid
        row_offset = (new_size[0] - grid.shape[0]) // 2
        col_offset = (new_size[1] - grid.shape[1]) // 2
        
        # Place the existing grid in the middle of the new grid
        new_grid[row_offset:row_offset + grid.shape[0], col_offset:col_offset + grid.shape[1]] = grid
        
        # Assign the expanded grid back to `grid`
        grid = new_grid
    
    # Update the grid using the existing update_grid function
    grid = update_grid(grid)  # Update grid at each step
    
    # Update the plot with the new grid size
    cax.set_array(grid)
    cax.set_extent([0, grid.shape[1], 0, grid.shape[0]])  # Update plot extent
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
