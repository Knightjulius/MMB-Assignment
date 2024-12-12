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

# Initialize grid with empty cells
grid = np.zeros(GRID_SIZE, dtype=int)

# Function to initialize cells (randomly place melanophores and xanthophores)
def initialize_cells(grid, num_melanophores=200, num_xanthophores=200):
    for _ in range(num_melanophores):
        x, y = random.randint(0, GRID_SIZE[0]-1), random.randint(0, GRID_SIZE[1]-1)
        grid[x, y] = MELANOPHORE
    for _ in range(num_xanthophores):
        x, y = random.randint(0, GRID_SIZE[0]-1), random.randint(0, GRID_SIZE[1]-1)
        grid[x, y] = XANTHOPHORE
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

# Function to calculate the number of adjacent melanophores and xanthophores
def count_neighbors(grid, x, y):
    melanophores = 0
    xanthophores = 0
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if 0 <= i < GRID_SIZE[0] and 0 <= j < GRID_SIZE[1] and (i != x or j != y):
                if grid[i, j] == MELANOPHORE:
                    melanophores += 1
                elif grid[i, j] == XANTHOPHORE:
                    xanthophores += 1
    return melanophores, xanthophores

# Function to update the grid based on interaction rules
def update_grid(grid):
    new_grid = grid.copy()
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            melanophores, xanthophores = count_neighbors(grid, x, y)
            
            if grid[x, y] == MELANOPHORE:
                # Melanophores repelling xanthophores
                if xanthophores > melanophores:  # If there are more xanthophores nearby, melanophores may die
                    new_grid[x, y] = EMPTY
            elif grid[x, y] == XANTHOPHORE:
                # Xanthophores repelling melanophores
                if melanophores > xanthophores:  # If there are more melanophores nearby, xanthophores may die
                    new_grid[x, y] = EMPTY
            else:
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

# Display the animation
plt.title("Zebrafish Stripe Formation")
plt.show()