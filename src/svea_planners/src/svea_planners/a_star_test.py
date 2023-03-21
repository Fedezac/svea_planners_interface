#! /usr/bin/env python3

from astar import AStarPlanner, AStarWorld
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import numpy as np

EMPTY_CELL = 0
OBSTACLE_CELL = 1
START_CELL = 2
GOAL_CELL = 3
MOVE_CELL = 4

# create discrete colormap
cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
bounds = [EMPTY_CELL, OBSTACLE_CELL, START_CELL, GOAL_CELL, MOVE_CELL ,MOVE_CELL + 1]
norm = colors.BoundaryNorm(bounds, cmap.N)

def plot_grid(data, saveImageName):
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

    ax.set_xticks(np.arange(0.5, rows, 1));
    ax.set_yticks(np.arange(0.5, cols, 1));
    plt.tick_params(axis='both', which='both', bottom=False,   
                    left=False, labelbottom=False, labelleft=False) 
    fig.set_size_inches((8.5, 11), forward=False)
    plt.savefig(saveImageName + ".png", dpi=500)

if __name__ == "__main__":
    rows = 20
    cols = 20
    # Randomly create 20 different grids
    for i in range(0, 10):
        data = np.zeros(rows * cols).reshape(rows, cols)
        start_x = random.randint(0, rows - 1)
        start_y = random.randint(0, cols - 1)

        goal_x = random.randint(0, rows - 1)
        # Dont want the start and end positions to be the same
        # so keep changing the goal x until its different. 
        # If X is different dont need to check Y
        while goal_x is start_x:
            goal_x = random.randint(0, rows - 1)
        goal_y = random.randint(0, cols - 1)

        # Generate random obstacles (supposing square map)
        obstacles = np.asarray([divmod(ele, rows) for ele in random.sample(range((rows) * (rows)), 30)])
        # Set gridmap cells to the corresponding color
        data[obstacles[:,0], obstacles[:, 1]] = OBSTACLE_CELL
        # Add obstacles' radius (set as 1) to each of them (necessary in AStarWorld)
        obstacles = [list(tup) + [1] for tup in obstacles]
        
        # Create a world with cells big 1 (for both x and y dimensions), with limits rows and cols, and obstacles
        world = AStarWorld(delta=[1, 1], limit=[[0, rows], [0, cols]], obstacles=obstacles)
        planner = AStarPlanner(world, [start_x, start_y], [goal_x, goal_y])
        path = np.asarray(planner.create_path())

        # Assign to each grid cell a color
        data[path[:, 0], path[:, 1]] = MOVE_CELL
        data[start_x, start_y] = START_CELL
        data[goal_x, goal_y] = GOAL_CELL
        
        plot_grid(data, "grid_" + str(i))