# required imports
import numpy as np
from copy import deepcopy
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# testing related imports
from gov_rek.envs.common.entities import SimpleGridRoadWorld

def plot_kernel(kernel_arr, title):
    x = range(kernel_arr.shape[0])
    y = range(kernel_arr.shape[1])
    # `plot_surface` expects `x` and `y` data to be 2D
    X, Y = np.meshgrid(x, y) 
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, kernel_arr, rstride=1,
                           cstride=1, alpha=0.65, cmap=cm.coolwarm)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z', fontsize=16)
    ax.set_zlim(np.min(kernel_arr), np.max(kernel_arr))
    ax.set_title(title, fontsize=20)
    plt.show()

def irregular_gradient_kernel(world_map, pos_slope_flag=True, left_pos_vals=True, slope_gradient=0.01):
    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1]))
    for i in range(0,world_map.shape[0]):
        for j in range(0,world_map.shape[1]):
            if pos_slope_flag:
                if left_pos_vals:
                    if i>=j:
                        gradient_kernel[i][j] = slope_gradient*(i+j)
                    else:
                        gradient_kernel[i][j] = slope_gradient*((world_map.shape[0]-i)-(world_map.shape[1]-j))
                else:
                    if i<=j:
                        gradient_kernel[i][j] = slope_gradient*(j+i)
                    else:
                        gradient_kernel[i][j] = slope_gradient*((world_map.shape[1]-j)-(world_map.shape[0]-i))
            else:
                if left_pos_vals:
                    if i>=j:
                        gradient_kernel[i][j] = slope_gradient*((world_map.shape[0]-i)+(world_map.shape[1]-j))
                    else:
                        gradient_kernel[i][j] = slope_gradient*((world_map.shape[0]-i)-(world_map.shape[1]-j))
                else:
                    if i<=j:
                        gradient_kernel[i][j] = slope_gradient*((world_map.shape[1]-j)+(world_map.shape[0]-i))
                    else:
                        gradient_kernel[i][j] = slope_gradient*((world_map.shape[1]-j)-(world_map.shape[0]-i))

    return gradient_kernel

def regular_gradient_kernel(world_map, pos_slope_flag=True, left_pos_vals=True, slope_gradient=0.01):
    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1]))
    for i in range(0,world_map.shape[0]):
        for j in range(0,world_map.shape[1]):
            if pos_slope_flag:
                if left_pos_vals:
                    if i>=j:
                        gradient_kernel[i][j] = slope_gradient*(i+j)
                    else:
                        gradient_kernel[i][j] = 0
                else:
                    if i<=j:
                        gradient_kernel[i][j] = slope_gradient*(j-i)
                    else:
                        gradient_kernel[i][j] = 0
            else:
                if left_pos_vals:
                    if i>=j:
                        gradient_kernel[i][j] = 0
                    else:
                        gradient_kernel[i][j] = slope_gradient*(j-i)
                else:
                    if i<=j:
                        gradient_kernel[i][j] = slope_gradient*((world_map.shape[1]-j)+(world_map.shape[0]-i))
                    else:
                        gradient_kernel[i][j] = 0

    gradient_kernel_copy = deepcopy(gradient_kernel)
    if pos_slope_flag:
        if left_pos_vals:
            gradient_kernel = gradient_kernel + np.transpose(gradient_kernel_copy)
        else:
            gradient_kernel = gradient_kernel + np.transpose(gradient_kernel_copy)
    else:
        if left_pos_vals:
            gradient_kernel = gradient_kernel + np.transpose(gradient_kernel_copy)
            gradient_kernel = np.rot90(gradient_kernel)
        else:
            gradient_kernel = gradient_kernel + np.transpose(gradient_kernel_copy)

    return gradient_kernel


def main():
    road_world = SimpleGridRoadWorld(size=15, default_world=True, num_blockers=0)
    print(road_world.world, road_world.world.shape)
    plot_kernel(irregular_gradient_kernel(road_world.world, pos_slope_flag=True, left_pos_vals=False, slope_gradient=0.01), title='')
    

if __name__ == '__main__':
    main()
