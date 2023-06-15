# required imports
import numpy as np
from copy import deepcopy
from gov_rek.envs.governance.utils import *

# testing related imports
from gov_rek.envs.common.entities import SimpleGridRoadWorld

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
            if i>=j:
                gradient_kernel[i][j] = slope_gradient*(i+j)
            else:
                gradient_kernel[i][j] = 0

    gradient_kernel_copy = deepcopy(gradient_kernel)
    gradient_kernel = np.tril(gradient_kernel, k=-1) + np.transpose(gradient_kernel_copy)

    if pos_slope_flag:
        if left_pos_vals:
            pass
        else:
            gradient_kernel = np.rot90(gradient_kernel)
    else:
        if left_pos_vals:
            gradient_kernel = np.rot90(np.rot90(gradient_kernel))
        else:
            gradient_kernel = np.rot90(np.rot90(np.rot90(gradient_kernel)))

    return gradient_kernel

def splitted_gradient_kernel(world_map, pos_slope_flag=True, slope_gradient=0.01):
    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1]))
    for i in range(0,world_map.shape[0]):
        for j in range(0,world_map.shape[1]):
            if i<=j:
                gradient_kernel[i][j] = slope_gradient*(j-i)
            else:
                gradient_kernel[i][j] = 0

    gradient_kernel_copy = deepcopy(gradient_kernel)
    gradient_kernel = gradient_kernel + np.transpose(gradient_kernel_copy)

    if not pos_slope_flag:
            gradient_kernel = np.rot90(gradient_kernel)

    return gradient_kernel

def inverse_radial_kernel(world_map, agent_name, slope_gradient=0.05):
    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1]))
    agent_idx_x, agent_idx_y = np.where(world_map == agent_name)

    for i in range(0,world_map.shape[0]):
        for j in range(0,world_map.shape[1]):
            if i == agent_idx_x[0] and j == agent_idx_y[0]:
                gradient_kernel[i][j] = 0
            else:
                gradient_kernel[i][j] = np.power((slope_gradient/(i+1) + slope_gradient/(j+1)), 1/2)
    
    return gradient_kernel

def squared_exponential_kernel(world_map, agent_name, size=5.0, length_scale=5.0):
    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1]))
    agent_idx_x, agent_idx_y = np.where(world_map == agent_name)

    for i in range(0,world_map.shape[0]):
        for j in range(0,world_map.shape[1]):
            if i == agent_idx_x[0] and j == agent_idx_y[0]:
                gradient_kernel[i][j] = 0
            else:
                gradient_kernel[i][j] = size * size * np.exp( -((i-agent_idx_x[0])**2 + \
                                                  (j-agent_idx_y[0])**2) / (2 * length_scale**2) )
    
    return gradient_kernel


def rational_quadratic_kernel(world_map, agent_name, size=5.0, length_scale=5.0, alpha=5.0):
    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1]))
    agent_idx_x, agent_idx_y = np.where(world_map == agent_name)

    for i in range(0,world_map.shape[0]):
        for j in range(0,world_map.shape[1]):
            if i == agent_idx_x[0] and j == agent_idx_y[0]:
                gradient_kernel[i][j] = 0
            else:
                gradient_kernel[i][j] = size * size * np.power(np.exp( -(1 + (i-agent_idx_x[0])**2 + \
                                                  (j-agent_idx_y[0])**2) / (2 * length_scale**2)),1/alpha)
    
    return gradient_kernel


def periodic_kernel(world_map, agent_name, size=5.0, length_scale=5.0, period=2*np.pi):
    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1]))
    agent_idx_x, agent_idx_y = np.where(world_map == agent_name)

    for i in range(0,world_map.shape[0]):
        for j in range(0,world_map.shape[1]):
            if i == agent_idx_x[0] and j == agent_idx_y[0]:
                gradient_kernel[i][j] = 0
            else:
                gradient_kernel[i][j] = size * size * np.exp((-2 / length_scale**2) * (np.sin( np.pi* (np.absolute(i-agent_idx_x[0]) + \
                                                  np.absolute(j-agent_idx_y[0])) / period)**2)) 
    
    return gradient_kernel

def locally_periodic_kernel(world_map, agent_name, size=5.0, length_scale=5.0, period=2*np.pi, alpha=5.0):
    return periodic_kernel(world_map, agent_name, size, length_scale, period) * \
           rational_quadratic_kernel(world_map, agent_name, size, length_scale, alpha)


def main():
    road_world = SimpleGridRoadWorld(size=15, default_world=True, num_blockers=0)
    print(road_world.world, road_world.world.shape)
    plot_planar_kernel(normalize_rewards(locally_periodic_kernel(road_world.world, agent_name=1),3), title='Sample Kernel Plot Visualization')


if __name__ == '__main__':
    main()
