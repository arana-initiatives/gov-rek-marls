# required imports
import numpy as np
from gov_rek.envs.governance.utils import *

# testing related imports
from gov_rek.envs.common.entities import SimpleGridDroneWorld

def ellipsoid_kernel(world_map, offsets = None, radial_params= (5, 2, 2)):
    # offset and radial params are in {z, y, x} numpy formats
    # instead of usually intuitive {x, y, z} coordinate formats
    if offsets is None:
        offsets = (int(world_map.shape[0]/2), int(world_map.shape[1]/2), int(world_map.shape[2]/2))
    
    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1], world_map.shape[2]))
    
    for i in range(0, gradient_kernel.shape[0]):
        for j in range(0, gradient_kernel.shape[1]):
            for k in range(0, gradient_kernel.shape[2]):
                if (k - offsets[0])**2 / radial_params[0]**2 + \
                   (j - offsets[1])**2 / radial_params[1]**2 + \
                   (i - offsets[2])**2 / radial_params[2]**2 <= 1:
                    gradient_kernel[i][j][k] = 1

    return gradient_kernel

def hyperboloid_kernel(world_map, offsets = None, radial_params= (2, 2, 2), axes = 2):
    # axes component represents the hyperboloid surface's orientation
    if offsets is None:
        offsets = (int(world_map.shape[0]/2), int(world_map.shape[1]/2), int(world_map.shape[2]/2))

    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1], world_map.shape[2]))

    for i in range(0, gradient_kernel.shape[0]):
        for j in range(0, gradient_kernel.shape[1]):
            for k in range(0, gradient_kernel.shape[2]):
                if axes == 0:
                    if -1 * (k - offsets[0])**2 / radial_params[0]**2 + \
                    (j - offsets[1])**2 / radial_params[1]**2 + \
                    (i - offsets[2])**2 / radial_params[2]**2 <= 1:
                        gradient_kernel[i][j][k] = 1
                elif axes == 1:
                    if (k - offsets[0])**2 / radial_params[0]**2 - \
                    (j - offsets[1])**2 / radial_params[1]**2 + \
                    (i - offsets[2])**2 / radial_params[2]**2 <= 1:
                        gradient_kernel[i][j][k] = 1
                else:
                    if (k - offsets[0])**2 / radial_params[0]**2 + \
                    (j - offsets[1])**2 / radial_params[1]**2 - \
                    (i - offsets[2])**2 / radial_params[2]**2 <= 1:
                        gradient_kernel[i][j][k] = 1

    return gradient_kernel

def hyperboloid_kernel(world_map, offsets = None, radial_params= (2, 2, 2), axes = 2):
    # axes component represents the hyperboloid surface's orientation
    if offsets is None:
        offsets = (int(world_map.shape[0]/2), int(world_map.shape[1]/2), int(world_map.shape[2]/2))

    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1], world_map.shape[2]))

    for i in range(0, gradient_kernel.shape[0]):
        for j in range(0, gradient_kernel.shape[1]):
            for k in range(0, gradient_kernel.shape[2]):
                if axes == 0:
                    if -1 * (k - offsets[0])**2 / radial_params[0]**2 + \
                    (j - offsets[1])**2 / radial_params[1]**2 + \
                    (i - offsets[2])**2 / radial_params[2]**2 <= 1:
                        gradient_kernel[i][j][k] = 1
                elif axes == 1:
                    if (k - offsets[0])**2 / radial_params[0]**2 - \
                    (j - offsets[1])**2 / radial_params[1]**2 + \
                    (i - offsets[2])**2 / radial_params[2]**2 <= 1:
                        gradient_kernel[i][j][k] = 1
                else:
                    if (k - offsets[0])**2 / radial_params[0]**2 + \
                    (j - offsets[1])**2 / radial_params[1]**2 - \
                    (i - offsets[2])**2 / radial_params[2]**2 <= 1:
                        gradient_kernel[i][j][k] = 1

    return gradient_kernel



def main():
    road_world = SimpleGridDroneWorld(size=15, default_world=True, num_blockers=0)
    # print(road_world.world, road_world.world.shape)
    gradient_kernel = hyperboloid_kernel(road_world.world)
    # print(gradient_kernel)
    plot_surface_kernel(rotate_surface(gradient_kernel), title='Sample Kernel Plot Visualization')


if __name__ == '__main__':
    main()

# TODO: 1. addition of reward distribution rotation feature to add more learning flexibility
#       -  computationally expensive step, need to evaluate whether there is need for it or not
