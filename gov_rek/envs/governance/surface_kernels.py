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
    # axes component represents the hyperboloid surface's orientation in {z, y, x} system
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
    # axes component represents the hyperboloid surface's orientation in {z, y, x} system
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

def elliptic_paraboloid_kernel(world_map, offsets = None, radial_params= (2, 2, 2), axes = 2):
    # axes component represents the hyperboloid surface's orientation in {z, y, x} system
    if offsets is None:
        offsets = (int(world_map.shape[0]/2), int(world_map.shape[1]/2), int(world_map.shape[2]/2))

    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1], world_map.shape[2]))

    for i in range(0, gradient_kernel.shape[0]):
        for j in range(0, gradient_kernel.shape[1]):
            for k in range(0, gradient_kernel.shape[2]):
                if axes == 0:
                    if (j - offsets[1])**2 / radial_params[1]**2 + \
                    (i - offsets[2])**2 / radial_params[2]**2 <= (k - offsets[0]) / radial_params[0]:
                        gradient_kernel[i][j][k] = 1
                elif axes == 1:
                    if (k - offsets[0])**2 / radial_params[0]**2 + \
                    (i - offsets[2])**2 / radial_params[2]**2 <= (j - offsets[1]) / radial_params[1]:
                        gradient_kernel[i][j][k] = 1
                else:
                    if (k - offsets[0])**2 / radial_params[0]**2 + \
                    (j - offsets[1])**2 / radial_params[1]**2 <= (i - offsets[2]) / radial_params[2]:
                        gradient_kernel[i][j][k] = 1

    return gradient_kernel

def hyperbolic_paraboloid_kernel(world_map, offsets = None, radial_params= (2, 2, 2), axes = 2):
    # axes component represents the hyperboloid surface's orientation in {z, y, x} system
    if offsets is None:
        offsets = (int(world_map.shape[0]/2), int(world_map.shape[1]/2), int(world_map.shape[2]/2))

    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1], world_map.shape[2]))

    for i in range(0, gradient_kernel.shape[0]):
        for j in range(0, gradient_kernel.shape[1]):
            for k in range(0, gradient_kernel.shape[2]):
                if axes == 0:
                    if (j - offsets[1])**2 / radial_params[1]**2 - \
                    (i - offsets[2])**2 / radial_params[2]**2 <= (k - offsets[0]) / radial_params[0]:
                        gradient_kernel[i][j][k] = 1
                elif axes == 1:
                    if (k - offsets[0])**2 / radial_params[0]**2 - \
                    (i - offsets[2])**2 / radial_params[2]**2 <= (j - offsets[1]) / radial_params[1]:
                        gradient_kernel[i][j][k] = 1
                else:
                    if (k - offsets[0])**2 / radial_params[0]**2 - \
                    (j - offsets[1])**2 / radial_params[1]**2 <= (i - offsets[2]) / radial_params[2]:
                        gradient_kernel[i][j][k] = 1

    return gradient_kernel

def torus_kernel(world_map, offsets = None, torus_params= (4, 2), axes = 2):
    # axes component represents the hyperboloid surface's orientation in {z, y, x} system
    if offsets is None:
        offsets = (int(world_map.shape[0]/2), int(world_map.shape[1]/2), int(world_map.shape[2]/2))

    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1], world_map.shape[2]))

    for i in range(0, gradient_kernel.shape[0]):
        for j in range(0, gradient_kernel.shape[1]):
            for k in range(0, gradient_kernel.shape[2]):
                if axes == 1:
                    if ((k - offsets[2])**2 + (j - offsets[1])**2 + (i - offsets[0])**2 + \
                    torus_params[0]**2 - torus_params[1]**2)**2 <= 4*(torus_params[0]**2)*((k - offsets[2])**2 + (i - offsets[0])**2):
                        gradient_kernel[i][j][k] = 1
                elif axes == 0:
                    if ((k - offsets[2])**2 + (j - offsets[1])**2 + (i - offsets[0])**2 + \
                    torus_params[0]**2 - torus_params[1]**2)**2 <= 4*(torus_params[0]**2)*((j - offsets[1])**2 + (i - offsets[0])**2):
                        gradient_kernel[i][j][k] = 1
                else:
                    if ((k - offsets[2])**2 + (j - offsets[1])**2 + (i - offsets[0])**2 + \
                    torus_params[0]**2 - torus_params[1]**2)**2 <= 4*(torus_params[0]**2)*((k - offsets[2])**2 + (j - offsets[0])**2):
                        gradient_kernel[i][j][k] = 1

    return gradient_kernel

def genus_ray_kernel(world_map, offsets = None):
    if offsets is None:
        offsets = (int(world_map.shape[0]/2), int(world_map.shape[1]/2), int(world_map.shape[2]/2))
    gradient_kernel = np.zeros((world_map.shape[0], world_map.shape[1], world_map.shape[2]))
    for i in range(0, gradient_kernel.shape[0]):
        for j in range(0, gradient_kernel.shape[1]):
            for k in range(0, gradient_kernel.shape[2]):
                    if (2*(1 - (i - offsets[0])**2)*(j - offsets[1])*((j - offsets[1])**2 - 3*(k - offsets[0])**2) + \
                    ((j - offsets[1])**2 + (k - offsets[2])**2)**2) \
                    <= (9.0*(i - offsets[0])**2-1)*(1-(i-offsets[0])**2):
                        gradient_kernel[i][j][k] = 1

    return gradient_kernel


def main():
    road_world = SimpleGridDroneWorld(size=20, default_world=True, num_blockers=0)
    # Note: offsets can be passed with agent locations to make agent specific kernels
    gradient_kernel = default_genus_kernel(road_world.world)
    plot_surface_kernel(gradient_kernel, title='Sample Kernel Plot Visualization')


if __name__ == '__main__':
    main()

# TODO: 1. addition of reward distribution rotation feature to add more learning flexibility
#       -  computationally expensive step, need to evaluate whether there is need for it or not
