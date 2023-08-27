# required imports
import math as m
import numpy as np
from matplotlib import cm
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def rotate_surface(kernel_surface, offsets = None, rotation_angles=(m.pi/4, 0, m.pi/4)):
    
    if offsets is None:
        offsets = (int(kernel_surface.shape[0]/2), int(kernel_surface.shape[1]/2), int(kernel_surface.shape[2]/2))

    rot_kernel_surface = np.zeros((kernel_surface.shape[0], kernel_surface.shape[1], kernel_surface.shape[2]))
    r_x = np.matrix([[ 1, 0 , 0 ],
                   [ 0, m.cos(rotation_angles[2]),-m.sin(rotation_angles[2])],
                   [ 0, m.sin(rotation_angles[2]), m.cos(rotation_angles[2])]])
    r_y = np.matrix([[ m.cos(rotation_angles[1]), 0, m.sin(rotation_angles[1])],
                   [ 0           , 1, 0           ],
                   [-m.sin(rotation_angles[1]), 0, m.cos(rotation_angles[1])]])
    r_z = np.matrix([[ m.cos(rotation_angles[0]), 0, m.sin(rotation_angles[0])],
                   [ 0           , 1, 0           ],
                   [-m.sin(rotation_angles[0]), 0, m.cos(rotation_angles[0])]])
    R = r_z * r_y * r_x
    
    for i in range(0, kernel_surface.shape[0]):
        for j in range(0, kernel_surface.shape[1]):
            for k in range(0, kernel_surface.shape[2]):
                rot_idx_vals = list(np.array((R * np.array([[k-offsets[2]], [j-offsets[1]], [i-offsets[0]]])).astype(int)).flatten())
                if 0 <= rot_idx_vals[0]+offsets[0] < kernel_surface.shape[0] and \
                   0 <= rot_idx_vals[1]+offsets[1] < kernel_surface.shape[1] and \
                   0 <= rot_idx_vals[2]+offsets[2] < kernel_surface.shape[2] :
                    rot_kernel_surface[rot_idx_vals[2]+offsets[0]][rot_idx_vals[1]+offsets[1]][rot_idx_vals[0]+offsets[2]] = kernel_surface[i][j][k]

    return rot_kernel_surface


def plot_planar_kernel(kernel_arr, title):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    x = range(kernel_arr.shape[0])
    y = range(kernel_arr.shape[1])
    # `plot_surface` expects `x` and `y` data to be 2D
    X, Y = np.meshgrid(x, y) 
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, kernel_arr, rstride=1,
                           cstride=1, alpha=0.65, cmap=cm.coolwarm)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_zlim(np.min(kernel_arr), np.max(kernel_arr))
    ax.set_title(title, fontsize=14)
    plt.show()


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_surface_kernel_voxels(kernel_arr, title):
    # high fidelity visualization: more aesthetics, relatively less utility
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    surface_shape = (kernel_arr > 0).astype(int).astype(str)
    surface_shape[surface_shape == '1'] = '#BCEE6855'
    surface_shape[surface_shape == '0'] = '#00000000'

    ax = plt.figure().add_subplot(projection='3d')

    colors = explode(surface_shape)
    filled = explode(np.ones((surface_shape.shape[0], surface_shape.shape[1], surface_shape.shape[2])))
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    ax.voxels(x, y, z, filled, facecolors=colors)
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=14)
    plt.show()

def plot_surface_kernel(kernel_arr, title):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    surface_shape = kernel_arr > 0
    colors = np.empty(surface_shape.shape, dtype=object)
    colors[surface_shape] = 'chartreuse'
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(surface_shape, facecolors=colors)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=14)
    plt.show()

def normalize_rewards(reward_kernel, max_reward):
    return np.around((reward_kernel*np.sqrt(max_reward))/(np.sum(reward_kernel)*2), decimals=4)
