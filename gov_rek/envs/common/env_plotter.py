# function for imports: render_road_agent, render_drone_agents
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def get_env_func(obs_arr):
    env_arr = obs_arr[:-1,:]
    return env_arr


def get_hex_obs_vals(env_obs_samples):
# transform observation samples into hex-values for voxel plots.
    color_dict = {
        '0' : '#F0F8FF55',    # aliceblue
        '1' : '#BCEE6855',    # darkolivegreen2
        '2' : '#6E8B3D66',    # darkolivegreen4
        '3' : '#FFD39B55',    # burlywood1
        '4' : '#FF6A6A55',    # indianred1
        '5' : '#838B8377',    # honeydew4
        # '6' : '#77889966'     # lightslategray
    }
    hex_val_render_list = []
    for env_obs in env_obs_samples:
        colors = env_obs
        colors = colors.astype(str)
        colors[colors == '0'] = color_dict['0']
        colors[colors == '1'] = color_dict['1']
        colors[colors == '2'] = color_dict['2']
        colors[colors == '3'] = color_dict['3']
        colors[colors == '4'] = color_dict['4']
        colors[colors == '5'] = color_dict['5']
        # colors[colors == '6'] = color_dict['6']
        hex_val_render_list.append(colors)
    return hex_val_render_list


def render_drone_agent(obs_list, file_path, env_out=False):
    if file_path == None or len(file_path) < 4:
        raise ValueError("file_path length must be greater than 4 characters, including the '.mp4' extension.")
    if file_path.split('.')[-1] > 'mp4':
        raise ValueError("file extension must be '.mp4', other format not supported.")
    # practical observation, obs_list length must be greater than 'num_secs*fps' value for saving animations
    fps = 5
    num_secs = 10
    snapshots = get_hex_obs_vals(obs_list)
    if env_out:
        snapshots = [ get_env_func(obs) for obs in snapshots ]


    def make_axes(grid=False):
        fig = plt.figure(figsize=(5,5) )
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(grid)
        return ax, fig

    ax,fig = make_axes(True)
    colors = snapshots[0]

    def explode(data):
        shape_arr = np.array(data.shape)
        size = shape_arr[:3]*2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    print(snapshots[0].shape[0] , snapshots[0].shape[1], snapshots[0].shape[2])
    filled = explode(np.ones((snapshots[0].shape[0], snapshots[0].shape[1], snapshots[0].shape[2])))
    colors = explode(colors)
    # supplementary color cadetblue: '#5F9EA077' for reference
    ax.voxels(filled, edgecolors='#5F9EA055', facecolors=colors, shade=False)
    im = plt.show()

    def render_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        # supplementary color cadetblue: '#5F9EA077' for reference
        ax.voxels(filled, edgecolors='#5F9EA055', facecolors=explode(snapshots[i]), shade=False)
        plt.show()
        return []

    anim = animation.FuncAnimation(
                                   fig, 
                                   render_func, 
                                   frames = num_secs * fps,
                                   interval = 1000 / fps, # duration in milliseconds
                                   )
    # preparing the 3d rendered output is a time consuming step compared to 2d render plots
    anim.save(file_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    print("environment rendered output loaded onto the %r file path." % file_path)


def render_road_agent(obs_list, file_path, env_out=False):
    if file_path == None or len(file_path) < 4:
        raise ValueError("file_path length must be greater than 4 characters, including the '.mp4' extension.")
    if file_path.split('.')[-1] > 'mp4':
        raise ValueError("file extension must be '.mp4', other format not supported.")
    # practical observation, obs_list length must be greater than 'num_secs*fps' value for saving animations
    fps = 3
    num_secs = 15
    snapshots = obs_list
    if env_out:
        snapshots = [ get_env_func(obs) for obs in obs_list ]
        
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(4,5))
    color_map = colors.ListedColormap(["lightsteelblue", "chocolate",
                                      "tomato", "navajowhite", "yellowgreen",
                                      "lightpink", "ivory"])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm_vals = colors.BoundaryNorm(bounds, color_map.N)
    a = snapshots[0]
    im = plt.imshow(a, cmap=color_map, norm=norm_vals)
    def render_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        im.set_array(snapshots[i])
        return [im]
    anim = animation.FuncAnimation(
                               fig, 
                               render_func, 
                               frames = num_secs * fps,
                               interval = 1000 / fps, # duration in milliseconds
                               )
    anim.save(file_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    print("environment rendered output loaded onto the %r file path." % file_path)

    # second alternative for saving the output as 'mp4' file
    # writer_video = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
    # anim.save(file_path, writer = writer_video)

    # third alternative for saving the output as 'gif' file
    # writer_gif = animation.PillowWriter(fps=fps) 
    # anim.save(file_path, writer = writer_gif)


def rand_grid_generator(world_type, grid_size):
    VALID_WORLDS = {'road', 'drone'}
    GRID_MAX_VALUE = 20
    if world_type not in VALID_WORLDS:
        raise ValueError("world_type must be one of %r." % VALID_WORLDS)
    if grid_size > 20:
        raise ValueError("grid_size must not be greater than %r." % GRID_MAX_VALUE)
     
    if world_type == 'road':
        world_arr = np.zeros((grid_size + 1, grid_size), dtype=np.int64)
    else:
        world_arr = np.zeros((grid_size + 1, grid_size, grid_size), dtype=np.int64)

    if world_type == 'road':
        for i in range(world_arr.shape[0]):
            for j in range(world_arr.shape[1]):
                world_arr[i][j] = random.choice(range(0,5))
    else:
        for i in range(world_arr.shape[0]):
            for j in range(world_arr.shape[1]):
                for k in range(world_arr.shape[2]):
                    world_arr[i][j][k] = random.choice(range(0,5))

    return world_arr


def rand_obs_generator(episode_count, world_type, grid_size):
    grid_max_value = 100
    if episode_count > 100:
        raise ValueError("episode_count must not be greater than %r." % grid_max_value)
    observation_sample_list = []
    world_arr = rand_grid_generator(world_type, grid_size)
    for i in range(episode_count):
        world_prob = random.uniform(0, 1)
        agent_arr = world_arr[-1,:]
        grid_arr = world_arr[:-1,:]
        grid_arr_two = np.rot90(grid_arr)
        grid_arr_three = np.rot90(grid_arr_two)
        grid_arr_four = np.rot90(grid_arr_three)
        if world_prob <= 0.25:
            grid_arr = np.append(grid_arr, [agent_arr], axis=0)
            observation_sample_list.append(grid_arr)
        elif world_prob > 0.25 and world_prob <= 0.50:
            grid_arr_two = np.append(grid_arr_two, [agent_arr], axis=0)
            observation_sample_list.append(grid_arr_two)
        elif world_prob > 0.50 and world_prob <= 0.75:
            grid_arr_three = np.append(grid_arr_three, [agent_arr], axis=0)
            observation_sample_list.append(grid_arr_three)
        else:
            grid_arr_four = np.append(grid_arr_four, [agent_arr], axis=0)
            observation_sample_list.append(grid_arr_four)
    
    return observation_sample_list


def main():
    # testing the road environment visualization function
    obs_lst = rand_obs_generator(episode_count = 50, world_type = 'drone', grid_size = 5)
    render_drone_agent(obs_lst, 'test_drone_render_plot.mp4')
    
    # testing the road environment visualization function
    obs_lst = rand_obs_generator(episode_count = 50, world_type = 'road', grid_size = 3)
    render_road_agent(obs_lst, 'test_road_render_plot.mp4', env_out=True)
    
if __name__ == '__main__':
    main()
