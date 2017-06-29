# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import numpy as np
import functools
import copy

maze_path = 'maze.jpg'

gain = 1 / 8
wall = 255
space = 0
end_point = 170
road = 85


def get_maze(path):
    maze = mpimg.imread(path)
    maze = maze[48:-40, 50:-50, :]
    maze = np.mean(misc.imresize(maze, gain), axis=2)
    maze = np.uint8((120 >= maze) & (maze > 50)) * wall + \
           np.uint8(maze > 120) * end_point + \
           np.uint8(50 >= maze) * space
    return maze


maze = get_maze(maze_path)
fig = plt.figure()
ax = fig.add_subplot(111)

print("Solving by Value Iteration")
end = np.where(maze == end_point)
end = (end[0][0], end[1][0])

actions = {(0, 1), (1, 0), (0, -1), (-1, 0)}


def get_next_state(state, action):
    next_state = (state[0] + action[0], state[1] + action[1])
    try:
        if maze[next_state] == space or maze[next_state] == end_point:
            return next_state
        else:
            return None
    except:
        return None


def get_reward(state, action):
    next_state = get_next_state(state, action)
    if next_state is None:
        return -float('Inf')
    else:
        r = 1 if next_state == end else 0
        return r


gamma = 0.995

V = np.zeros(maze.shape, dtype=float)
changed_set = {(i, j) for i in range(maze.shape[0]) for j in range(maze.shape[1]) if maze[i, j] == space}

i = 0
while True:
    max_delta = 0
    new_changed = set()
    for s in changed_set:
        new_s = max(get_reward(s, a) + gamma * V[get_next_state(s, a)] for a in actions \
                    if get_next_state(s, a) is not None)
        if new_s != V[s]:
            new_changed.add(s)
            V[s] = new_s
    if len(new_changed) == 0:
        break

    changed_set = new_changed
    changed_set = functools.reduce(lambda x, y: x | y, ({get_next_state(i, a) for a in actions} for i in changed_set))
    changed_set -= {None, end}

    i += 1
    if i % 20 == 0:
        ax.imshow(V, cmap=plt.cm.gray)
        plt.pause(0.01)

print("Solved")

def print_path(start, maze, path):
    maze = copy.copy(maze)
    maze[start] = end_point
    for i in path:
        maze[i] = road
    ax.imshow(maze, cmap=plt.cm.gray_r)
    plt.pause(0.001)


try:
    def get_axis(event):
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if maze[iy, ix] == space:
            path = []
            start = (iy, ix)
            s = start
            while True:
                v, a = max((get_reward(s, a) + gamma * V[get_next_state(s, a)], a) for a in actions \
                           if get_next_state(s, a) is not None)
                s = get_next_state(s, a)
                if s != end:
                    path.append(s)
                else:
                    break
            print_path(start, maze, path)
        return


    cid = fig.canvas.mpl_connect('button_press_event', get_axis)
    plt.pause(float('Inf'))
except:
    pass
