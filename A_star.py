"""
只需要改3个地方即可
1、起点
2、终点
3、安全区域
"""

import queue
import numpy as np
import matplotlib.image as mp
import matplotlib.pyplot as plt


def Heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def Neighbors(current):
    a = [current[0], current[1] + 1]
    b = [current[0], current[1] - 1]
    c = [current[0] + 1, current[1]]
    d = [current[0] - 1, current[1]]
    e = [current[0] - 1, current[1] - 1]
    f = [current[0] + 1, current[1] + 1]
    g = [current[0] - 1, current[1] + 1]
    h = [current[0] + 1, current[1] - 1]
    return [a, b, c, d, e, f, g, h]


def A_star_search(start, goal):
    frontier = queue.PriorityQueue()
    frontier.put([0, start])
    cost_so_far = {}
    came_from = {}
    cost_so_far[start] = 0
    came_from[start] = start

    while not frontier.empty():
        current = frontier.get()
        current = current[1]

        if current == goal:
            print('找到最短路径')
            break

        for next in Neighbors(current):
            next = tuple(next)
            new_cost = cost_so_far[current] + 1  # 当前待见
            if (next not in cost_so_far or new_cost < cost_so_far[next]) and image[next] == safe:
                cost_so_far[next] = new_cost
                priority = new_cost + Heuristic(goal, next)  # 当前代价+预估代价
                frontier.put([priority, next])
                came_from[next] = current

    return came_from, cost_so_far


"""
注意！！！
正常情况下画出来的图是倒的，使用了np.flipud()，虽然图像可以回正，但是相对的对数据做出改变时也需要对调x, y
同时plt.imshow(origin='lower')，在于lower进行匹配的时候也会有一些差异，需要注意调试
"""

image = mp.imread('map3.png')
im = image
im = np.flipud(im)
im = np.array(im)

image = image[:, :, 0]
image = np.flipud(image)  # 矩阵上下颠倒
image = np.array(image)

safe = image[50, 50]  # 障碍物位置x, y最好一样，这样就不需要考虑图像倒置,这里应该指的是左下角，原本指的是左上角
start = (41, 68)  # 起点画图时显示为(y, x)
goal = (221, 200)  # 终点画图时显示为(y, x)
came_from, cost_so_far = A_star_search(start, goal)  # 当前方块到之前方块的一个映射，当前代价
path = [goal]
i = came_from[goal]
for x in range(len(came_from)):  # 实际上循环不到len(came_from)次
    path.append(i)
    a = came_from[i]
    path.append(a)
    i = came_from[a]

for i in range(len(path)):
    x = path[i][0]
    y = path[i][1]
    image[x, y] = 0.2
    im[x, y] = [0.92941177, 0.10980392, 0.14117648, 1.0]
plt.xlabel("x")
plt.ylabel("y")
plt.imshow(im, cmap="hot", origin='lower')
plt.colorbar()
plt.show()