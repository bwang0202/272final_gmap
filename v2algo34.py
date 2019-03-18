import numpy
from PIL import Image, ImageDraw
import random
import math

from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from os import curdir, sep
import os
import json

HEIGHT = 100
WIDTH = 100
RANDOM_POINTS = 10
THRESHOLD = 5

def get_x(node):
    return node[0]

def get_y(node):
    return node[1]

def get_c(node):
    return node[2]

def voronoi(points, cidx):
    shape=(HEIGHT, WIDTH)
    depthmap = numpy.ones(shape,numpy.float)*1e308
    colormap = numpy.zeros(shape,numpy.int)

    def hypot(X,Y):
        return (X-x)**2 + (Y-y)**2

    for i,(x,y,c) in enumerate(points):
        paraboloid = numpy.fromfunction(hypot, shape)
        idx = -1
        if c in cidx:
            idx = cidx.index(c)
        colormap = numpy.where(paraboloid < depthmap, idx + 1, colormap)
        depthmap = numpy.where(paraboloid < depthmap, paraboloid, depthmap)

    return colormap

def draw_map(colormap, eig_colors):
    shape = colormap.shape

    palette_raw = [[255, 255, 255]]
    for i in range(len(eig_colors)):
        palette_raw.append(eig_colors[i][1][0:3])


    palette = numpy.array(palette_raw)

    colormap = numpy.transpose(colormap)
    print("transpose")

    print(colormap)

    pixels = numpy.empty(colormap.shape+(4,),numpy.int8)
    print(palette[colormap])
    pixels[:,:,0:3] = palette[colormap]
    pixels[:,:,3] = 0xFF

    image = Image.frombytes("RGBA", shape, pixels)
    image.show()
    image.save('voronoi.png')

def gougu(c1, c2):
    return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2

def color_sum(adj_matrix, colors):
    result = 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if (adj_matrix[i][j]):
                result += ((colors[i][1][3] - colors[j][1][3])/(256 * 16)) ** 2
    return result

def gmap(cells):
    # Calculate nodes closest neighbor
    closest = {}
    cidx = []
    for i in range(len(cells)):
        # track countries
        if not get_c(cells[i]) in cidx:
            cidx.append(get_c(cells[i]))
        
        # track distances
        closest[i] = (10000 ** 2, -1)
        for j in range(len(cells)):
            if j == i:
                continue
            dist = gougu(cells[i], cells[j])
            if (dist < closest[i][0]):
                closest[i] = (dist, j)

    # init color matrices
    color_num = len(cidx)
    colors = []
    spectrue = 255 ** 3
    for i in range(color_num):
        color = (i + 3) * spectrue / (color_num + 6)
        r = color/(255 ** 2)
        g = (color % (255 ** 2)) / 255
        b = (color % (255 ** 2)) % 255
        colors.append([r, g, b, i])
    adj_m = numpy.zeros((color_num, color_num), numpy.int)

    # fill in adj_m for colors
    for i in closest:
        c = get_c(cells[i])
        c2 = get_c(cells[closest[i][1]])
        adj_m[cidx.index(c)][cidx.index(c2)] = 1
        adj_m[cidx.index(c2)][cidx.index(c)] = 1
    # get degree matrix
    degree_m = numpy.diag(adj_m.sum(axis=0))
    # get laplacian matrix
    lap_m = degree_m - adj_m
    # get eig value
    eig_value, eig_vector = numpy.linalg.eig(lap_m)
    big_value = eig_value[0]
    big_vector = eig_vector[0]
    for i in range(color_num):
        if (big_value < eig_value[i]):
            big_vector = eig_vector[i]
            big_value = eig_value[i]

    # permutate colors
    eig_colors = []
    for i in range(color_num):
        eig_colors.append((big_vector[i], colors[i]))
    eig_colors.sort(key=lambda x: x[0])

    # greedy
    current_sum = color_sum(adj_m, eig_colors)
    for i in range(color_num):
        for j in range(color_num):
            if i == j:
                continue
            if adj_m[i][j] == 0:
                continue
            tmp = eig_colors[i]
            eig_colors[i] = eig_colors[j]
            eig_colors[j] = tmp
            new_sum = color_sum(adj_m, eig_colors)
            if (new_sum < current_sum):
                tmp = eig_colors[i]
                eig_colors[i] = eig_colors[j]
                eig_colors[j] = tmp

    # ADD Randoms to outer boundary
    random_cells_country = "B029B267"
    remaining = RANDOM_POINTS
    # outer boundary threshold
    threshold = (WIDTH + HEIGHT) / (4 + 2 * math.floor(math.sqrt(len(cells))))
    # Add random points to cells
    while remaining > 0:
        cand_x = random.randrange(WIDTH)
        cand_y = random.randrange(HEIGHT)
        can_add = True
        for i in range(len(cells)):
            d = (cells[i][0] - cand_x) ** 2 + (cells[i][1] - cand_y) ** 2
            if (d < threshold ** 2):
                can_add = False
                break
        if can_add:
            cells.append([cand_x, cand_y, random_cells_country])
            remaining -= 1

    # ADD random boxes points for non-random points

    draw_map(voronoi(cells, cidx), eig_colors)

if __name__ == '__main__':
    cells = [[10,10,"a"],[30,10,"b"],[50,50,"c"],[10,80,"d"],[80,90,"a"]]
    gmap(cells)