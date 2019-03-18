import numpy
from PIL import Image, ImageDraw
import random
import math

from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from os import curdir, sep
import os
import json

HEIGHT = 700
WIDTH = 700
RANDOM_POINTS = 100
BOX_LEN = 5

def build_cell(x, y, c):
    return {"x": x, "y": y, "group": c}

def get_x(node):
    return node["x"]

def get_y(node):
    return node["y"]

def get_c(node):
    return node["group"]

def voronoi(points, cidx):
    shape=(HEIGHT, WIDTH)
    depthmap = numpy.ones(shape,numpy.float)*1e308
    colormap = numpy.zeros(shape,numpy.int)

    def hypot(X,Y):
        return (X-x)**2 + (Y-y)**2

    for i in range(len(points)):
        x = get_x(points[i])
        y = get_y(points[i])
        c = get_c(points[i])
        paraboloid = numpy.fromfunction(hypot, shape)
        idx = -1
        if c in cidx:
            idx = cidx.index(c)
        colormap = numpy.where(paraboloid < depthmap, idx + 1, colormap)
        depthmap = numpy.where(paraboloid < depthmap, paraboloid, depthmap)

    return colormap

def draw_map(colormap, eig_colors, names):
    shape = colormap.shape
    path = "voronoi.png"

    palette_raw = [[255, 255, 255]]
    for i in range(len(eig_colors)):
        palette_raw.append(eig_colors[i][1][0:3])


    palette = numpy.array(palette_raw)

    colormap = numpy.transpose(colormap)


    pixels = numpy.empty(colormap.shape+(4,),numpy.int8)
    pixels[:,:,0:3] = palette[colormap]
    pixels[:,:,3] = 0xFF

    image = Image.frombytes("RGBA", shape, pixels)
    for i in range(len(names)):
        ImageDraw.Draw(image).text((names[i][0], names[i][1]), names[i][2], (0,0,0))
    image.save(path)
    image.show()

    return os.path.join(os.getcwd(), path)

def gougu(c1, c2):
    return (get_x(c1) - get_x(c2)) ** 2 + (get_y(c1) - get_y(c2)) ** 2

def color_sum(adj_matrix, colors):
    result = 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if (adj_matrix[i][j]):
                result += ((colors[i][1][3] - colors[j][1][3])/(256 * 16)) ** 2
    return result

def gmap(cells):
    # pass in name labels
    names = []
    for i in range(len(cells)):
        names.append([get_x(cells[i]), get_y(cells[i]), cells[i]["id"]])
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
    original_cells_len = len(cells)
    # Add random points to cells
    while remaining > 0:
        cand_x = random.randrange(WIDTH)
        cand_y = random.randrange(HEIGHT)
        can_add = True
        for i in range(original_cells_len):
            d = (get_x(cells[i]) - cand_x) ** 2 + (get_y(cells[i]) - cand_y) ** 2
            if (d < threshold ** 2):
                can_add = False
                break
        if can_add:
            cells.append(build_cell(cand_x, cand_y, random_cells_country))
            remaining -= 1

    # ADD random boxes points for non-random points
    for i in range(original_cells_len):
        center_x = get_x(cells[i])
        center_y = get_y(cells[i])
        country = get_c(cells[i])

        radius = closest[i][0]

        radius = math.floor(math.sqrt(radius) / 3)

        for j in range(BOX_LEN):
            cells.append(build_cell(center_x - radius, center_y - radius + radius * j / BOX_LEN - j%2, country))
            cells.append(build_cell(center_x + radius, center_y - radius + radius * j / BOX_LEN - j%2, country))
            cells.append(build_cell(center_x - radius + radius * j / BOX_LEN - j%2, center_y - radius, country))
            cells.append(build_cell(center_x - radius + radius * j / BOX_LEN - j%2, center_y + radius, country))

    return draw_map(voronoi(cells, cidx), eig_colors, names)

if __name__ == '__main__':
    data = """
    {"nodes":[{"id":"Myriel","group":1,"index":0,"x":322.1193537956844,"y":154.1539242364042,"vy":1.7801813434376934,"vx":-3.3041471826601576},
    {"id":"Napoleon","group":2,"index":1,"x":291.53541619621245,"y":190.269664315252,"vy":3.8538123261918105,"vx":-5.36041664239701}]}
    """
    gmap(json.loads(data)["nodes"])



class myHandler(BaseHTTPRequestHandler):
    
    #Handler for the GET requests
    def do_POST(self):
        try:
            content_len = int(self.headers.getheader('content-length', 0))
            post_body = self.rfile.read(content_len)
            test_data = json.loads(post_body)
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            path = gmap(test_data["nodes"])
            self.wfile.write(path)
            return

        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)

try:
    #cells = []
    #for i in range(10):
    #    cells.append(build_cell(random.randrange(WIDTH), random.randrange(HEIGHT), i%4))
    #gmap(cells)
    #Create a web server and define the handler to manage the
    #incoming request
    server = HTTPServer(('', PORT_NUMBER), myHandler)
    print 'Started httpserver on port ' , PORT_NUMBER
    
    #Wait forever for incoming htto requests
    server.serve_forever()

except KeyboardInterrupt:
    print '^C received, shutting down the web server'
    server.socket.close()