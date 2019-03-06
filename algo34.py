from PIL import Image, ImageDraw
import random
import math
import numpy

RANDOM_POINTS = 100
BOX_LEN = 5

def color_sum(adj_matrix, colors):
    result = 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            result += ((colors[i][1][3] - colors[j][1][3])/(256 * 16)) ** 2
    return result

def generate_voronoi_diagram(width, height, cells):
    image = Image.new("RGB", (width, height))
    putpixel = image.putpixel
    imgx, imgy = image.size
    nx = []
    ny = []
    nc = []
    country_color = {}

    # get countries
    countries = []
    for i in range(len(cells)):
        if cells[i][3] not in countries:
            countries.append(cells[i][3])
    num_countries = len(countries)

    # sequential of colors
    colors = []
    spectrue = 255 ** 3
    for i in range(num_countries):
        color = (i + 3) * spectrue / (num_countries + 6)
        r = color/(255 ** 2)
        g = (color % (255 ** 2)) / 255
        b = (color % (255 ** 2)) % 255
        colors.append((r, g, b, i))
    print(colors)

    # adj matrix for coloring
    adj_matrix = []
    degree_matrix = []
    lap_matrix = []
    for i in range(num_countries):
        adj_row = []
        degree_row = []
        lap_row = []
        for j in range(num_countries):
            adj_row.append(0)
            degree_row.append(0)
            lap_row.append(0)
        adj_matrix.append(adj_row)
        degree_matrix.append(degree_row)
        lap_matrix.append(lap_row)

    # outer boundary threshold
    threshold = (width + height) / (4 + 2 * math.floor(math.sqrt(len(cells))))

    # random points outside
    added = []
    while len(added) < RANDOM_POINTS:
        cand_x = random.randrange(width)
        cand_y = random.randrange(height)
        can_add = True
        for i in range(len(cells)):
            d = (cells[i][0] - cand_x) ** 2 + (cells[i][1] - cand_y) ** 2
            if (d < threshold ** 2):
                can_add = False
                break
        if can_add:
            added.append([cand_x, cand_y])
    for i in range(len(added)):
        nx.append(added[i][0])
        ny.append(added[i][1])
        nc.append(-1)

    # original points
    for i in range(len(cells)):
        center_x = cells[i][0]
        center_y = cells[i][1]
        country = cells[i][3]
        nx.append(center_x)
        ny.append(center_y)
        nc.append(country)

    # calculate color adj matrix
    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx-1, imgy-1)
            j = -1
            dmin2 = math.hypot(imgx-1, imgy-1)
            j2 = -1
            for i in range(len(nx)):
                d = math.hypot(nx[i]-x, ny[i]-y)
                if d < dmin:
                    dmin2 = dmin
                    j2 = j
                    dmin = d
                    j = i
                elif d < dmin2:
                    dmin2 = d
                    j2 = i
            if (nc[j] != -1 and nc[j2] != -1):
                country1_idx = countries.index(nc[j])
                country2_idx = countries.index(nc[j2])
                adj_matrix[country1_idx][country2_idx] = 1
                adj_matrix[country2_idx][country1_idx] = 1
    print(adj_matrix)
    # calculate degree matrix
    for i in range(num_countries):
        degree = 0
        for j in range(num_countries):
            degree += adj_matrix[i][j]
        degree_matrix[i][i] = degree
    # calculate Laplacian matrix
    for i in range(num_countries):
        for j in range(num_countries):
            lap_matrix[i][j] = degree_matrix[i][j] - adj_matrix[i][j]

    # find color with this Laplacian matrix, re-organize colors based on eigen vector
    eig_value, eig_vector = numpy.linalg.eig(lap_matrix)
    big_value = eig_value[0]
    big_vector = eig_vector[0]
    for i in range(num_countries):
        if (big_value < eig_value[i]):
            big_vector = eig_vector[i]
            big_value = eig_value[i]

    # permutate colors
    eig_colors = []
    for i in range(num_countries):
        eig_colors.append((big_vector[i], colors[i]))
    eig_colors.sort(key=lambda x: x[0])
    # greedy
    print(eig_colors)
    current_sum = color_sum(adj_matrix, eig_colors)
    for i in range(num_countries):
        for j in range(num_countries):
            if i == j:
                continue
            if adj_matrix[i][j] == 0:
                continue
            tmp = eig_colors[i]
            eig_colors[i] = eig_colors[j]
            eig_colors[j] = tmp
            new_sum = color_sum(adj_matrix, eig_colors)
            if (new_sum < current_sum):
                tmp = eig_colors[i]
                eig_colors[i] = eig_colors[j]
                eig_colors[j] = tmp
            else:
                current_sum += new_sum
    print(eig_colors)

    # assign colors
    for i in range(num_countries):
        country_color[countries[i]] = eig_colors[i][1][:-1]
    print(country_color)

    # inner boxes
    for i in range(len(cells)):
        center_x = cells[i][0]
        center_y = cells[i][1]
        country = cells[i][3]

        radius = width
        for j in range(len(cells)):
            if i == j: continue
            d = math.hypot(cells[j][0] - center_x, cells[j][1] - center_y)
            if radius > d:
                radius = d
        radius = math.floor(math.sqrt(((radius/2) ** 2) / 2))

        for j in range(BOX_LEN):
            nx.append(center_x - radius)
            ny.append(center_y - radius + radius * j / BOX_LEN - j%2)
            nc.append(country)
            nx.append(center_x + radius)
            ny.append(center_y - radius + radius * j / BOX_LEN - j%2)
            nc.append(country)
            nx.append(center_x - radius + radius * j / BOX_LEN - j%2)
            ny.append(center_y - radius)
            nc.append(country)
            nx.append(center_x - radius + radius * j / BOX_LEN - j%2)
            ny.append(center_y + radius)
            nc.append(country)

    # actual plot
    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx-1, imgy-1)
            j = -1
            for i in range(len(nx)):
                d = math.hypot(nx[i]-x, ny[i]-y)
                if d < dmin:
                    dmin = d
                    j = i
            if (nc[j] == -1):
                putpixel((x, y), (255, 255, 255))
            else:
                putpixel((x, y), country_color[nc[j]])

    return image



# make up nodes
width = 500
height = 500
cells = []
for i in range(10):
    cells.append([random.randrange(width), random.randrange(height), "node%d" % i, i%6])
    print(cells[-1])
# cells.append([300, 200])
# cells.append([200, 250])
# cells.append([300, 300])

image = generate_voronoi_diagram(width, height, cells)

for i in range(len(cells)):
    ImageDraw.Draw(image).text((cells[i][0], cells[i][1]), cells[i][2], (0, 0, 0))
image.save("a.png", "PNG")
image.show()
