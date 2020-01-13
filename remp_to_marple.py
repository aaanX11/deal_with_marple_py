import gc
import os

import numpy as np

MAX_DIM = 3
MIN_INTEGER = -2147483648

# TODO: m b store object-presence-flag for a z-pile to search in log(nz) (if object is present)  or const (if not)
# TODO: m b alternative or additional index
# TODO: m b numba

# TODO: space --> memmap
# TODO: counts --> dictionary
# TODO: deco
# TODO: test !


def points_loop(space, counts):

    volumes_count = counts[3]
    nx, ny, nz = space.shape
    nx -= 2
    ny -= 2
    nz -= 2

    # returns (or cahces):
    idx = 0
    pts_idx = np.zeros((nx+1, ny+1), dtype=np.int32)
    pts = np.zeros((volumes_count + (nx + 1)*(ny + 1) + (ny + 1)*(nz + 1) + (nx + 1)*(nz + 1)), dtype=(np.uint16, 3))

    # loop over points
    for ix in range(nx+1):
        for iy in range(ny + 1):
            pts_idx[ix, iy] = idx
            for iz in range(nz + 1):
                if not space[ix:ix+1, iy:iy+1, iz:iz+1].all():
                    pts[idx] = (ix, iy, iz)
                    idx += 1

    np.save("pts_idx.npy", pts_idx)
    np.save("pts.npy", pts)

    del pts_idx
    del pts

    counts[0] = idx


def x_edges_loop(space, counts):

    volumes_count = counts[3]
    nx, ny, nz = space.shape
    nx -= 2
    ny -= 2
    nz -= 2

    # returns (or caches):
    idx = 0
    x_edges_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    x_edges = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 3))
    x_edges_vertices = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))
    # uses:
    pts_idx = np.load("pts_idx.npy")
    pts = np.load("pts.npy")
    # loop over x-edges:
    for ix in range(nx):
        for iy in range(ny + 1):
            x_edges_idx[ix, iy] = idx
            for iz in range(nz + 1):
                if not space[ix + 1, iy:iy + 1, iz:iz + 1].all():
                    x_edges[idx] = (ix, iy, iz)
                    # find start and end points
                    # start point (ix, iy, iz)
                    p1_i = pts_idx[ix, iy]
                    while (ix, iy, iz) != tuple(pts[p1_i]):
                        p1_i += 1

                    # end point (ix + 1, iy, iz)
                    p2_i = pts_idx[ix + 1, iy]
                    while (ix + 1, iy, iz) != tuple(pts[p2_i]):
                        p2_i += 1
                    x_edges_vertices[idx] = (p1_i, p2_i)
                    idx += 1

    np.save("x_edges.npy", x_edges)
    np.save("x_edges_idx.npy", x_edges_idx)
    np.save("x_edges_vertices.npy", x_edges_vertices)

    del x_edges_idx
    del x_edges
    del x_edges_vertices
    del pts_idx
    del pts

    counts[1][0] = idx


def y_edges_loop(space, counts):
    volumes_count = counts[3]
    nx, ny, nz = space.shape
    nx -= 2
    ny -= 2
    nz -= 2

    # returns (or caches):
    idx = 0
    y_edges_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    y_edges = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 3))
    y_edges_vertices = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))

    # uses:
    pts_idx = np.load("pts_idx.npy")
    pts = np.load("pts.npy")

    # loop over y-edges:
    for ix in range(nx + 1):
        for iy in range(ny):
            y_edges_idx[ix, iy] = idx
            for iz in range(nz + 1):
                if not space[ix:ix + 1, iy + 1, iz:iz + 1].all():
                    y_edges[idx] = (ix, iy, iz)
                    p1_i = pts_idx[ix, iy]
                    while (ix, iy, iz) != tuple(pts[p1_i]):
                        p1_i += 1
                    p2_i = pts_idx[ix, iy + 1]
                    while (ix, iy + 1, iz) != tuple(pts[p2_i]):
                        p2_i += 1
                    y_edges_vertices[idx] = (p1_i, p2_i)
                    idx += 1

    np.save("y_edges", y_edges)
    np.save("y_edges_idx", y_edges_idx)
    np.save("y_edges_vertices", y_edges_vertices)

    del y_edges_idx
    del y_edges
    del y_edges_vertices
    del pts_idx
    del pts

    counts[1][1] = idx


def z_edges_loop(space, counts):
    volumes_count = counts[3]
    nx, ny, nz = space.shape
    nx -= 2
    ny -= 2
    nz -= 2

    # returns (or caches):
    idx = 0
    z_edges_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    z_edges = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 3))
    z_edges_vertices = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))

    # uses:
    pts_idx = np.load("pts_idx.npy")
    pts = np.load("pts.npy")

    # loop over z-edges:
    for ix in range(nx + 1):
        for iy in range(ny + 1):
            z_edges_idx[ix, iy] = idx
            for iz in range(nz):
                if not space[ix:ix + 1, iy:iy + 1, iz + 1].all():
                    z_edges[idx] = (ix, iy, iz)
                    p1_i = pts_idx[ix, iy]
                    while (ix, iy, iz) != tuple(pts[p1_i]):
                        p1_i += 1
                    p2_i = p1_i + 1
                    z_edges_vertices[idx] = (p1_i, p2_i)
                    idx += 1

    np.save("z_edges", z_edges)
    np.save("z_edges_idx", z_edges_idx)
    np.save("z_edges_vertices", z_edges_vertices)

    del z_edges_idx
    del z_edges
    del z_edges_vertices
    del pts_idx
    del pts

    counts[1][2] = idx


def x_faces_loop(space, counts):
    volumes_count = counts[3]
    nx, ny, nz = space.shape
    nx -= 2
    ny -= 2
    nz -= 2
    x_edges_count, y_edges_count, z_edges_count = counts[1]

    # returns (or caches):
    idx = 0
    x_faces_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    x_faces = np.zeros((volumes_count + ny * nz), dtype=(np.uint16, 3))
    x_faces_edges = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 4))

    xmin_boundary = np.zeros((ny * nz), dtype=np.uint16)
    xmax_boundary = np.zeros((ny * nz), dtype=np.uint16)
    object_boundary = np.zeros((3 * nx * ny * nz - volumes_count + nx * ny + ny * nz + nx * nz), dtype=np.uint16)

    # uses:
    y_edges_idx = np.load("y_edges_idx.npy")
    y_edges = np.load("y_edges.npy")
    z_edges_idx = np.load("z_edges_idx.npy")
    z_edges = np.load("z_edges.npy")

    idx_xmin = idx_xmax = idx_obj_b = 0
    for ix in range(nx + 1):
        for iy in range(ny):
            x_faces_idx[ix, iy] = idx
            for iz in range(nz):
                if not space[ix:ix + 1, iy + 1, iz + 1].all():
                    if ix == 0:
                        xmin_boundary[idx_xmin] = idx
                        idx_xmin += 1
                    elif ix == nx:
                        xmax_boundary[idx_xmax] = idx
                        idx_xmax += 1
                    elif space[ix:ix + 1, iy + 1, iz + 1].any():
                        object_boundary[idx_obj_b] = idx
                        idx_obj_b += 1
                    x_faces[idx] = (ix, iy, iz)
                    # x-face (ix, iy, iz) has edges
                    # (ix, iy, iz), (ix, iy, iz+1) from y-edges,
                    # (ix, iy, iz), (ix, iy+1, iz) from z-edges
                    p1_i = y_edges_idx[ix, iy]
                    while (ix, iy, iz) != tuple(y_edges[p1_i]):
                        p1_i += 1

                    p3_i = p1_i + 1

                    p2_i = z_edges_idx[ix, iy]
                    while (ix, iy, iz) != tuple(z_edges[p2_i]):
                        p2_i += 1

                    p4_i = z_edges_idx[ix, iy + 1]
                    while (ix, iy + 1, iz) != tuple(z_edges[p4_i]):
                        p4_i += 1
                    # orientation (yz):
                    # p1 -> p4 -> -p3 -> -p2

                    x_faces_edges[idx] = (
                        x_edges_count + p1_i,
                        x_edges_count + y_edges_count + p4_i,
                        x_edges_count + p3_i,
                        x_edges_count + y_edges_count + p2_i)
                    idx += 1

    np.save("x_faces", x_faces)
    np.save("x_faces_idx", x_faces_idx)
    np.save("x_faces_edges", x_faces_edges)

    np.save("xmin_boundary", xmin_boundary)
    np.save("xmax_boundary", xmax_boundary)

    del x_faces_idx
    del x_faces
    del x_faces_edges

    del y_edges_idx
    del y_edges
    del z_edges_idx
    del z_edges

    counts[2][0] = idx
    counts[4][0] = idx_xmin
    counts[4][1] = idx_xmax
    counts[4][6] += idx_obj_b


def y_faces_loop(space, counts):

    volumes_count = counts[3]
    nx, ny, nz = space.shape
    nx -= 2
    ny -= 2
    nz -= 2
    x_edges_count, y_edges_count, z_edges_count = counts[1]
    x_faces_count, _, _ = counts[2]

    # returns (or caches):
    idx = 0
    y_faces_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    y_faces = np.zeros((volumes_count + nx * nz), dtype=(np.uint16, 3))
    y_faces_edges = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 4))

    ymin_boundary = np.zeros((ny * nz), dtype=np.uint16)
    ymax_boundary = np.zeros((ny * nz), dtype=np.uint16)
    object_boundary = np.zeros((ny * nz * nz - volumes_count + nx * ny + ny * nz + nx * nz), dtype=np.uint16)

    # uses:
    x_edges_idx = np.load("x_edges_idx.npy")
    x_edges = np.load("x_edges.npy")
    z_edges_idx = np.load("z_edges_idx.npy")
    z_edges = np.load("z_edges.npy")

    idx_ymin = idx_ymax = 0
    idx_obj_b = counts[4][6]
    for ix in range(nx):
        for iy in range(ny + 1):
            y_faces_idx[ix, iy] = idx
            for iz in range(nz):
                if not space[ix + 1, iy:iy + 1, iz + 1].all():
                    if iy == 0:
                        ymin_boundary[idx_ymin] = idx
                        idx_ymin += 1
                    elif iy == ny:
                        ymax_boundary[idx_ymax] = idx
                        idx_ymax += 1
                    elif space[ix + 1, iy:iy + 1, iz + 1].any():
                        object_boundary[idx_obj_b] = x_faces_count + idx
                        idx_obj_b += 1
                    y_faces[idx] = (ix, iy, iz)
                    # y-face (ix, iy, iz) has edges
                    # (ix, iy, iz), (ix, iy, iz+1) from x-edges,
                    # (ix, iy, iz), (ix+1, iy, iz) from z-edges
                    p1_i = x_edges_idx[ix, iy]
                    while (ix, iy, iz) != tuple(x_edges[p1_i]):
                        p1_i += 1

                    p3_i = p1_i + 1

                    p2_i = z_edges_idx[ix, iy]
                    while (ix, iy, iz) != tuple(z_edges[p2_i]):
                        p2_i += 1

                    p4_i = z_edges_idx[ix + 1, iy]
                    while (ix + 1, iy, iz) != tuple(z_edges[p4_i]):
                        p4_i += 1

                    y_faces_edges[idx] = (
                        p1_i,
                        x_edges_count + y_edges_count + p4_i,
                        p3_i,
                        x_edges_count + y_edges_count + p2_i)
                    idx += 1

    np.save("y_faces", y_faces)
    np.save("y_faces_idx", y_faces_idx)
    np.save("y_faces_edges", y_faces_edges)

    np.save("ymin_boundary", ymin_boundary)
    np.save("ymax_boundary", ymax_boundary)

    del y_faces_idx
    del y_faces
    del y_faces_edges

    del x_edges_idx
    del x_edges
    del z_edges_idx
    del z_edges

    counts[2][1] = idx
    counts[4][2] = idx_ymin
    counts[4][3] = idx_ymax
    counts[4][6] += idx_obj_b


def z_faces_loop(space, counts):
    volumes_count = counts[3]
    nx, ny, nz = space.shape
    nx -= 2
    ny -= 2
    nz -= 2
    x_edges_count, y_edges_count, z_edges_count = counts[1]
    x_faces_count, y_faces_count, _ = counts[2]

    # returns (or caches):
    idx = 0
    z_faces_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    z_faces = np.zeros((volumes_count + nx * ny), dtype=(np.uint16, 3))
    z_faces_edges = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 4))

    zmin_boundary = np.zeros((ny * nz), dtype=np.uint16)
    zmax_boundary = np.zeros((ny * nz), dtype=np.uint16)
    object_boundary = np.zeros((ny * nz * nz - volumes_count + nx * ny + ny * nz + nx * nz), dtype=np.uint16)

    # uses:
    x_edges_idx = np.load("x_edges_idx.npy")
    x_edges = np.load("x_edges.npy")
    y_edges_idx = np.load("y_edges_idx.npy")
    y_edges = np.load("y_edges.npy")

    idx_zmin = idx_zmax = 0
    idx_obj_b = counts[4][6]
    for ix in range(nx):
        for iy in range(ny):
            z_faces_idx[ix, iy] = idx
            for iz in range(nz + 1):
                if not space[ix + 1, iy + 1, iz:iz+1].all():
                    if iz == 0:
                        zmin_boundary[idx_zmin] = idx
                        idx_zmin += 1
                    elif iz == nz:
                        zmax_boundary[idx_zmax] = idx
                        idx_zmax += 1
                    elif space[ix + 1, iy + 1, iz:iz + 1].any():
                        object_boundary[idx_obj_b] = x_faces_count + y_faces_count + idx
                        idx_obj_b += 1

                    z_faces[idx] = (ix, iy, iz)
                    # z-face (ix, iy, iz) has edges
                    # (ix, iy, iz), (ix, iy+1, iz) from x-edges,
                    # (ix, iy, iz), (ix+1, iy, iz) from y-edges
                    p1_i = x_edges_idx[ix, iy]
                    while (ix, iy, iz) != tuple(x_edges[p1_i]):
                        p1_i += 1

                    p3_i = x_edges_idx[ix, iy + 1]
                    while (ix, iy + 1, iz) != tuple(x_edges[p3_i]):
                        p3_i += 1

                    p2_i = y_edges_idx[ix, iy]
                    while (ix, iy, iz) != tuple(y_edges[p2_i]):
                        p2_i += 1

                    p4_i = y_edges_idx[ix + 1, iy]
                    while (ix + 1, iy, iz) != tuple(y_edges[p4_i]):
                        p4_i += 1

                    z_faces_edges[idx] = (
                        p1_i,
                        x_edges_count + p4_i,
                        p3_i,
                        x_edges_count + p2_i
                    )
                    idx += 1

    np.save("z_faces", z_faces)
    np.save("z_faces_idx", z_faces_idx)
    np.save("z_faces_edges", z_faces_edges)

    np.save("zmin_boundary", zmin_boundary)
    np.save("zmax_boundary", zmax_boundary)

    del z_faces_idx
    del z_faces
    del z_faces_edges

    del x_edges_idx
    del x_edges
    del y_edges_idx
    del y_edges

    counts[2][2] = idx
    counts[4][4] = idx_zmin
    counts[4][5] = idx_zmax
    counts[4][6] += idx_obj_b


def write_region_geo_2(fn, space, counts):
    """def volumes_loop(space, counts):"""

    volumes_count = counts[3]
    nx, ny, nz = space.shape
    nx -= 2
    ny -= 2
    nz -= 2
    [x_faces_count, y_faces_count, z_faces_count] = counts[2]

    # returns (or caches):
    ## idx = 0
    ##volumes = np.zeros((volumes_count + nx * ny), dtype=(np.uint16, 3))
    ##volumes_faces = np.zeros((volumes_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 6))

    # uses:
    x_faces_idx = np.load("x_faces_idx.npy")
    x_faces = np.load("x_faces.npy")
    y_faces_idx = np.load("y_faces_idx.npy")
    y_faces = np.load("y_faces.npy")
    z_faces_idx = np.load("z_faces_idx.npy")
    z_faces = np.load("z_faces.npy")

    faces_count = x_faces_count + y_faces_count + z_faces_count
    fp = open(fn, 'a')
    fp.write('{} {}'.format(
        volumes_count, faces_count
    ))
    fp.write('\n')
    for i in range(0, 6*volumes_count, 6):
        fp.write('{}'.format(i))
        fp.write('\n')
    fp.write('{}'.format(6*volumes_count))
    fp.write('\n')
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if not space[ix + 1, iy + 1, iz + 1]:
                    ## volumes[idx] = (ix, iy, iz)

                    # volume (ix, iy, iz) has faces:
                    # (ix, iy, iz), (ix+1, iy, iz) from x-faces etc
                    p1_i = x_faces_idx[ix, iy]
                    while (ix, iy, iz) != tuple(x_faces[p1_i]):
                        p1_i += 1

                    p4_i = x_faces_idx[ix + 1, iy]
                    while (ix + 1, iy, iz) != tuple(x_faces[p4_i]):
                        p4_i += 1

                    p2_i = y_faces_idx[ix, iy]
                    while (ix, iy, iz) != tuple(y_faces[p2_i]):
                        p2_i += 1

                    p5_i = y_faces_idx[ix, iy]
                    while (ix, iy + 1, iz) != tuple(y_faces[p5_i]):
                        p5_i += 1

                    p3_i = z_faces_idx[ix, iy]
                    while (ix, iy, iz) != tuple(z_faces[p3_i]):
                        p3_i += 1

                    p6_i = p3_i + 1
                    # orientation: p1, p2, p3, -p4, -p5, - p6
                    ##volumes_faces[idx] = \
                    tu = (
                        p1_i,
                        x_faces_count + p2_i,
                        x_faces_count + y_faces_count + p3_i,
                        p4_i,
                        x_faces_count + p5_i,
                        x_faces_count + y_faces_count + p6_i
                    )
                    ## idx += 1
                    fp.write('{} {} {} {} {} {}'.format(
                        tu[0],
                        tu[1],
                        tu[2],
                        MIN_INTEGER + tu[3],
                        MIN_INTEGER + tu[4],
                        MIN_INTEGER + tu[5]
                    ))
                    fp.write('\n')

    fp.write('\n')
    fp.close()
    ## del volumes
    ## del volumes_faces

    del x_faces_idx
    del x_faces
    del y_faces_idx
    del y_faces
    del z_faces_idx
    del z_faces

    # np.save("volumes", volumes)
    # np.save("volumes_faces", volumes_faces)

    # return idx


def read_grd(fn):

    with open(fn) as fp:
        for _ in range(6):
            fp.readline()
        xx = list(map(float, fp.readline().strip().split()))
        for _ in range(2):
            fp.readline()
        yy = list(map(float, fp.readline().strip().split()))
        for _ in range(2):
            fp.readline()
        zz = list(map(float, fp.readline().strip().split()))

    return xx, yy, zz


def read_grd_dim(fn):

    with open(fn) as fp:
        for _ in range(5):
            fp.readline()
        nx = int(fp.readline().strip())
        for _ in range(2):
            fp.readline()
        ny = int(fp.readline().strip())
        for _ in range(2):
            fp.readline()
        nz = int(fp.readline().strip())

    return nx, ny, nz


def read_cel(fn, n):
    li = np.zeros((n), dtype=np.uint8)
    fp = open(fn, 'r')
    idx = 0
    while idx < n:
        lay = int(fp.readline().strip())
        count = int(fp.readline().strip())
        li[idx: idx + count] = lay
        idx += count
    return li


def write_region_geo_3(fn, grd_fn, counts):
    xx, yy, zz = read_grd(grd_fn)
    pts = np.load("pts.npy")
    fp = open(fn, 'a')

    for p in pts[:counts[0]]:
        fp.write('({}, {}, {})'.format(xx[p[0]], yy[p[1]], zz[p[2]]))
        fp.write('\n')
    fp.write('\n')
    fp.close()


def write_edges(fn, counts):
    pts_count = counts[0]
    [x_edges_count, y_edges_count, z_edges_count] = counts[1]

    x_edges_vertices = np.load("x_edges_vertices.npy")
    y_edges_vertices = np.load("y_edges_vertices.npy")
    z_edges_vertices = np.load("z_edges_vertices.npy")

    edges_count = x_edges_count + y_edges_count + z_edges_count
    fp = open(fn, 'a')
    fp.write('{} {}'.format(
        edges_count, pts_count
    ))
    fp.write('\n')
    for i in range(0, 2*edges_count, 2):
        fp.write('{}'.format(i))
        fp.write('\n')
    fp.write('{}'.format(2*edges_count))
    fp.write('\n')
    for i in range(x_edges_count):
        fp.write('{} {}'.format(MIN_INTEGER + x_edges_vertices[i][0], x_edges_vertices[i][1]))
        fp.write('\n')
    for i in range(y_edges_count):
        fp.write('{} {}'.format(MIN_INTEGER + y_edges_vertices[i][0], y_edges_vertices[i][1]))
        fp.write('\n')
    for i in range(z_edges_count):
        fp.write('{} {}'.format(MIN_INTEGER + z_edges_vertices[i][0], z_edges_vertices[i][1]))
        fp.write('\n')
    fp.write('\n')
    fp.close()

    del x_edges_vertices
    del y_edges_vertices
    del z_edges_vertices


def write_faces(fn, counts):
    [x_edges_count, y_edges_count, z_edges_count] = counts[1]
    [x_faces_count, y_faces_count, z_faces_count] = counts[2]
    x_faces_edges = np.load("x_faces_edges.npy")
    y_faces_edges = np.load("y_faces_edges.npy")
    z_faces_edges = np.load("z_faces_edges.npy")

    edges_count = x_edges_count + y_edges_count + z_edges_count
    faces_count = x_faces_count + y_faces_count + z_faces_count
    fp = open(fn, 'a')
    fp.write('{} {}'.format(
        faces_count, edges_count
    ))
    fp.write('\n')
    for i in range(0, 4 * faces_count, 4):
        fp.write('{}'.format(i))
        fp.write('\n')
    fp.write('{}'.format(4 * faces_count))
    fp.write('\n')
    for i in range(x_faces_count):
        fp.write('{} {} {} {}'.format(
            x_faces_edges[i][0],
            x_faces_edges[i][1],
            MIN_INTEGER + x_faces_edges[i][2],
            MIN_INTEGER + x_faces_edges[i][3]))
        fp.write('\n')
    for i in range(y_faces_count):
        fp.write('{} {} {} {}'.format(
            y_faces_edges[i][0],
            y_faces_edges[i][1],
            MIN_INTEGER + y_faces_edges[i][2],
            MIN_INTEGER + y_faces_edges[i][3]))
        fp.write('\n')
    for i in range(z_faces_count):
        fp.write('{} {} {} {}'.format(
            z_faces_edges[i][0],
            z_faces_edges[i][1],
            MIN_INTEGER + z_faces_edges[i][2],
            MIN_INTEGER + z_faces_edges[i][3]))
        fp.write('\n')
    fp.write('\n')
    fp.close()
    del x_faces_edges
    del y_faces_edges
    del z_faces_edges


def write_region_geo_1(fn, counts):
    write_edges(fn, counts)
    write_faces(fn, counts)


def write_region_geo(fn, counts):
    pts_count = counts[0]
    [x_edges_count, y_edges_count, z_edges_count] = counts[1]
    [x_faces_count, y_faces_count, z_faces_count] = counts[2]
    volumes_count = counts[3]
    fp = open(fn, 'a')
    fp.write('{}\n'.format(MAX_DIM))
    fp.write('{} {} {} {}\n'.format(
        pts_count,
        x_edges_count + y_edges_count + z_edges_count,
        x_faces_count + y_faces_count + z_faces_count,
        volumes_count)
    )
    fp.write('0\n')
    fp.write('0 0 0 0\n')
    fp.write('1 0 0 0\n')
    fp.write('0 1 0 0\n')
    fp.write('0 0 1 0\n')
    fp.write('\n')
    fp.close()


def write_remp_region_geo(grd_fn, cel_fn, mesh_fn):
    nx, ny, nz = read_grd_dim(grd_fn)
    space_li = read_cel(cel_fn, nx * ny * nz)
    space = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.bool)
    space[1:-1, 1:-1, 1:-1] = space_li.reshape((nx, ny, nz))
    space[0, ...] = space[1, ...]
    space[-1, ...] = space[-2, ...]
    space[:, 0, :] = space[:, 1, :]
    space[:, -1, :] = space[:, -2, :]
    space[..., 0] = space[..., 1]
    space[..., -1] = space[..., -2]

    elems, counts = np.unique(space_li, return_counts=True)
    idx_False = 0
    if elems[idx_False]:
        idx_False = 1
    volumes_count = counts[idx_False]

    counts = [0, [0, 0, 0], [0, 0, 0], volumes_count, [0]*7]
    print("processing points")
    points_loop(space, counts)
    gc.collect()
    print("processing edges")
    x_edges_loop(space, counts)
    gc.collect()
    print("x")
    y_edges_loop(space, counts)
    gc.collect()
    print("y")
    z_edges_loop(space, counts)
    gc.collect()
    print("z")
    print("processing faces")
    x_faces_loop(space, counts)
    gc.collect()
    print("x")
    y_faces_loop(space, counts)
    gc.collect()
    print("y")
    z_faces_loop(space, counts)
    gc.collect()
    print("z")
    write_region_geo(mesh_fn, counts)
    print("writing edges and faces")
    write_region_geo_1(mesh_fn, counts)
    gc.collect()
    print("processing and writing volumes")
    write_region_geo_2(mesh_fn, space, counts)
    gc.collect()
    print("writing coordinates")
    write_region_geo_3(mesh_fn, grd_fn, counts)
    gc.collect()


def write_boundaries(fn, counts, space):

    nx, ny, nz = space.shape
    nx -= 2
    ny -= 2
    nz -= 2

    x_faces_count, y_faces_count, z_faces_count = counts[2]
    xmin_faces_count, xmax_faces_count, \
        ymin_faces_count, ymax_faces_count, \
        zmin_faces_count, zmax_faces_count, object_faces_count = counts[4]

    xmin_boundary = np.load("xmin_boundary.npy")
    xmax_boundary = np.load("xmax_boundary.npy")
    ymin_boundary = np.load("ymin_boundary.npy")
    ymax_boundary = np.load("ymax_boundary.npy")
    zmin_boundary = np.load("zmin_boundary.npy")
    zmax_boundary = np.load("zmax_boundary.npy")
    object_boundary = np.load("object_boundary.npy")

    fp = open(fn, 'a')

    fp.write('7\n\n')

    idx = 0
    fp.write('{}\n'.format(idx))    # xmin description starts

    idx += xmin_faces_count
    fp.write('{}\n'.format(idx))    # ymin description starts

    idx += ymin_faces_count
    fp.write('{}\n'.format(idx))    # zmin description starts

    idx += zmin_faces_count
    fp.write('{}\n'.format(idx))

    idx += xmax_faces_count
    fp.write('{}\n'.format(idx))

    idx += ymax_faces_count
    fp.write('{}\n'.format(idx))

    idx += zmax_faces_count
    fp.write('{}\n\n'.format(idx))

    idx += object_faces_count
    fp.write('{}\n\n'.format(idx))

    for idx in range(xmin_faces_count):
        fp.write('{} '.format(xmin_boundary[idx]))
    fp.write('\n')
    for idx in range(ymin_faces_count):
        fp.write('{} '.format(x_faces_count + ymin_boundary[idx]))
    fp.write('\n')
    for idx in range(zmin_faces_count):
        fp.write('{} '.format(x_faces_count + y_faces_count + zmin_boundary[idx]))
    fp.write('\n')
    for idx in range(xmax_faces_count):
        fp.write('{} '.format(xmax_boundary[idx]))
    fp.write('\n')
    for idx in range(ymax_faces_count):
        fp.write('{} '.format(x_faces_count + ymax_boundary[idx]))
    fp.write('\n')
    for idx in range(zmax_faces_count):
        fp.write('{} '.format(x_faces_count + y_faces_count + zmax_boundary[idx]))
    fp.write('\n')
    for idx in range(object_faces_count):
        fp.write('{} '.format(object_boundary[idx]))
    fp.write('\n')
    fp.write('\n')

    fp.write('XMIN\n'
             'XMAX\n'
             'YMIN\n'
             'YMAX\n'
             'ZMIN\n'
             'ZMAX\n'
             'INNER\n\n')


    fp.close()


if __name__ == '__main__':
    write_remp_region_geo(r'DEFAULT.GRD', r'DEFAULT.CEL', 'test.mrp')