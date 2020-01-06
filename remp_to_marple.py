import gc

import numpy as np

MIN_INTEGER = -2147483648


def points_loop():
    # TODO: arrays should be deleted for gc to work properly
    idx = 0
    pts_idx = np.zeros((nx+1, ny+1), dtype=np.int32)
    pts = np.zeros((nonzero_cells_count + nx*ny + ny*nz + nx*nz), dtype=(np.uint16, 3))
    for ix in range(nx+1):
        for iy in range(ny + 1):
            pts_idx[ix, iy] = idx
            for iz in range(nz + 1):
                if not space[ix:ix+1, iy:iy+1, iz:iz+1].any():
                    pts[idx] = (ix, iy, iz)
                    idx += 1

    np.save("pts_idx", pts_idx)
    np.save("pts", pts)

    return idx


def x_edges_loop():
    # TODO: arrays should be deleted for gc to work properly
    # TODO: adjust indices
    idx = 0
    x_edges_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    x_edges = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 3))
    x_edges_vertices = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))
    pts_idx = np.load("pts_idx")
    pts = np.load("pts")
    for ix in range(nx):
        for iy in range(ny + 1):
            x_edges_idx[ix, iy] = idx
            for iz in range(nz + 1):
                if not space[ix, iy:iy + 1, iz:iz + 1].any():
                    x_edges[idx] = (ix, iy, iz)
                    # find start and end points
                    # start point (ix, iy, iz)
                    p1_i = pts_idx[ix, iy]
                    while (ix, iy, iz) != pts[p1_i]:
                        p1_i += 1

                    # end point (ix + 1, iy, iz)
                    p2_i = pts_idx[ix, iy]
                    while (ix + 1, iy, iz) != pts[p2_i]:
                        p2_i += 1
                    x_edges_vertices[idx] = (p1_i, p2_i)
                    idx += 1

    np.save("x_edges", x_edges)
    np.save("x_edges_idx", x_edges_idx)
    np.save("x_edges_vertices", x_edges_vertices)

    return idx

def y_edges_loop():
    # TODO: arrays should be deleted for gc to work properly
    # TODO: adjust indices
    idx = 0
    y_edges_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    y_edges = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 3))
    y_edges_vertices = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))
    pts_idx = np.load("pts_idx")
    pts = np.load("pts")
    for ix in range(nx + 1):
        for iy in range(ny):
            y_edges_idx[ix, iy] = idx
            for iz in range(nz + 1):
                if not space[ix:ix + 1, iy, iz:iz + 1].any():
                    y_edges[idx] = (ix, iy, iz)
                    p1_i = pts_idx[ix, iy]
                    while (ix, iy, iz) != pts[p1_i]:
                        p1_i += 1

                    p2_i = pts_idx[ix, iy + 1]
                    while (ix, iy + 1, iz) != pts[p2_i]:
                        p2_i += 1
                    y_edges_vertices[idx] = (p1_i, p2_i)
                    idx += 1

    np.save("y_edges", y_edges)
    np.save("y_edges_idx", y_edges_idx)
    np.save("y_edges_vertices", y_edges_vertices)

    return idx

def z_edges_loop():
    # TODO: arrays should be deleted for gc to work properly
    # TODO: adjust indices
    idx = 0
    z_edges_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    z_edges = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 3))
    z_edges_vertices = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))
    pts_idx = np.load("pts_idx")
    pts = np.load("pts")
    for ix in range(nx + 1):
        for iy in range(ny + 1):
            z_edges_idx[ix, iy] = idx
            for iz in range(nz):
                if not space[ix:ix + 1, iy:iy + 1, iz].any():
                    z_edges[idx] = (ix, iy, iz)
                    # find start and end points
                    # start point (ix, iy, iz)
                    p1_i = pts_idx[ix, iy]
                    while (ix, iy, iz) != pts[p1_i]:
                        p1_i += 1
                    p2_i = p1_i + 1
                    z_edges_vertices[idx] = (p1_i, p2_i)
                    idx += 1

    np.save("z_edges", z_edges)
    np.save("z_edges_idx", z_edges_idx)
    np.save("z_edges_vertices", z_edges_vertices)

    return idx


def x_faces_loop():
    # TODO: arrays should be deleted for gc to work properly
    # TODO: adjust indices
    idx = 0
    x_faces_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    x_faces = np.zeros((nonzero_cells_count + nx * ny), dtype=(np.uint16, 3))
    x_faces_edges = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))

    y_edges_idx = np.load("y_edges_idx")
    y_edges = np.load("y_edges")
    z_edges_idx = np.load("z_edges_idx")
    z_edges = np.load("z_edges")

    for ix in range(nx + 1):
        for iy in range(ny):
            x_faces_idx[ix, iy] = idx
            for iz in range(nz):
                if not space[ix:ix + 1, iy, iz].any():
                    x_faces[idx] = (ix, iy, iz)

                    p1_i = y_edges_idx[ix, iy]
                    while (ix, iy, iz) != y_edges[p1_i]:
                        p1_i += 1

                    p3_i = y_edges_idx[ix, iy]
                    while (ix, iy, iz) != y_edges[p3_i]:
                        p3_i += 1

                    p2_i = z_edges_idx[ix, iy]
                    while (ix, iy, iz) != z_edges[p2_i]:
                        p2_i += 1

                    p4_i = p2_i + 1

                    x_faces_edges[idx] = (p1_i, p2_i, p3_i, p4_i)
                    idx += 1

    np.save("x_faces", x_faces)
    np.save("x_faces_idx", x_faces_idx)
    np.save("x_faces_edges", x_faces_edges)

    return idx


def y_faces_loop():
    # TODO: arrays should be deleted for gc to work properly
    # TODO: adjust indices
    idx = 0
    y_faces_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    y_faces = np.zeros((nonzero_cells_count + nx * ny), dtype=(np.uint16, 3))
    y_faces_edges = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))

    x_edges_idx = np.load("x_edges_idx")
    x_edges = np.load("x_edges")
    z_edges_idx = np.load("z_edges_idx")
    z_edges = np.load("z_edges")

    for ix in range(nx):
        for iy in range(ny + 1):
            y_faces_idx[ix, iy] = idx
            for iz in range(nz):
                if not space[ix, iy:iy + 1, iz].any():
                    y_faces[idx] = (ix, iy, iz)

                    p1_i = x_edges_idx[ix, iy]
                    while (ix, iy, iz) != x_edges[p1_i]:
                        p1_i += 1

                    p3_i = x_edges_idx[ix, iy]
                    while (ix, iy, iz) != x_edges[p3_i]:
                        p3_i += 1

                    p2_i = z_edges_idx[ix, iy]
                    while (ix, iy, iz) != z_edges[p2_i]:
                        p2_i += 1

                    p4_i = p2_i + 1

                    y_faces_edges[idx] = (p1_i, p2_i, p3_i, p4_i)
                    idx += 1

    np.save("x_faces", y_faces)
    np.save("x_faces_idx", y_faces_idx)
    np.save("x_faces_edges", y_faces_edges)

    return idx


def z_faces_loop():
    # TODO: arrays should be deleted for gc to work properly
    # TODO: adjust indices
    idx = 0
    z_faces_idx = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    z_faces = np.zeros((nonzero_cells_count + nx * ny), dtype=(np.uint16, 3))
    z_faces_edges = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))

    x_edges_idx = np.load("x_edges_idx")
    x_edges = np.load("x_edges")
    y_edges_idx = np.load("z_edges_idx")
    y_edges = np.load("z_edges")

    for ix in range(nx):
        for iy in range(ny + 1):
            z_faces_idx[ix, iy] = idx
            for iz in range(nz):
                if not space[ix, iy:iy + 1, iz].any():
                    z_faces[idx] = (ix, iy, iz)

                    p1_i = x_edges_idx[ix, iy]
                    while (ix, iy, iz) != x_edges[p1_i]:
                        p1_i += 1

                    p3_i = x_edges_idx[ix, iy]
                    while (ix, iy, iz) != x_edges[p3_i]:
                        p3_i += 1

                    p2_i = y_edges_idx[ix, iy]
                    while (ix, iy, iz) != y_edges[p2_i]:
                        p2_i += 1

                    p4_i = y_edges_idx[ix, iy]
                    while (ix, iy, iz) != y_edges[p4_i]:
                        p4_i += 1

                    z_faces_edges[idx] = (p1_i, p2_i, p3_i, p4_i)
                    idx += 1

    np.save("z_faces", z_faces)
    np.save("z_faces_idx", z_faces_idx)
    np.save("z_faces_edges", z_faces_edges)

    return idx


def write_region_geo_2(fn):
    # TODO: should write to file
    # TODO: adjust indices
    """def volumes_loop():"""

    # returns (or caches):
    idx = 0
    volumes = np.zeros((nonzero_cells_count + nx * ny), dtype=(np.uint16, 3))
    volumes_faces = np.zeros((nonzero_cells_count + nx * ny + ny * nz + nx * nz), dtype=(np.uint16, 2))

    # uses:
    x_faces_idx = np.load("x_faces_idx")
    x_faces = np.load("x_faces")
    y_faces_idx = np.load("y_faces_idx")
    y_faces = np.load("y_faces")
    z_faces_idx = np.load("z_faces_idx")
    z_faces = np.load("z_faces")

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if not space[ix, iy, iz]:
                    volumes[idx] = (ix, iy, iz)

                    p1_i = x_faces_idx[ix, iy]
                    while (ix, iy, iz) != x_faces[p1_i]:
                        p1_i += 1

                    p4_i = x_faces_idx[ix, iy]
                    while (ix, iy, iz) != x_faces[p4_i]:
                        p4_i += 1

                    p2_i = y_faces_idx[ix, iy]
                    while (ix, iy, iz) != y_faces[p2_i]:
                        p2_i += 1

                    p5_i = y_faces_idx[ix, iy]
                    while (ix, iy, iz) != y_faces[p5_i]:
                        p5_i += 1

                    p3_i = z_faces_idx[ix, iy]
                    while (ix, iy, iz) != z_faces[p3_i]:
                        p3_i += 1

                    p6_i = p3_i + 1

                    volumes_faces[idx] = (p1_i, p2_i, p3_i, p4_i, p5_i, p6_i)
                    idx += 1

    # np.save("volumes", volumes)
    # np.save("volumes_faces", volumes_faces)

    # return idx


def read_grd(fn):
    # TODO: should work

    xx = []
    yy = []
    zz = []
    with open(fn) as fp:
        lines = fp.readlines()
        xx = map(float, lines[3].strip().split())
        yy = map(float, lines[5].strip().split())
        zz = map(float, lines[7].strip().split())
    xx = list(xx)
    yy = list(yy)
    zz = list(zz)

    return x, y, z


def read_grd_dim(fn):
    # TODO: should work
    xx = []
    yy = []
    zz = []
    with open(fn) as fp:
        lines = fp.readlines()
        xx = map(float, lines[3].strip().split())
        yy = map(float, lines[5].strip().split())
        zz = map(float, lines[7].strip().split())
    xx = list(xx)
    yy = list(yy)
    zz = list(zz)

    return x, y, z


def read_cel(fn):
    return [0] * nx * ny * nz


def write_region_geo_3(fn, grd_fn):
    xx, yy, zz = read_grd(grd_fn)
    # write_pts(fn)


def write_edges(fn):
    pass


def write_faces(fn):
    pass


def write_region_geo_1(fn):
    write_edges(fn)
    write_faces(fn)


if __name__ == '__main__':
    nx, ny, nz = read_grd_dim(r'DEFAULT.GRD')
    space_li = read_cel(r'DEFAULT.CEL')
    space = np.asarray(space_li, dtype=np.bool).resize((nx, ny, nz))
    # TODO: space --> memmap

    elems, counts = np.unique(space, return_counts=True)
    idx_False = 0
    if elems[idx_False]:
        idx_False = 1
    nonzero_cells_count = counts[idx_False]

    pts_count = points_loop()
    gc.collect()

    x_edges_count = x_edges_loop()
    gc.collect()
    y_edges_count = y_edges_loop()
    gc.collect()
    z_edges_count = z_edges_loop()
    gc.collect()

    x_faces_count = x_faces_loop()
    gc.collect()
    y_faces_count = y_faces_loop()
    gc.collect()
    z_faces_count = z_faces_loop()
    gc.collect()

    write_region_geo_1(r'mesh.mrp')
    gc.collect()

    write_region_geo_2(r'mesh.mrp')
    gc.collect()

    write_region_geo_3(r'mesh.mrp', r'DEFAULT.GRD')
    gc.collect()
