# fro https://github.com/NVIDIA/warp/blob/f966e847e49e8dc07fec479a8edfa86557e2c1d6/warp/tests/test_hash_grid.py

import numpy as np

import warp as wp
from warp.tests.test_base import *
wp.init()
num_points = 3
dim_x = 32
dim_y = 32
dim_z = 32

scale = 150.0

query_radius = 50.0


@wp.kernel
def count_neighbors(grid : wp.uint64,
                    radius: float,
                    points: wp.array(dtype=wp.vec3),
                    counts: wp.array(dtype=wp.types.float32)):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # query point    
    p = points[i]
    print(p)
    maxx = float(0)

    # construct query around point p
    
    neighbors = wp.hash_grid_query(grid, p, radius)
    for index in neighbors:
        # compute distance to point
        d = wp.length(p - points[index])
        if (d <= radius):
            maxx= wp.max(d,maxx)

    counts[i] = maxx



def test_hashgrid_query( device):
    grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
    points = [[1.0, 2.0, 2.0]
               ,[1.0, 22.0, 2.0]
                ,[1.0,0.0,2.0]]
    points_arr = wp.array(points, dtype=wp.vec3, device=device)
    counts_arr = wp.zeros(len(points), dtype=wp.types.float32, device=device)

    grid.build(points_arr, query_radius)
    wp.synchronize()

    wp.launch(kernel=count_neighbors, dim=len(points), inputs=[grid.id, query_radius, points_arr, counts_arr], device=device)
    wp.synchronize()

    counts = counts_arr.numpy()


    print(f"suum: {np.max(counts)}")

devices = wp.get_devices()
print(devices[1])
test_hashgrid_query(devices[1])
    #test.assertTrue(np.array_equal(counts, counts_ref))
        


# @wp.kernel
# def count_neighbors_reference(
#                     radius: float,
#                     points: wp.array(dtype=wp.vec3),
#                     counts: wp.array(dtype=int),
#                     num_points: int):

#     tid = wp.tid()

#     i = tid%num_points
#     j = tid//num_points
    
#     # query point
#     p = points[i]
#     q = points[j]

#     # compute distance to point
#     d = wp.length(p - q)

#     if (d <= radius):
#         wp.atomic_add(counts, i, 1)