# fro https://github.com/NVIDIA/warp/blob/f966e847e49e8dc07fec479a8edfa86557e2c1d6/warp/tests/test_hash_grid.py

import numpy as np
import torch
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



def test_hashgrid_query( device,points_arr, lenPoints ): #points_arr
    grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
    # points = [[1.0, 2.0, 2.0]
    #            ,[1.0, 22.0, 2.0]
    #             ,[1.0,0.0,2.0]]
    # points_arr = wp.array(points, dtype=wp.vec3, device=device)
    counts_arr = wp.zeros(lenPoints, dtype=wp.types.float32, device=device)

    grid.build(points_arr, query_radius)
    wp.synchronize()

    wp.launch(kernel=count_neighbors, dim=lenPoints, inputs=[grid.id, query_radius, points_arr, counts_arr], device=device)
    wp.synchronize()

    counts = counts_arr.numpy()


    print(f"suum: {np.max(counts)}")

labelBoolTensorA = torch.zeros(dim_x,dim_y,dim_z, dtype=torch.bool)
labelBoolTensorB = torch.zeros(dim_x,dim_y,dim_z, dtype=torch.bool)

labelBoolTensorA[0,0,0]=True
labelBoolTensorA[0,0,1]=True

labelBoolTensorB[0,0,0]=True
labelBoolTensorB[0,0,4]=True
labelBoolTensorB[0,0,5]=True
labelBoolTensorB[0,0,2]=True
labelBoolTensorB[0,0,15]=True


#true positives
eqq =torch.logical_and(labelBoolTensorA,labelBoolTensorB)  #torch.eq(labelBoolTensorA,summB)
#false negatives labelBoolTensorA - gold stadard
fn=  torch.logical_and(torch.logical_not(eqq),labelBoolTensorA)
#false positives labelBoolTensorA - gold stadard
fp=  torch.logical_and(torch.logical_not(eqq),labelBoolTensorB)
#combined
comb= torch.logical_or(labelBoolTensorA,labelBoolTensorB)


print(fp.to_sparse().indices().size())

sumA = torch.sum(eqq).item() +torch.sum(fn).item()+torch.sum(fp).item()

####first we will look around of all fp points 
num_points = torch.sum(fp).item()
print(f" {num_points} ")
# fpIndicies= wp.from_torch( fp.to_sparse().indices().type(torch.int32) 
#                             , dtype=wp.types.int32)

goldIndicies = wp.from_torch( torch.t(fp.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 
                             , dtype=wp.vec3)


devices = wp.get_devices()
print(devices[1])
test_hashgrid_query(devices[1],goldIndicies,num_points)
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