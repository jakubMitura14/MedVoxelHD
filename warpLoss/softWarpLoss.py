import os
import math

import numpy as np

import torch

import warp as wp

    
# wp.init()

def prepare_tensors_for_warp_loss(y_true, y_hat,radius,device):
    """
    y_true and hould be boolean tensors with the same dimensions as y_hat (y_hat should be float tensor)
    we need to parse the arrays to format of indicies in order to be able to use warp with HashGrid
    """
    # y_true=y_true[0,0,:,:,:]
    # y_hat=y_hat[0,0,:,:,:]

    sizz= y_true.size()
    dim_x = sizz[0]
    dim_y = sizz[1]
    dim_z = sizz[2]

# grid : wp.uint64,
#                     points_in_grid: wp.array(dtype=wp.vec3),
#                     points_labelArr: wp.array(dtype=wp.types.int32),
#                     counts: wp.array(dtype=float),
#                     y_hat_arr : wp.array(dtype=float)
    num_points_gold = torch.sum(y_true).item()
    num_points_gold_false= torch.numel(y_true) - num_points_gold
    #print(f"num_points_gold {num_points_gold} num_points_gold_false {num_points_gold_false}   ")

    points_in_grid=torch.argwhere(torch.logical_not(y_true)).type(torch.float32).contiguous().to('cuda')
    points_labelArr=torch.argwhere(y_true).type(torch.float32).contiguous().to('cuda')

    counts_arr = torch.zeros(num_points_gold_false, dtype=torch.float32, requires_grad=True).to('cuda') 


    # return (points_in_grid, points_labelArr,  y_hat, counts_arr
    # ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false)
    return (points_in_grid, points_labelArr,  torch.sigmoid(y_hat), counts_arr
    ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false)




class getHausdorff_soft(torch.autograd.Function):
    """
    based on example from https://github.com/NVIDIA/warp/blob/main/examples/example_sim_fk_grad_torch.py
    """
    @staticmethod
    def forward(ctx
    ,points_in_grid, points_labelArr, y_hat, counts_arr
    ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false
    ):
        ctx.tape = wp.Tape()
        ctx.points_in_grid=wp.from_torch( points_in_grid, dtype=wp.vec3)
        ctx.points_labelArr=wp.from_torch( points_labelArr, dtype=wp.vec3)
        ctx.y_hat=wp.from_torch(y_hat , dtype=wp.types.float32)
        ctx.counts_arr=wp.from_torch(counts_arr , dtype=wp.types.float32)



        wp.synchronize()

        print(device)

        grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
        idd=grid.id
        grid.build(ctx.points_in_grid, radius)
        wp.synchronize()


        with ctx.tape:
            wp.launch(kernel=count_neighbors, dim=  num_points_gold_false, inputs= [grid.id,
                                                                            ctx.points_in_grid,
                                                                            ctx.points_labelArr,
                                                                            ctx.counts_arr,
                                                                            ctx.y_hat
                                                                            ,num_points_gold
                                                                            ,(dim_x+dim_y+dim_z)/40], device = device)#(dim_x+dim_y+dim_z)/10


        # return (wp.to_torch(ctx.counts_arr_fp),
        #         wp.to_torch(ctx.counts_arr_fn))



    @staticmethod
    def backward(ctx,points_in_grid, points_labelArr, y_hat, counts_arr
    ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false):

        # map incoming Torch grads to our output variables
        ctx.points_in_grid=wp.from_torch( points_in_grid, dtype=wp.vec3)
        ctx.points_labelArr=wp.from_torch( points_labelArr, dtype=wp.vec3)
        ctx.y_hat=wp.from_torch(y_hat , dtype=wp.types.float32)
        ctx.counts_arr=wp.from_torch(counts_arr , dtype=wp.types.float32)

        ctx.tape.backward()

        # return adjoint w.r.t. inputs
        return (wp.to_torch(ctx.tape.gradients[ctx.points_in_grid]), 
                wp.to_torch(ctx.tape.gradients[ctx.points_labelArr]),
                wp.to_torch(ctx.tape.gradients[ctx.y_hat]),
                wp.to_torch(ctx.tape.gradients[ctx.counts_arr])
                 ,None,None,None,None,None,None, None)





@wp.kernel
def count_neighbors(grid : wp.uint64,
                    points_in_grid: wp.array(dtype=wp.vec3),
                    points_labelArr: wp.array(dtype=wp.vec3),
                    counts: wp.array(dtype=float),
                    y_hat_arr : wp.array(dtype=float,ndim=3),
                    points_labelArr_len: wp.types.int32,
                    max_dist: float

                    ):
    """
    grid- the data structure with set of points
    points_in_grid - points that were used to build up a grid
    points_labelArr - array with points where gold standard evaluates to true 
    points_labelArr_len - length of points_labelArr_len
    y_hat_arr - array with algorithm output
    counts - array for storing output
    max_dist - maximuym expected distance - used to scale down the values
    """
    # tid = wp.tid()
    # # order threads by cell
    i = wp.hash_grid_point_id(grid, wp.tid())
    # # query point    
    # p = points_in_grid[i]
    p = points_in_grid[i]
    dist = float(1000000)
    weight = float(0)

    # # construct query around point p
    # neighbors = wp.hash_grid_query(grid, p, radius)

    for point_label_ind in range(0,points_labelArr_len):
        point_label=points_labelArr[point_label_ind]
        # compute distance to point
        d = wp.length(p - point_label)
        if((dist) >d):
            dist=d
            weight= y_hat_arr[int(p[0]),int(p[1]),int(p[2])]
        # if(i==0):
        #     print(p[0])
        #     print(p[1])
        #     print(p[2])
        #     print("diiizt ")
        #     print(d)
        #     print("weeight ")
        #     print(y_hat_arr[int(p[0]),int(p[1]),int(p[2])])



    counts[i] = (dist)*weight#*float(max_dist) # TODO(scale down the distance)
