import torch
import torch.nn.functional as F
import numpy as np


def get_pred_torch(probability):
    n,c,h,w=probability.shape
    _, max_indices = torch.max(probability, dim=1)
    max_indices = max_indices.view(-1, 1, h, w)
    return max_indices


def get_pred_with_softmax_torch(probability):
    p = F.softmax(probability,dim=1)
    n,c,h,w=p.shape
    _, max_indices = torch.max(p, dim=1)
    max_indices = max_indices.view(-1, 1, h, w)
    return max_indices


def get_depth_abs_err_torch(depth,gt):
    """
    gt 为 0 的地方应该不参与计算
    """
    abs_err = torch.abs(gt.to(torch.float32) -depth.to(torch.float32)).to(torch.float32)
    abs_err [abs_err > 1000] = 1000
    return abs_err

def get_depth_abs_err_conf_torch(err):
    err_conf = torch.zeros(err.shape,dtype=torch.float32).cuda()
    # err_conf[err<5] =  6
    # err_conf[torch.logical_and(err<10,err>=5)] = 5
    # err_conf[torch.logical_and(err<20,err>=10)] = 4
    # err_conf[torch.logical_and(err<50,err>=20)] = 3
    # err_conf[torch.logical_and(err<70,err>=50)] = 2
    # err_conf[torch.logical_and(err<100,err>=70)] = 1
    # err_conf[err>=100] = 0
    
    # err_conf[err<5] =  7
    # err_conf[torch.logical_and(err<10,err>=5)] = 6
    # err_conf[torch.logical_and(err<20,err>=10)] = 5
    # err_conf[torch.logical_and(err<30,err>=20)] = 4
    # err_conf[torch.logical_and(err<50,err>=30)] = 3
    # err_conf[torch.logical_and(err<70,err>=50)] = 2
    # err_conf[torch.logical_and(err<100,err>=70)] = 1
    # err_conf[err>=100] = 0
    
    
    err_conf[err<10] =  7
    err_conf[torch.logical_and(err<22,err>=10 )] = 6
    err_conf[torch.logical_and(err<40,err>=22 )] = 5
    err_conf[torch.logical_and(err<60,err>=40 )] = 4
    err_conf[torch.logical_and(err<75,err>=60 )] = 3
    err_conf[torch.logical_and(err<90,err>=75 )] = 2
    err_conf[torch.logical_and(err<100,err>=90)] = 1
    err_conf[err>=100] = 0
    # err_conf.astype(torch.uint8)
    return err_conf


def get_depth_abs_err_np(depth,gt):
    """
    gt 为 0 的地方应该不参与计算
    """
    abs_err = np.abs(gt.astype(np.int32) -depth.astype(np.int32)).astype(np.uint16)
    abs_err [abs_err > 1000] = 1000
    return abs_err

def get_depth_abs_err_conf_np(err):
    err_conf = np.zeros(err.shape,dtype=np.uint8)
    # err_conf[err<5] =  6
    # err_conf[np.logical_and(err<10,err>=5)] = 5
    # err_conf[np.logical_and(err<20,err>=10)] = 4
    # err_conf[np.logical_and(err<50,err>=20)] = 3
    # err_conf[np.logical_and(err<70,err>=50)] = 2
    # err_conf[np.logical_and(err<100,err>=70)] = 1
    # err_conf[err>=100] = 0
    # err_conf[err<5] =  7
    # err_conf[np.logical_and(err<10,err>=5)] = 6
    # err_conf[np.logical_and(err<20,err>=10)] = 5
    # err_conf[np.logical_and(err<30,err>=20)] = 4
    # err_conf[np.logical_and(err<50,err>=30)] = 3
    # err_conf[np.logical_and(err<70,err>=50)] = 2
    # err_conf[np.logical_and(err<100,err>=70)] = 1
    # err_conf[err>=100] = 0
    # err_conf.astype(np.uint8)
    err_conf[err<10] =  7
    err_conf[np.logical_and(err<22,err>=10 )] = 6
    err_conf[np.logical_and(err<40,err>=22 )] = 5
    err_conf[np.logical_and(err<60,err>=40 )] = 4
    err_conf[np.logical_and(err<75,err>=60 )] = 3
    err_conf[np.logical_and(err<90,err>=75 )] = 2
    err_conf[np.logical_and(err<100,err>=90)] = 1
    err_conf[err>=100] = 0
    err_conf.astype(np.uint8)
    return err_conf
