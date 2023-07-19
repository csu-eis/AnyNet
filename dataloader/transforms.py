# Author: Huan Yang
# Date: 2023/07/03
# Description: Brief description of the code's purpose

# Copyright (c) 2023 Huan Yang
# All rights reserved.


import numpy as np
import cv2
import torch
from typing import Any, Optional,Union,Tuple
import copy


class Compose:
    def __init__(self,ops) -> None:
        self.ops = ops
        
    def __call__(self, input:np.ndarray):
        out = None
        for op in self.ops:
            if type(out)!=type(None):
                out = op(out)
            else:
                out = op(input)
        return out
    
class OCVSobel:
    def __init__(self) -> None:
        pass
    def __call__(self, x:np.ndarray) -> np.ndarray:
        gray = copy.deepcopy(x)
        if len(gray.shape)==3 and gray.shape[2]==3:
            gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值和方向
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        # gradient_direction = np.arctan2(sobely, sobelx)
        return gradient_magnitude      

class ZScoreNormalize:
    
    def __init__(self,mean,std) -> None:
        self.mean = mean
        self.std = std
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        input = copy.deepcopy(x)
        out = (input-self.mean)/self.std
        return out
    
class AntiZScoreNormalize:
    
    def __init__(self,mean,std) -> None:
        self.mean = mean
        self.std = std
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        input = copy.deepcopy(x)
        out = input * self.std + self.mean
        return out
    
    
class MinMaxNormalize:
    
    def __init__(self,min_val,max_val) -> None:
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        input = copy.deepcopy(x)
        out = cv2.normalize(input, None, self.min_val, self.max_val, cv2.NORM_MINMAX)
        return out
   
class OCVResize:
    
    def __init__(self,dsize: Union[list,tuple]=None) -> None:
        self.dsize = dsize
        
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        input = copy.deepcopy(x)
        out  = cv2.resize(input,self.dsize,None)
        if len(out.shape)==2:
            out = out[:,:,np.newaxis]
        return out


class ToTensor:
    def __init__(self,dtype=torch.float32) -> None:
        self.dtype = dtype
        
    
    def __call__(self, x:np.ndarray) -> torch.Tensor:
        input = copy.deepcopy(x)
        # 将n,h,w,c -> n,h,w,c
        if len(input.shape)==2:
            input = input[:,:,np.newaxis] 
        out = input.transpose([2,0,1])
        out = torch.from_numpy(out).to(self.dtype)
        return out

class BGR2Gray:
    def __init__(self) -> None:
        self.weights = np.array([ 0.1140, 0.5870,0.2989],dtype=np.float32)
        
    
    def __call__(self, x:np.ndarray) -> torch.Tensor:
        input = copy.deepcopy(x)
        # 输入b,g,r,[h,w,c]的图像
        luminance = np.sum(input * self.weights, axis=2)
        return luminance



class ToNumpy:
    def __init__(self,dtype=np.float32) -> None:
        self.dtype = dtype
        
    def __call__(self, input:torch.Tensor) -> np.ndarray:
        n,c,h,w = input.shape
        out = input.detach().to('cpu').numpy().transpose([0,2,3,1]).astype(self.dtype)
        return out


