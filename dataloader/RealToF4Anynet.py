from .RealToF import RealToFDataset
import json
import numpy as np
import cv2
import dataloader.transforms as T
import torch


class RealToF4Anynet(RealToFDataset):
    
    def __init__(self, json_file, num=None,dsize=[128,128]):
        super(RealToF4Anynet,self).__init__(json_file,num=num,dsize=dsize)


    def __getitem__(self, index):
        stereo_left = cv2.imread(self.file_list[index]["stereo_left_file"]).astype(np.float32)
        stereo_right = cv2.imread(self.file_list[index]["stereo_right_file"]).astype(np.float32)
        stereo_q = torch.from_numpy(np.load(self.file_list[index]["stereo_q"]).astype(np.float32)).view(1,4,4)
        rsgt_depth = self.op4resize(np.load(self.file_list[index]["rsgt_depth_file"]).astype(np.float32))
        
        disp_left = self.convert_depth_to_disp(rsgt_depth,stereo_q)
        norm_stereo_left = self.op4rgb(stereo_left)
        norm_stereo_right = self.op4rgb(stereo_right)
        raw_stereo_left = self.op4resize(stereo_left)
        raw_stereo_right = self.op4resize(stereo_right)
        return {
            "stereo_left":norm_stereo_left,
            "stereo_right":norm_stereo_right,
            "disp_left":disp_left,
            "raw_stereo_left":raw_stereo_left,
            "raw_stereo_right":raw_stereo_right
        }
