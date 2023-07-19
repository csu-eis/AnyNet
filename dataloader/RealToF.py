from torch.utils.data import Dataset, DataLoader
import json
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from utils.postprocess_util import get_depth_abs_err_conf_np,get_depth_abs_err_np

import dataloader.transforms as T
from torchvision import transforms
import os

class RealToFDataset(Dataset):

    def __init__(self, json_files, num=None,dsize=[240,180]):
        super().__init__()
        self.file_list = []
        self.dsize = dsize
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        with open(json_files[0], "r") as f:
            obj =json.load(f)
            
            self.file_list = obj["data"]
            if  num is not None:
                self.file_list = self.file_list [:min(num,len(self.file_list))]
            self.mean_rgb,self.std_rgb = obj["rgb_norm_param"]["mean"],obj["rgb_norm_param"]["std"]
            self.mean_norm,self.std_norm = obj["normal_norm_param"]["mean"],obj["normal_norm_param"]["std"]
            self.mean_conf,self.std_conf = obj["confidence_norm_param"]["mean"],obj["confidence_norm_param"]["std"]
            self.mean_depth,self.std_depth = obj["depth_norm_param"]["mean"],obj["depth_norm_param"]["std"]
        for json_file in json_files[1:]:
            with open(json_file, "r") as f:
                obj =json.load(f)
                self.file_list.extend(obj["data"])
        
        
        self.op4rgb = T.Compose([
            T.ZScoreNormalize(self.mean_rgb,self.std_rgb),
            T.MinMaxNormalize(0,1),
            T.OCVResize(dsize),
            T.ToTensor()
        ])
        self.op4depth = T.Compose([
            T.ZScoreNormalize(self.mean_depth,self.std_depth),
            T.MinMaxNormalize(0,1),
            T.OCVResize(dsize),
            T.ToTensor()
        ])
        self.op4conf = T.Compose([
            T.ZScoreNormalize(self.mean_conf,self.std_conf),
            T.MinMaxNormalize(0,1),
            T.OCVResize(dsize),
            T.ToTensor()
        ])
        self.op4norm = T.Compose([
            T.ZScoreNormalize(self.mean_norm,self.std_norm),
            T.MinMaxNormalize(0,1),
            T.OCVResize(dsize),
            T.ToTensor()
        ])
        
        self.op4resize = T.Compose([
            T.OCVResize(dsize),
            T.ToTensor()
        ])

        self.op4totensor = T.Compose([
            T.ToTensor()
        ])
        
        self.sobel = T.Compose([
            T.OCVSobel(),
            T.MinMaxNormalize(0,1),
            T.OCVResize(dsize),
            T.ToTensor()
        ])
        

    def convert_depth_to_disp(self,depth,stereo_q):
        tmp = torch.div(stereo_q[0,2,3] , depth) - stereo_q[0,3,3]
        disp = tmp / stereo_q[0,3,2]
        disp[torch.isinf(disp)]=0
        return disp
    
    
    def generate_indepth_data(self,stereo_left,tof_depth):
        depth_scale_factor = 1000.0
        NEAR_CLIP = 0.1
        self.depth_resize = transforms.Resize((256, 320), interpolation=Image.BILINEAR)
        self.color_resize = transforms.Resize((256, 320), interpolation=Image.BICUBIC)
        self.rgb_transforms = transforms.Compose((
            # Inception Color Jitter
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2, hue=0.06),
            transforms.ToTensor(),
            # Do not use PCA to change lighting - image looks terrible and out of range
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ))
        self.original_rgb_to_tensor = transforms.Compose((
                transforms.ToTensor(),
            ))
        color_image = Image.fromarray(cv2.cvtColor(stereo_left.astype(np.uint8),cv2.COLOR_BGR2RGB))
        depth_image = Image.fromarray(np.squeeze(tof_depth.astype(np.int32),axis=2),mode="I")
        
        color_image = self.color_resize(color_image)
        depth_image = self.depth_resize(depth_image)
        
        depth_image = np.array(depth_image)
        depth_image = depth_image / depth_scale_factor
        depth_image = torch.unsqueeze(torch.from_numpy(depth_image.astype(np.float32)), dim=0)
        with torch.no_grad():
            # Intentionally make the input of the network smaller
            depth_image = depth_image / 16
            depth_image[depth_image < NEAR_CLIP / 16.0] = 0.0
        original_color_image = color_image
        color_image = self.rgb_transforms(color_image)
        original_color_tensor = self.original_rgb_to_tensor(original_color_image)

        return color_image, depth_image,depth_image, original_color_tensor
     
    
    # TODO 优化这里的变量生成
    def __getitem__(self, index):
        tof_depth = np.load(self.file_list[index]["tof_depth_file"]).astype(
            np.float32)
        tof_depth_conf = np.load(self.file_list[index]["tof_depth_conf_file"]).astype(
            np.float32)


        rsgt_depth = np.load(
            self.file_list[index]["rsgt_depth_file"]).astype(np.float32)

        tof_normal = np.load(
            self.file_list[index]["tof_normal_file"]).astype(np.float32)

        stereo_left = cv2.imread(self.file_list[index]["stereo_left_file"]).astype(np.float32)
        
        _,ext=os.path.splitext(self.file_list[index]["stereo_left_file_640x480"])
        if ext==".png" or ext==".jpg":
            stereo_left_640x480 = cv2.imread(self.file_list[index]["stereo_left_file_640x480"]).astype(np.float32)
        elif ext ==".npy":
            stereo_left_640x480 = np.load(self.file_list[index]["stereo_left_file_640x480"]).astype(np.float32)
        
        stereo_right = cv2.imread(self.file_list[index]["stereo_right_file"]).astype(np.float32)
        
        
        stereo_q = torch.from_numpy(np.load(self.file_list[index]["stereo_q"]).astype(np.float32)).view(1,4,4)
        
        
        
        tof_err = get_depth_abs_err_np(tof_depth,rsgt_depth).astype(np.float32)
        tof_err_conf = get_depth_abs_err_conf_np(tof_err).astype(np.float32)
        
        

        norm_tof_depth_conf = self.op4conf(tof_depth_conf)
        raw_tof_conf = self.op4resize(tof_depth_conf)
        norm_tof_depth = self.op4depth(tof_depth)
        raw_tof_depth = self.op4resize(tof_depth)
        

        norm_stereo_left = self.op4rgb(stereo_left)
        norm_stereo_right = self.op4rgb(stereo_right)
        raw_stereo_left = self.op4resize(stereo_left)

        grad_stereo_left = self.sobel(stereo_left)
        norm_gt_depth = self.op4depth(rsgt_depth)           
        norm_tof_normal = self.op4norm(tof_normal)
        raw_gt_conf = self.op4resize(tof_err_conf)
        raw_gt_depth = self.op4resize(rsgt_depth)
        
        raw_stereo_left = self.op4resize(stereo_left)
        raw_stereo_right = self.op4resize(stereo_right)
        disp_left = self.convert_depth_to_disp(raw_gt_depth,stereo_q)
        raw_tof_normal = self.op4resize(tof_normal)
        raw_tof_depth_conf = self.op4resize(tof_depth_conf)
       
        indepth_color_image, indepth_depth_image,indepth_depth_target, indepth_original_color_tensor = self.generate_indepth_data(stereo_left,tof_depth)
        stereo_left_640x480 = self.op4totensor(stereo_left_640x480)
        return {
         
            "stereo_left":norm_stereo_left,
            "stereo_right":norm_stereo_right,
            "tof_depth": norm_tof_depth,
            "tof_normal": norm_tof_normal,
            "tof_conf":norm_tof_depth_conf,
            "tof_err": tof_err,
            "gt_depth":norm_gt_depth,
            "gt_conf": raw_gt_conf,
            "raw_tof_conf": raw_tof_conf,
            "raw_tof_depth":raw_tof_depth,
            "raw_gt_depth":raw_gt_depth,
            "raw_stereo_left":raw_stereo_left,
            "stereo_q":stereo_q,
            "grad_stereo_left":grad_stereo_left,
            "raw_stereo_left":raw_stereo_left,
            "raw_stereo_right":raw_stereo_right,
            "raw_tof_normal":raw_tof_normal,
            "disp_left":disp_left,
            "indepth_color_input":indepth_color_image,
            "indepth_depth_input":indepth_depth_image,
            "indepth_depth_target":indepth_depth_target,
            "indepth_original_color_tensor":indepth_original_color_tensor,
            "stereo_left_640x480":stereo_left_640x480
        }

    def __len__(self):
        return len(self.file_list)


