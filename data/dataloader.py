import numpy as np
import torch
import argparse
from torch.utils.data import Dataset
from PIL import Image
import pickle
import glob
import cv2
import random
import os, sys
import math
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
import torch.utils.data
from config.dataconfig import Config as DataConfig

from data.fashion_base_function import get_random_params, get_transform

class Interhand26m(Dataset):
    def __init__(self,opt, is_inference, labels_required = False)->None:

        self.data_root = opt.path
        self.labels_required = labels_required

        self.split = 'train' if not is_inference else 'train'
        self.data_Lst = glob.glob(os.path.join(self.data_root, self.split, "*.pkl"))
        self.is_inference = is_inference
        self.scale_param = opt.scale_param if not is_inference else 0
        self.resolution = opt.resolution
            
        self.limbSeq = [[0, 1],[1, 2],[2, 3],[3, 4],[0, 5], [5, 6],[6, 7],[7, 8],[0, 9],[9, 10], 
                    [10, 11],[11, 12],[0, 13], [13, 14],[14, 15],[15, 16],[0, 17],[17, 18],[18, 19],[19, 20]]
        self.colors = [[255, 255, 255], [100, 0, 0], [150, 0, 0],  [200, 0, 0], [255, 0, 0,], [100, 100, 0], [150, 150, 0], [200,  200, 0], [255, 255, 0], 
                    [0, 100, 50], [0, 150, 75], [0, 200, 100], [0, 255, 125], [0, 50, 100], [0, 75, 150], [0,  100, 200], [0, 125, 255], [100, 0, 100], [150, 0, 150], [200, 0, 200], [255, 0, 255]]
        print("loading dataset: %s, split of %s total frame-view paired is %d"%("Interhand26m", self.split, len(self.data_Lst)))

    def __len__(self):
        return len(self.data_Lst)

    def __getitem__(self, index):
        path_item = self.data_Lst[index]
        data_Dic = pickle.load(open(path_item, "rb"), encoding="latin1")
        # print("number of view %d"%len(data_Dic))
        i = np.random.choice(list(range(1, len(data_Dic))))
        source_Dic = data_Dic["view%d"%i]

        ref_img = source_Dic["image"]
        ref_img = cv2.resize(ref_img, dsize=(self.resolution, self.resolution))
        ref_tensor, param = self.get_image_tensor(ref_img)
        if self.labels_required:
            label_ref_tensor = self.get_label_tensor(self.proj_joints(source_Dic["joint_c"], source_Dic["cam_param"]["cameraIn"]), ref_tensor, param) 
        # cv2.imshow("pose_dist", label_ref_tensor[:3].permute(1,2,0).numpy())
        # cv2.imshow("img_dist", (255*cv2.cvtColor(ref_tensor.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)).astype(np.uint8))
        # cv2.waitKey(-1)

        # load target, always at the first, different from zero123  
        target_Dic = data_Dic['view0']
        tgt_img = target_Dic["image"]
        tgt_img = cv2.resize(tgt_img, dsize=(self.resolution, self.resolution))
        target_image_tensor, param = self.get_image_tensor(tgt_img)
        if self.labels_required:
            target_label_tensor = self.get_label_tensor(self.proj_joints(target_Dic["joint_c"], target_Dic["cam_param"]["cameraIn"]), target_image_tensor, param) 

        # data arguement
        if not self.is_inference:
            if torch.rand(1) < 0.5:
                target_image_tensor = F.hflip(target_image_tensor)
                ref_tensor = F.hflip(ref_tensor)

                if self.labels_required:

                    target_label_tensor = F.hflip(target_label_tensor)
                    label_ref_tensor = F.hflip(label_ref_tensor)

        if self.labels_required:
            input_dict = {'target_skeleton': target_label_tensor,
                        'target_image': target_image_tensor,
                        'source_image': ref_tensor,
                        'source_skeleton': label_ref_tensor,
                        }
        else:
            input_dict = {'target_image': target_image_tensor,
                          'source_image': ref_tensor,
                         }

        return input_dict

    def get_image_tensor(self, cv_Arr):
        # with self.env.begin(write=False) as txn:
        #     key = f'{path}'.encode('utf-8')
        #     img_bytes = txn.get(key) 
        # buffer = BytesIO(img_bytes)
        # img = Image.open(cv2.cvtColor())

        img = Image.fromarray(cv2.cvtColor(cv_Arr, cv2.COLOR_BGR2RGB))
        param = get_random_params(img.size, self.scale_param)
        trans = get_transform(param, normalize=True, toTensor=True)
        img = trans(img)
        return img, param

    def proj_joints(self, joints_cam, camIn):
        jointpix = np.matmul(camIn, joints_cam.astype(np.float32).T).T
        jointpix = jointpix[:, :2] / jointpix[:, 2:]
        # jointpix = torch.tensor(jointpix, dtype=torch.float32)
        return jointpix

    def get_label_tensor(self, keypoint, img, param):
        canvas = np.zeros((img.shape[1], img.shape[2], 3)).astype(np.uint8)
        keypoint = self.trans_keypoins(keypoint, param, img.shape[1:])

        stickwidth = 1
        for i in range(21):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1 or np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(canvas, (int(x), int(y)), 1, self.colors[i], thickness=-1)
        joints = []
        for i in range(20):
            Y = keypoint[np.array(self.limbSeq[i]), 0]
            X = keypoint[np.array(self.limbSeq[i]), 1]            
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X or np.isnan(sum(X)) or np.isnan(sum(Y)):
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i+1])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            joint = np.zeros_like(cur_canvas[:, :, 0]) # 初始化值为0的单张map
            cv2.fillConvexPoly(joint, polygon, 255)   
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0) # 得到的joint其实就是非0即255的skeleton的mask
            # cv2.imshow("joint", joint)
            # cv2.waitKey(-1)
            joints.append(joint)
        pose = F.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))) #含有所有joints和skeleton的pose,不含背景

        tensors_dist = 0
        e = 1
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3) # skeleton与纯色背景颜色呼唤，skeleton由纯白转为纯黑
            im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8) # 像素值除以3，并约束到[0,255]之间
            tensor_dist = F.to_tensor(Image.fromarray(im_dist))
            tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
            e += 1

        # 搞不懂为什么要这么cat
        label_tensor = torch.cat((pose, tensors_dist), dim=0) 
        return label_tensor

    def trans_keypoins(self, keypoints, param, img_size):
        missing_keypoint_index = keypoints == -1
        
        # crop the white line in the original dataset
        # keypoints[:,0] = (keypoints[:,0]-40)

        # resize the dataset
        img_h, img_w = img_size
        scale_w = 1.0/256.0 * img_w
        scale_h = 1.0/256.0 * img_h

        if 'scale_size' in param and param['scale_size'] is not None:
            new_h, new_w = param['scale_size']
            scale_w = scale_w / img_w * new_w
            scale_h = scale_h / img_h * new_h
        

        if 'crop_param' in param and param['crop_param'] is not None:
            w, h, _, _ = param['crop_param']
        else:
            w, h = 0, 0

        keypoints[:,0] = keypoints[:,0]*scale_w - w
        keypoints[:,1] = keypoints[:,1]*scale_h - h
        keypoints[missing_keypoint_index] = -1
        return keypoints


def create_dataloader(opt, labels_required, is_inference):
    instance = Interhand26m(opt, is_inference, labels_required)
    phase = 'val' if is_inference else 'training'
    batch_size = opt.val.batch_size if is_inference else opt.train.batch_size
    print("%s dataset [%s] of size %d was created" %(phase, opt.type, len(instance)))
    
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=batch_size,
        # sampler=data_sampler(instance, shuffle=not is_inference),
        sampler=data_sampler(instance, shuffle=True),
        drop_last=not is_inference,
        num_workers=getattr(opt, 'num_workers', 0),
    )          
    return dataloader

def data_sampler(dataset, shuffle):
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)

def get_train_val_dataloader(opt, labels_required=False):

    val_dataset = create_dataloader(opt, labels_required = labels_required, is_inference=True)
    train_dataset = create_dataloader(opt, labels_required = labels_required, is_inference=False)

    return val_dataset, train_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--DataConfigPath', type=str, default=r'config\data.yaml')
    args = parser.parse_args()

    DataConf = DataConfig(args.DataConfigPath)
    val_dataset, train_dataset = get_train_val_dataloader(DataConf.data, labels_required = True)
    print("train split len:", len(train_dataset))
    print("val split len:", len(val_dataset))
    for iter_idx, datapac_i in enumerate(train_dataset):
        print(iter_idx)