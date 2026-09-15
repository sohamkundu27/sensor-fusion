"""KITTI camera/LiDAR car inputs with explicit rectified-camera geometry.

The internal local frame is X=forward, Y=left, Z=up relative to rectified camera 0.
It is not a geographic/global pose. KITTI labels are bottom-centered in camera 0;
P2's translation is retained when projecting into camera 2. Radar is unavailable.
"""
from pathlib import Path
import re
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import torch
from torch.utils.data import Dataset
from .geometry import transform_points, project_points, filter_visible_points, rasterize_features

LOCAL_TO_RECT = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


def read_calibration(path):
    entries = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        name, values = line.split(':',1)
        entries[name] = np.asarray([float(x) for x in values.split()])
    p2 = entries['P2'].reshape(3,4)
    rect = np.eye(4)
    rect[:3,:3] = entries['R0_rect'].reshape(3,3)
    velo = np.eye(4)
    velo[:3] = entries['Tr_velo_to_cam'].reshape(3,4)
    intrinsic = p2[:,:3]
    offset = np.linalg.solve(intrinsic,p2[:,3])
    camera2 = np.eye(4)
    camera2[:3,3] = offset
    if not all(np.isfinite(x).all() for x in (p2,rect,velo,offset)):
        raise ValueError(f'Nonfinite calibration: {path}')
    return intrinsic, offset, camera2 @ rect @ velo, p2, rect @ velo


def read_labels(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        parts = line.split()
        if len(parts) != 15: raise ValueError(f'Expected 15 KITTI label fields: {path}')
        values = np.asarray([float(x) for x in parts[1:]],dtype=np.float64)
        if not np.isfinite(values).all(): raise ValueError(f'Nonfinite label: {path}')
        rows.append(dict(name=parts[0],truncation=values[0],occlusion=int(values[1]),
                         box=values[3:7],hwl=values[7:10],bottom_center=values[10:13],rotation_y=values[13]))
    return rows


class KittiFusionDataset(Dataset):
    def __init__(self, root, split='train', image_hw=(384,640), max_depth=100.):
        self.root = Path(root).expanduser().resolve()
        if split not in ('train','val'): raise ValueError('Use a labeled train/val split')
        self.ids = (self.root/'ImageSets'/f'{split}.txt').read_text().splitlines()
        if not self.ids or len(set(self.ids))!=len(self.ids) or not all(re.fullmatch(r'\d{6}',x) for x in self.ids):
            raise ValueError('Invalid or empty KITTI split')
        self.image_hw = tuple(image_hw)
        if len(self.image_hw)!=2 or any(x <= 0 or x % 16 for x in self.image_hw):
            raise ValueError('Image dimensions must be positive multiples of 16')
        self.max_depth = max_depth

    def __len__(self): return len(self.ids)

    def __getitem__(self, index):
        sample = self.ids[index]
        base = self.root/'training'
        intrinsic, offset, velo_to_camera, p2, velo_to_rect = read_calibration(base/'calib'/f'{sample}.txt')
        h,w = self.image_hw
        image_path = base/'image_2'/f'{sample}.png'
        with Image.open(image_path) as source:
            image = source.convert('RGB')
            ow,oh = image.size
            scale = min(w/ow,h/oh)
            rw,rh = round(ow*scale),round(oh*scale)
            left,top = (w-rw)//2,(h-rh)//2
            canvas = Image.new('RGB',(w,h))
            canvas.paste(image.resize((rw,rh),Image.Resampling.BILINEAR),(left,top))
            rgb = np.asarray(canvas,dtype=np.float32).transpose(2,0,1).copy()/255
        affine = np.array([[rw/ow,0,left],[0,rh/oh,top],[0,0,1.]])
        mask = np.zeros((1,h,w),dtype=bool)
        mask[:,top:top+rh,left:left+rw] = True
        raw = np.fromfile(base/'velodyne'/f'{sample}.bin',dtype='<f4')
        if raw.size%4 or not np.isfinite(raw).all(): raise ValueError('Invalid KITTI point cloud')
        cloud = raw.reshape(-1,4)
        camera_points = transform_points(cloud[:,:3],velo_to_camera)
        points,indices = project_points(camera_points,intrinsic,(oh,ow),1.,self.max_depth,return_indices=True)
        if len(points): points[:,:2] = (np.column_stack((points[:,:2],np.ones(len(points)))) @ affine.T)[:,:2]
        points,valid = filter_visible_points(points,self.image_hw,return_mask=True)
        indices = indices[valid]
        # KITTI reflectance is already [0,1], unlike nuScenes intensity/255.
        features = np.column_stack((points[:,2]/self.max_depth,
                                    np.clip((cloud[indices,2]+3)/8,0,1),
                                    np.clip(cloud[indices,3],0,1))).astype(np.float32)
        lidar,lidar_mask = rasterize_features(points,features,self.image_hw,mask)
        g2c = np.eye(4)
        g2c[:3,:3],g2c[:3,3] = LOCAL_TO_RECT,offset
        boxes,ignore,annotations = [],[],[]
        labels = read_labels(base/'label_2'/f'{sample}.txt')
        for number,row in enumerate(labels):
            box = row['box'].copy()
            box[[0,2]] = np.clip(box[[0,2]],0,ow)*affine[0,0]+left
            box[[1,3]] = np.clip(box[[1,3]],0,oh)*affine[1,1]+top
            if box[2]<=box[0] or box[3]<=box[1]: continue
            if row['name'] in ('DontCare','Van'):
                ignore.append(box)
                continue
            if row['name'] != 'Car': continue
            height,width,length = row['hwl']
            center = row['bottom_center'].copy()
            center[1] -= height/2
            if min(height,width,length)<=0 or not 1 < (center+offset)[2] <= self.max_depth:
                ignore.append(box)
                continue
            yaw_local = -np.pi/2-row['rotation_y']
            annotations.append(dict(token=f'{sample}:{number}', translation=(LOCAL_TO_RECT.T @ center).tolist(),
                                    size=[width,length,height],rotation=Quaternion(axis=[0,0,1],radians=yaw_local).elements.tolist(),
                                    supervise_3d=True,velocity_valid=False,velocity_global=[0,0,0],attribute_names=[]))
            boxes.append(box)
        target = dict(boxes=torch.tensor(np.asarray(boxes,dtype=np.float32).reshape(-1,4)),
                      labels=torch.ones(len(boxes),dtype=torch.long),annotations_3d=annotations,
                      ignore_boxes=torch.tensor(np.asarray(ignore,dtype=np.float32).reshape(-1,4)))
        return dict(camera=torch.from_numpy(rgb),lidar=torch.from_numpy(lidar),radar=torch.zeros_like(torch.from_numpy(lidar)),
                    camera_mask=torch.from_numpy(mask),lidar_mask=torch.from_numpy(lidar_mask),radar_mask=torch.zeros_like(torch.from_numpy(mask)),
                    lidar_depth_m=torch.from_numpy(lidar[:1]*self.max_depth),radar_depth_m=torch.zeros(1,h,w),target=target,
                    metadata=dict(dataset='kitti',sample_token=sample,camera_channel='image_2',camera_path=str(image_path),
                                  image_hw=self.image_hw,original_hw=(oh,ow),intrinsic=affine@intrinsic,
                                  intrinsic_original=intrinsic,pixel_transform=affine,global_to_camera=g2c,ego_to_global=np.eye(4),
                                  coordinate_note='Internal local frame, not a geographic pose',p2=p2,velo_to_rect=velo_to_rect,
                                  input_lidar_points=len(cloud),projected_lidar_points=len(points)),
                    projections=dict(lidar=torch.from_numpy(points),radar=torch.empty(0,3)))
