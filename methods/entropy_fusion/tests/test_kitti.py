import io
import zipfile
import numpy as np
from PIL import Image
import pytest
import torch

from data.kitti_dataset import KittiFusionDataset, LOCAL_TO_RECT
from models.boxes import prepare_targets, encode_3d, decode_3d
from models.losses import detection_loss
from scripts.download_kitti_subset import RemoteZipReader


def test_kitti_projection_and_box_roundtrip_preserve_baseline_and_bottom_center(tmp_path):
    (tmp_path/'ImageSets').mkdir()
    (tmp_path/'ImageSets/train.txt').write_text('000000\n')
    for name in ('calib','image_2','velodyne','label_2'):
        (tmp_path/'training'/name).mkdir(parents=True)
    Image.new('RGB',(100,50),(100,120,140)).save(tmp_path/'training/image_2/000000.png')
    (tmp_path/'training/calib/000000.txt').write_text(
        'P2: 100 0 50 -50 0 100 25 0 0 0 1 0\n'
        'R0_rect: 1 0 0 0 1 0 0 0 1\n'
        'Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n\n')
    np.array([[10,0,0,.7]],dtype='<f4').tofile(tmp_path/'training/velodyne/000000.bin')
    (tmp_path/'training/label_2/000000.txt').write_text(
        'Car 0 0 0 30 15 70 40 2 2 4 0 1 10 0\n'
        'DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10\n')
    item = KittiFusionDataset(tmp_path,image_hw=(64,128))[0]
    assert np.allclose(item['projections']['lidar'][0].numpy(),[57.6,32,10])
    assert item['lidar'][2].max().item() == pytest.approx(.7)
    assert not item['radar_mask'].any() and not item['radar'].any()
    assert len(item['target']['ignore_boxes']) == 1
    target = prepare_targets(item['target'],item['metadata'],'cpu')
    assert len(target['values']) == 1 and not target['velocity_valid'].any()
    xyz,size,rotations,_ = decode_3d(encode_3d(target['values'],target['boxes']),target['boxes'],item['metadata'])
    rect_center = xyz.numpy() @ LOCAL_TO_RECT.T
    assert np.allclose(rect_center,[[0,0,10]],atol=1e-5)
    assert np.allclose(size.numpy(),[[2,4,2]])
    from pyquaternion import Quaternion
    length_axis = LOCAL_TO_RECT @ Quaternion(rotations[0]).rotation_matrix[:,0]
    assert np.allclose(length_axis,[1,0,0],atol=1e-6)


def test_fully_ignored_image_has_finite_zero_background_loss():
    anchors = torch.tensor([[0.,0.,16.,16.]])
    prediction = dict(anchors=anchors,logits=torch.zeros(1,1,2,requires_grad=True),
                      boxes=torch.zeros(1,1,4,requires_grad=True),boxes3d=torch.zeros(1,1,10,requires_grad=True),
                      attributes=torch.zeros(1,1,8,requires_grad=True))
    target = dict(boxes=torch.empty(0,4),labels=torch.empty(0,dtype=torch.long),annotations_3d=[],ignore_boxes=anchors)
    meta = dict(global_to_camera=np.eye(4),ego_to_global=np.eye(4),intrinsic=np.eye(3))
    losses = detection_loss(prediction,[target],[meta],torch.ones(1,1,16,16,dtype=torch.bool))
    assert losses['total'].item() == 0
    losses['total'].backward()
    assert not prediction['logits'].grad.any()


def test_remote_zip_range_reader_extracts_without_reading_entire_archive(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        archive.writestr('unused.bin',b'x'*(3*1024**2))
        archive.writestr('wanted.txt',b'camera lidar')
    payload = buffer.getvalue()
    class Response(io.BytesIO):
        def __init__(self,data,status,headers):
            super().__init__(data)
            self.status,self.headers = status,headers
    def open_request(request,timeout):
        if request.get_method() == 'HEAD':
            return Response(b'',200,{'Content-Length':str(len(payload)),'ETag':'test'})
        start,end = map(int,request.get_header('Range').removeprefix('bytes=').split('-'))
        return Response(payload[start:end+1],206,{'Content-Range':f'bytes {start}-{end}/{len(payload)}'})
    monkeypatch.setattr('urllib.request.urlopen',open_request)
    reader = RemoteZipReader('https://example.test/archive.zip')
    with zipfile.ZipFile(reader) as archive:
        assert archive.read('wanted.txt') == b'camera lidar'
    assert reader.transferred < len(payload)//2


def test_remote_reader_rejects_server_ignoring_range(monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {'Content-Length':'100','ETag':'test'}
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:Response(b'x'*100))
    reader = RemoteZipReader('https://example.test/archive.zip')
    with pytest.raises(IOError,match='exact byte range'):
        reader.read(1)
