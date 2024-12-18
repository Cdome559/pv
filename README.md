Enhanced 3D Object Detection through Extended RoI Pooling and Point-Voxel Feature Fusion
========================================================================================
The overall structure of the proposed model, the original point cloud is voxelized and key point sampling is performed. The key point features, voxel features and BEV features are obtained. After the key point aggregated voxel features, the fine-grained information is collected through the extended ROI pooling strategy for proposal generation and optimization.
![image](/docs/model.pdf)

Requirements
------------
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.7+
* PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
* CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)

training
--------
```
python train.py --cfg_file cfgs/kitti_models/pv.yaml
```
