
CUDA_VISIBLE_DEVICES=7,0 nohup torchrun --standalone --nproc_per_node=2 \
  train.py \
  --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml \
  > train.log

CUDA_VISIBLE_DEVICES=7,0 nohup torchrun --standalone --nproc_per_node=2 \
  train.py \
  --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml \
  > train.log

