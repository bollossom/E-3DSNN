# CUDA_VISIBLE_DEVICES=4,5 nohup torchrun --standalone --nproc_per_node=2 --rdzv_backend=c10d \
#   test.py \
#   --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml \
#   --batch_size 12 \
#   --ckpt /public/liguoqi/qxr/point/OpenPCDet/voxel_rcnn_car_84.54.pth \
#   > output_test_voxel.log
  CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1  \
  test.py \
  --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml \
  --batch_size 1 \
  --ckpt /public/liguoqi/qxr/point/OpenPCDet/output/kitti_models/voxel_rcnn_car/default/ckpt/checkpoint_epoch_79.pth \
  > output_test_voxelnet_snn_3dv79.log