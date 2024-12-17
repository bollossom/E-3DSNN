# #Semantic KITTI
# CUDA_VISIBLE_DEVICES=0 sh scripts/train.sh -g 1 -d semantic_kitti -c semseg-oacnn-v1m1-0-base -n semseg-oacnn-v1m1-0-base


# ModelNet
# CUDA_VISIBLE_DEVICES=7 sh scripts/train.sh -g 1 -d modelnet40 -c oacnns_v1m1_base -n oacnns_v1m1_base
CUDA_VISIBLE_DEVICES=0 sh scripts/train.sh -g 1 -d modelnet40 -c cls-E3DSNN-v1m1-0-T -n E_3DSNN

# CUDA_VISIBLE_DEVICES=0 sh scripts/train.sh -g 1 -d semantic_kitti -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
