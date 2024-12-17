
tester
test = dict(type="ClsTester")
_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 16  # bs: total bs in all gpus
# batch_size_val = 8
empty_cache = False
enable_amp = False

# model settings
#3m
model = dict(
    type="DefaultClassifier",
    backbone_embed_dim=160,
    num_classes=40,
    backbone=dict(
        type="SNN_3dv",
        in_channels=6,
        num_classes=40,
        embed_channels=24,
        enc_channels=[24, 48, 96, 160],
        enc_depth=[1, 1, 1, 1], 
    ),
    criteria=[dict(type="CrossEntropyLoss", label_smoothing=0.2,loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 300
eval_epoch = 100
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.001, nesterov=True)
#direct training SGD better
# optimizer = dict(type="AdamW", lr=0.001,weight_decay=0.05)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)

# dataset settings
dataset_type = "ModelNetDataset"
data_root = "data/modelnet40_normal_resampled"
cache_data = False
class_names = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]

data = dict(
    num_classes=40,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=10000, mode="random"),
            # dict(type="CenterShift", apply_z=True),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "normal"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "normal"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "normal"],
            ),
        ],
        test_mode=True,
    ),
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

# tester
test = dict(type="ClsTester")
