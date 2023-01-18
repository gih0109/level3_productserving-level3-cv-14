# dataset settings
dataset_type = "CocoDataset"
data_root = "D:/Workspace/project/productserving/data/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1024, 724), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 724),
        flip=True,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.0),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

classes = (
    "a0", "a1", "a2", "a3", "a4", "a5",
    # "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
    # "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20",
    # "q21", "q22", "q23", "q24", "q25", "q26", "q27", "q28", "q29", "q30",
)


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/train_13_20.json",
        img_prefix=data_root + "images/train_13_20",
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/val_21_23.json",
        img_prefix=data_root + "images/val_21_23",
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/val_21_23.json",
        img_prefix=data_root + "images/val_21_23",
        classes=classes,
        pipeline=test_pipeline,
    ),
)
# evaluation = dict(interval=1, metric="bbox", classwise=True)
