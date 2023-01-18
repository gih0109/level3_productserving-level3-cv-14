_base_ = [
    './datasets/coco_detection.py',
    './models/cascade_rcnn_r50_pafpn.py',
    './schedules/schedule_adamw_1x_last.py',
    './default_runtime.py',
]

custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='xlarge',
        out_indices=[0,1,2,3],
        drop_path_rate=0.8,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[256, 512, 1024, 2048]))