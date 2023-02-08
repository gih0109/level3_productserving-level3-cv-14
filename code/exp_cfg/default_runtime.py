checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="MMDetWandbHook",
            init_kwargs={
                "project": "final_project",
                "name": "base model",
                "entity": "level2_object-detection-cv14",
            },
            interval=100,
            log_checkpoint=False,
            log_checkpoint_metadata=True,
            num_eval_images=0,
        ),
    ],
)


# yapf:enable
custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"

evaluation = dict(interval=1, metric="bbox", classwise=True, save_best="bbox_mAP")

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(base_batch_size=64)