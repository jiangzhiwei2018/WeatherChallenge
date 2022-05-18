# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer
schedule_work_name = '{{fileBasenameNoExtension}}'
optimizer = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99)
    )
)
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
optimizer_config = None
# lr_config = None
lr_config = dict(
    # warmup='linear',  # ['constant', 'linear', 'exp']
    # warmup_iters=1000,
    # warmup_ratio=0.1,
    # warmup_by_epoch=False,

    policy='CosineRestart',
    by_epoch=True,
    periods=[300],
    restart_weights=[1],
    min_lr=0.)

runner = dict(type='EpochBasedRunner', max_epochs=300)
