checkpoint_config = dict(
                         interval=1,
                         by_epoch=True,
                         save_optimizer=True,
                         out_dir=None,
                         max_keep_ckpts=1,
                         save_last=True,
                         )
# yapf:disable
log_config = dict(
    interval=50,
    # by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir=None)
    ])
# yapf:enable
custom_hooks = [
    # dict(type='NumClassCheckHook')
]
gpu_ids = [0]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
evaluation = dict(interval=1, gpu_collect=True, less_keys=("MSE", ), save_best="MSE")
test_evaluation = dict(interval=1, gpu_collect=True, greater_keys=None, save_best=None,
                       tensor_board_figure_out=True,
                       tensor_board_figure_out_dir=None,
                       eval_tag="test", do_evaluate=False,
                       write_img=True,
                       test_val_steps=[1]
                       )






