from mmcv import Config



base_info = Config.fromfile("{{ fileDirname }}/base_cfg.py")

final_model_name = base_info.model_work_name
final_dataset_name = base_info.dataset_work_name
final_schedule_name = base_info.schedule_work_name
# print(base_info)
_base_ = base_info.base_file_list
del base_info

# final_work_name = '{{fileBasenameNoExtension}}'

# checkpoint_config = dict(interval=3000, save_optimizer=True, by_epoch=False)
# evaluation = dict(interval=3000, save_image=True, gpu_collect=True)
visual_config = None
work_dir = f'./work_dirs/{final_dataset_name}/{final_schedule_name}/{final_model_name}'
find_unused_parameters = False
auto_resume = True


