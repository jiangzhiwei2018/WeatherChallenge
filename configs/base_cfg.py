base_file_list = [
    './_base_/default_runtime.py',
    './models/weather_pan_net.py',
    # './models/weather_model.py',
    # './models/weather_model_resnet.py',
    # './schedules/ep100_sgd_lr1e-3_cosinR25.py',
    './schedules/ep300_adam_lr1e-3_cosinR300.py',
    './datasets/weather_datasets.py'
]

_base_ = base_file_list

