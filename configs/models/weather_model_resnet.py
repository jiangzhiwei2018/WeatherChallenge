model_work_name = 'BasicWeather_ResnetGenerator_DCNv2_interpolate_MSELoss'

model = dict(
    type="BasicWeatherModel",
    generator=dict(type="BasicWeather",
                   generator_backbone_cfg=dict(type="ResnetGenerator", num_blocks=0),
                   upsample_type="interpolate", conv_cfg=dict(type='DCNv2')),
    pixel_loss=dict(
        type="MSELoss"  # smooth L1
        # type="CharbonnierLoss"  # smooth L1
        # type="CharbonnierLoss"  # smooth L1
    ),
    train_cfg=None,
    test_cfg=dict(metrics=("MSE",)),
)
cudnn_benchmark = True
