model_work_name = 'BasicWeather_SegPSPNet'

model = dict(
    type="BasicWeatherModel",
    generator=dict(type="BasicWeather",
                   generator_backbone_cfg=dict(type="SegPSPNet",
                                               in_channels=60,
                                               classes=60,
                                               encoder_weights=None,
                                               encoder_name="resnet34")
                   ),
    pixel_loss=dict(
        type="MSELoss"  # smooth L1
        # type="CharbonnierLoss"  # smooth L1
        # type="CharbonnierLoss"  # smooth L1
    ),
    train_cfg=None,
    test_cfg=dict(metrics=("MSE",)),
)
cudnn_benchmark = True
