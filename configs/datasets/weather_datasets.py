dataset_work_name = "weather_Precip_Radar_Wind_bs4"

train_pipeline = [
    dict(
        type="LoadImages",
        keys=("Precip", "Radar", "Wind")
    ),
    dict(
        type="Collect",
        keys=["input_img", "gt"],
        meta_keys=["num_range"]
    )
]

val_pipeline = [
    dict(
        type="LoadImages",
        keys=("Precip", "Radar", "Wind")
    ),
    dict(
        type="Collect",
        keys=["input_img",  "gt"], meta_keys=["num_range"]
    )
]

test_pipeline = [
    dict(
        type="LoadImages",
        keys=("Precip", "Radar", "Wind")
    ),
    dict(
        type="Collect",
        keys=["input_img"], meta_keys=["num_range"]
    )
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 1 gpu
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type="WeatherDataset",
        dataset_folder_name="Train",
        pipeline=train_pipeline,
        dataset_prefix=r"G:\LargeDataset\TIANCHI\weather",
        data_type_name=("Precip", "Radar", "Wind"),
        test_mode=False
    ),
    val=dict(
        type="WeatherDataset",
        dataset_folder_name="TestA",
        pipeline=val_pipeline,
        dataset_prefix=r"G:\LargeDataset\TIANCHI\weather",
        data_type_name=("Precip", "Radar", "Wind"),
        test_mode=False
    ),
    test=dict(
        type="WeatherDataset",
        dataset_folder_name="TestB1",
        pipeline=test_pipeline,
        dataset_prefix=r"G:\LargeDataset\TIANCHI\weather",
        data_type_name=("Precip", "Radar", "Wind"),
        test_mode=True
    )
)



