import lerobot.common.datasets.push_dataset_to_hub.dobbe_format as dobbe_format

dobbe_format.check_format("D:\\2024-05-07--11-38-12-20240529T141713Z-001")
data_dict, episode_data_idx = dobbe_format.load_from_raw(
    "D:\\2024-05-07--11-38-12-20240529T141713Z-001", 30, False)
print(data_dict["timestamp"].shape)
