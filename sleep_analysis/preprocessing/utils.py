import mesa_data_importer as mesa
import pandas as pd


def extract_edf_channel(path, subj_id, channel, tmin=None, tmax=None):
    if tmin and tmax:
        edf = mesa.load_edf(path, subj_id).crop(tmin=tmin, tmax=tmax)
    else:
        edf = mesa.load_edf(path, subj_id)

    channel_data = edf.pick_channels([channel])
    data = channel_data.get_data()[0, :]

    time, epochs = _create_datetime_index(channel_data.info["meas_date"], times_array=channel_data.times)
    if channel == "EKG":
        data = pd.DataFrame(data, index=time).rename(columns={0: "ecg"})
    else:
        data = pd.DataFrame(data, index=time).rename(columns={0: "resp"})
    return data, epochs


def _create_datetime_index(starttime, times_array):
    starttime_s = starttime.timestamp()  # * 1000000000
    times_array = times_array + starttime_s
    datetime_index = pd.to_datetime(times_array, unit="s")
    epochs = _generate_epochs(datetime_index)
    return datetime_index, epochs


def _generate_epochs(datetime_index):
    # 20260729 - rdwang: 作者对epoch编号的处理存在bug，start_time = 20:29:59和start_time = 20:30:01仅差2s，但epoch编号会错位15s，举例：
    # 实际时间点20:30:14，start_time = 20:29:59时，epochs_30s被赋值20:30:00，epoch_clear = 1 / 30，epochs = 0
    # 实际时间点20:30:16，start_time = 20:29:59时，epochs_30s被赋值20:30:30，epoch_clear = 31 / 30，epochs = 1
    # epochs 1 此时对应的时间范围是 20:30:15-20:30:44
    # 实际时间点20:30:14，start_time = 20:30:01时，epochs_30s被赋值20:30:00，epoch_clear = -1 / 30，epochs = 0
    # 实际时间点20:30:16，start_time = 20:30:01时，epochs_30s被赋值20:30:30，epoch_clear = 29 / 30，epochs = 0
    # 只有当实际时间20:30:45，start_time = 20:30:01时，epochs_30s被赋值20:31:00，epoch_clear = 59 / 30，epochs = 1
    # epochs 1 此时对应的时间范围是 20:30:45 - 20:31:14
    # start_time仅仅差2s，但却导致epoch覆盖范围相差 30s 
    start_time = datetime_index[0]
    epochs_30s = datetime_index.round("30s")

    epochs_clear = (epochs_30s - start_time).total_seconds()
    # epochs_clear.total_seconds()

    epochs_clear = epochs_clear / 30
    epochs = epochs_clear.astype(int)
    return epochs
