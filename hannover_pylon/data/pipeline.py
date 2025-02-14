# data_pipeline.py

import torch
from torchdata.datapipes import iter as it
from config import settings
import numpy as np
from scipy.io import loadmat
import hannover_pylon.data.utils as ut
from pathlib import Path

# Ensure that DILL is available for pickling complex objects
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
def custom_collate_fn(batch):
    """
    Custom collate function to combine data from multiple files into a single batch.
    Each field is handled appropriately.
    """
    # Initialize lists to hold batched data
    file_paths = []
    dates = []
    corrupted_flags = []
    time_series = []
    fs_list = []
    channel_names_list = []
    channel_units_list = []
    levels = []
    directions = []
    sensors = []

    # Iterate over each data dictionary in the batch
    for data in batch:
        file_data = data['file_data']
        time_series.extend((file_data['data']))  # Assuming data is converted to tensor for processing
        number_of_samples = len(file_data['data'])
        file_paths.append(data['file_path'])
        dates.extend([data['date']]*number_of_samples)
        corrupted_flags.extend([data['file_corrupted']]*number_of_samples)
        
        # Access the file_data dictionary for relevant fields
        fs_list.append(file_data['fs'])
        channel_names_list.extend(file_data['channel_names'])
        channel_units_list.extend(file_data['channel_units'])
        levels.extend(file_data['level'])
        
        # Handle directions and sensors, ensuring None values are preserved
        directions.extend(file_data['direction'])
        sensors.extend(file_data['sensor'])

    # Convert lists to tensors where appropriate (e.g., psd_data)
    time_series = torch.from_numpy(np.stack(time_series,dtype=np.float32))
    data= {'time_series': time_series,
           'channel_names': channel_names_list, 'channel_units': channel_units_list, 
           'levels': levels, 'directions': directions, 'sensors': sensors,
           'corrupted_flags': corrupted_flags, 'dates': dates}

    # Combine the data into a single dictionary for the batch
    batch_data = {
        'file_paths': file_paths,
        'fs': fs_list,
        'data': data 
    }

    return batch_data


def build_pipeline(n_fft= 16392):
    dp =it.FileLister(root = settings.path.raw,recursive=True, masks='*.mat')
    dp = dp.map(ut.process_file_path)
    dp_acc, dp_meta_met, dp_meta_struct, dp_corrupted = dp.demux(num_instances=4, \
        classifier_fn= ut.demux_data,buffer_size=70_000)
    dp_corrupted = dp_corrupted.map(ut.readfile)
    tracker = ut.TrackCorruptedFiles()
    dp_corrupted = dp_corrupted.map(tracker)
    list(dp_corrupted)
    tracker.setup()
    dp_acc = dp_acc.map(tracker)
    dp_acc = dp_acc.map(ut.readfile)
    dp_acc = dp_acc.map(ut.extract_channel_info)
    dp_acc = dp_acc.batch(30)
    dp_acc = dp_acc.collate(custom_collate_fn)
    welch = ut.Welch(n_fft=n_fft)
    dp_acc = dp_acc.map(welch)
    dp_meta_met = dp_meta_met.map(ut.readfile)
    dp_meta_met = dp_meta_met.map(ut.extend_date)
    return dp_acc, dp_meta_met, dp_meta_struct, dp_corrupted, welch

def build_pipeline_meteo():
    dp = it.FileLister(root=settings.path.raw, recursive=True, masks='*.mat')
    dp = dp.map(ut.process_file_path)
    _, dp_meta_met, _, _ = dp.demux(num_instances=4, \
        classifier_fn= ut.demux_data,buffer_size=70_000)
    dp_meta_met = dp_meta_met.map(ut.readfile)
    meteo_avg = ut.MeteoAverage(win_size='10T')
    dp_meta_met = dp_meta_met.map(meteo_avg)
    reshape = ut.ReshapeToTuples()
    dp_meta_met = dp_meta_met.map(reshape)
    return dp_meta_met


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dp_acc, dp_meta_met, dp_meta_struct, dp_corrupted,_ = build_pipeline()
    df_meta_meteo = build_pipeline_meteo()
    for data in df_meta_meteo:
        print(data['file_data']['data'].shape)
        print(data)
        break

    for data in dp_acc:
        break
        print(data['data'].keys())
        data_dict = data['data'] 
        mask = [sensor == 'accel' for sensor in data_dict['sensors']]
        time_series = data_dict['log_psd'][mask]         
        fig,ax = plt.subplots(5,2,figsize=(10,5))       
        for i in range(10):
            ax[i//2,i%2].plot(time_series[i])
            ax[i//2,i%2].set_title(data['file_paths'][i])
        fig.savefig('psd.png')
        break