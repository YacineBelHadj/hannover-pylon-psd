import datetime
import h5py
import torch
from torchaudio.transforms import Spectrogram
from pathlib import Path
from scipy.io import loadmat

class Welch(torch.nn.Module):
    def __init__(self, n_fft=32784, fs=1651, max_freq=None):
        super(Welch, self).__init__()
        self.n_fft = n_fft
        self.fs = fs
        self.max_freq = max_freq
        self.freq_line = torch.linspace(0, fs/2, n_fft//2+1)
        self.max_freq = max_freq if max_freq else fs/2
        self.freq_mask = self.freq_line <= self.max_freq
        self.freq_line_masked = self.freq_line[self.freq_mask]
        self.spectrogram = Spectrogram(n_fft=n_fft)
        
    def forward(self, data):
        x = data['data'].pop('time_series')
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_detrend = x - x.mean(-1, keepdim=True)
        psd = self.spectrogram(x_detrend)
        psd = psd.mean(-1)
        psd = psd[..., self.freq_mask]
        psd = torch.log10(psd + 1e-6)
        data['data'].update({'log_psd': psd})
        return data
    
    def __str__(self) -> str:
        return f'Welch(n_fft={self.n_fft}, fs={self.fs}, max_freq={self.max_freq})'
        
        

def extract_references(dataset, file):
    """Extract referenced objects from the HDF5 dataset."""
    decoded_strings = []
    for ref in dataset:
        deref_obj = file[ref[0]]  # Dereference the object reference
        # Decode string if it's a byte array
        if isinstance(deref_obj, h5py.Dataset):
            decoded_strings.append(deref_obj[()].tobytes().decode('utf-16'))  # Decode as UTF-16 here
    return decoded_strings

# Clean up channel names by decoding UTF-16 and removing null characters
def clean_channel_name(name):
    return name.replace('\x00', '')  # Remove null characters

def get_date(file_path : Path| str):
    """Extract the date from the file path."""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    file_name_splited = file_path.stem.split('_')
    if len(file_name_splited) == 1 :
        year = file_path.parent.parent
        month = file_path.parent
        return datetime.datetime(int(year.stem), int(month.stem),1)
    
    file_date = file_name_splited[1]

    if len(file_date) == 12:
        return datetime.datetime.strptime(file_date, '%Y%m%d%H%M')
    elif len(file_date) == 6:
        return datetime.datetime.strptime(file_date, '%Y%m')
    else :
        raise ValueError(f'Unexpected file name format: {file_date}')
    

def read_mat(file_path):
    """Read the .mat file and extract relevant information."""
    with h5py.File(file_path, 'r') as f:
        # Extract the channel names
        dat = f['Dat']
        channel_names = extract_references(dat['ChannelNames'], f)
        channel_names = [clean_channel_name(name) for name in channel_names]
        
        data  = dat['Data'][()]
        fs = dat['Fs'][0][0]
        channel_units = extract_references(dat['ChannelUnits'], f)
        channel_units = [clean_channel_name(unit) for unit in channel_units]
        
    return {'data': data, 'fs': float(fs), 'channel_names': channel_names, 'channel_units': channel_units}    
        
def identify_file(path:Path| str):
    if isinstance(path, str):
        path = Path(path)
    if 'meta_met' in path.stem:
        return 'meta_met'
    elif 'meta_struct' in path.stem:
        return 'meta_struct'
    elif 'listOfCorruptedFiles' in path.stem:
        return 'corrupted_files'
    elif len(path.stem.split('_')[1]) == 12:
        return 'acceleration'
    else:   
        raise ValueError(f'Unknown file type for {path.stem}')
    
import re
def extract_channel_info(data):
    channel_names = data['file_data'].get('channel_names', [])
    levels = []
    directions = []
    sensor_names = []
    pattern = re.compile(r'(?P<sensor>[A-Za-z]+)(?P<level>\d+)(?P<direction>[xy]?)')
    for name in channel_names:
        match = pattern.match(name)
        if match:
            level_str = match.group('level')
            level = int(level_str.lstrip('0')) if level_str.lstrip('0') else 0
            direction = match.group('direction') if match.group('direction') else None
            sensor = match.group('sensor')
        else:
            level = None
            direction = None
        
        levels.append(level)
        directions.append(direction)
        sensor_names.append(sensor)
    
    data['file_data'].update({'level': levels, 'direction': directions, 'sensor': sensor_names})
    return data

def extend_date(data):
    date = data['date']
    number_of_samples= data['file_data']['data'].shape[-1]
    sample_rate  = datetime.timedelta(minutes=1)
    dates = [date + i*sample_rate for i in range(number_of_samples)]
    data['file_data'].update({'dates': dates})
    return data

def process_file_path(file_path: Path| str):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    date = get_date(file_path)
    file_nat = identify_file(file_path)
    return {'file_path': file_path, 'file_nature': file_nat, 'date': date}

def readfile(data):
    if data['file_nature'] == 'corrupted_files':
        data.update({'file_data': loadmat(data['file_path'])['listOfCorruptedFiles']}) 
    if data['file_nature'] == 'acceleration':
        data.update({'file_data': read_mat(data['file_path'])})
    if data['file_nature'] == 'meta_met':
        data.update({'file_data': read_mat(data['file_path'])})
    return data

class TrackCorruptedFiles:
    def __init__(self):
        self.corrupted_files = []
        
    def setup(self):
        self.corrupted_files = [str(f[0][0]) for f in self.corrupted_files]
    def __call__(self, data):
        if data['file_nature'] == 'corrupted_files':
            self.corrupted_files.extend(data['file_data'])
            
        if data['file_nature'] != 'corrupted_files':
            if data['file_path'].stem in self.corrupted_files:
                data['file_corrupted'] = True
            else:
                data['file_corrupted'] = False
        return data
    
import pandas as pd
    
class MeteoAverage:
    def __init__(self, win_size='10min'):
        """
        Initialize the MeteoAverage class with the desired window size.

        Parameters:
        - win_size (str): Window size as a pandas offset alias (e.g., '10T' for 10 minutes, '1H' for 1 hour).
        """
        self.win_size = win_size

    def __call__(self, data):
        """
        Perform averaging over the specified window size and include the start date of each window.

        Parameters:
        - data (dict): Dictionary containing the meteorological data and metadata.

        Returns:
        - data (dict): Updated dictionary with averaged data and window start dates.
        """
        fs = data['file_data']['fs']  # Sampling frequency in Hz 0.016  60 sample per second 
        start_date = data['date']     # Start date of the data
        data_array = data['file_data']['data']  # Numpy array of shape (channels, samples)
        channel_names = data['file_data']['channel_names']

        # Calculate the time delta between samples
        delta_seconds = 1 / fs
        total_samples = data_array.shape[1]

        # Create a date range for the data samples
        timestamps = pd.date_range(
            start=start_date,
            periods=total_samples,
            freq=pd.to_timedelta(delta_seconds, unit='s')
        )

        # Create a pandas DataFrame with the data and timestamps
        df = pd.DataFrame(
            data=data_array.T,  # Transpose to have samples as rows
            index=timestamps,
            columns=channel_names
        )
        df.dropna(inplace=True,how='all')
        # Resample the data using the specified window size and compute the mean
        df_resampled = df.resample(self.win_size).mean()

        # Extract the averaged data and the window start dates
        averaged_data = df_resampled.values.T  # Transpose back to shape (channels, windows)
        window_start_dates = df_resampled.index.to_pydatetime().tolist()

        # Update the data dictionary with the averaged data and window start dates
        data['averaged_data'] = averaged_data  # Shape: (channels, num_windows)
        data['window_start_dates'] = window_start_dates
        return data
    
class ReshapeToTuples:
    def __init__(self):
        """
        Initialize the ReshapeToTuples class.
        """
        pass

    def __call__(self, data):
        """
        Reshape the averaged meteorological data into tuples of (date, value, channelName).

        Parameters:
        - data (dict): Dictionary containing the averaged data and metadata.

        Returns:
        - data (dict): Updated dictionary with data reshaped into tuples.
        """
        averaged_data = data['averaged_data']          # Shape: (channels, windows)
        window_start_dates = data['window_start_dates']  # List of datetime objects
        channel_names = data['file_data']['channel_names']  # List of channel names

        # Transpose averaged_data to have shape (windows, channels)
        averaged_data_transposed = averaged_data.T  # Shape: (windows, channels)

        # Create the DataFrame
        df = pd.DataFrame(
            data=averaged_data_transposed,
            index=window_start_dates,
            columns=channel_names
        )
        df = df.reset_index().rename(columns={'index': 'date'})

        # Melt the DataFrame to long format
        df_long = df.melt(id_vars='date', var_name='channelName', value_name='value')
        df_long = df_long[['date', 'value', 'channelName']]

        # Convert to list of tuples
        data_tuples = [tuple(x) for x in df_long.to_numpy()]

        # Update the data dictionary
        data['data_tuples'] = data_tuples

        return data

def demux_data(data):
    expected_file_nature = {'acceleration':0
                            ,'meta_met':1
                            ,'meta_struct':2
                            ,'corrupted_files':3}
    return expected_file_nature[data['file_nature']]


