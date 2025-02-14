import numpy as np
import torch
from torch import nn 
from torchaudio.transforms import Spectrogram

class Welch(nn.Module):
    """
    A merged Welch transform class that:
      - uses a Hann window,
      - can compute average (mean or std) across frames,
      - allows customizing n_fft, sample rate fs, and scaling,
      - detrends the input before computing the spectrogram.
    """
    def __init__(self, n_fft: int = 4096, fs: float = 400.0, average: str = 'mean'):
        super().__init__()
        self.n_fft = n_fft
        self.fs = fs
        self.average = average

        # 1) Create a Hann window of length n_fft
        self.window = torch.hann_window(self.n_fft)

        # 2) Define 50% overlap
        self.hop_length = self.n_fft // 2

        # 3) Build the Spectrogram transform
        #    - Use power=2 => magnitude squared
        #    - center=False => no padding
        #    - window_fn => returns our Hann window
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            power=2.0,
            normalized=False,
            window_fn=lambda _: self.window
        )

        # 4) Prepare a frequency axis from 0 .. fs/2  (since onesided STFT is default)
        self.freq_axis = torch.linspace(0, self.fs / 2, self.n_fft // 2 + 1)

        # 5) Set up the averaging function
        #    If you add more keys (e.g., 'median'), update this dict
        self.average_dict = {'mean': torch.mean, 'std': torch.std}
        if self.average not in self.average_dict:
            raise ValueError(f"average must be one of {list(self.average_dict.keys())}")
        self.average_fn = self.average_dict[self.average]

        # 6) Optionally define a scaling factor
        #    This is just an example; adapt it if you want a different PSD normalization.
        self.scaling = 1.0 / (self.fs**2)

    def get_frequency_axis(self):
        """
        Returns the frequency axis (one-sided) from 0..fs/2
        """
        return self.freq_axis

    def forward(self, data):
        """
        Args:
            x (Tensor): shape (batch, time) or (batch, channel, time).
                        If (time,) we'll expand dims to (1, time).
        Returns:
            Tensor: shape (batch, freq) or (batch, channel, freq)
                    after averaging across STFT frames.
        """
        # Ensure shape: if 1D, expand to (1, time)
        x = data['data'].pop('time_series')
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Detrend by subtracting the mean along -1 (time dimension)
        x = x - torch.mean(x, dim=-1, keepdim=True)

        # Compute power spectrogram => shape [batch, freq, frames] (for 2D input)
        spec = self.spectrogram(x)

        # Average across frames (dim=-1) => shape [batch, freq]
        # or [batch, channel, freq] if x had shape (batch, cha
                # Average across frames (dim=-1) => shape [batch, freq]
        # or [batch, channel, freq] if x had shape (batch, channel, time).
        spec_avg = self.average_fn(spec, dim=-1)
        res = spec_avg*self.scaling
        data['data'].update({'psd': res})

        return data

    def __str__(self):
        return (f"Welch(n_fft={self.n_fft}, fs={self.fs}, average={self.average}, "
                f"scaling={self.scaling}, hop_length={self.hop_length})")
    
