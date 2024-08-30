from typing import Optional, Union, Dict

import librosa
import torch
import torch.nn.functional as F

from whisper import Whisper, ModelDimensions
from whisper.audio import N_FFT, HOP_LENGTH


class Mel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.register_buffer('mel_80', torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)))

    def forward(self,
            audio: torch.Tensor,
            n_mels: int = 80,
            padding: int = 0,
            device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Compute the log-Mel spectrogram of audio

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
            The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

        n_mels: int
            The number of Mel-frequency filters, only 80 is supported

        padding: int
            Number of zero samples to pad to the right

        device: Optional[Union[str, torch.device]]
            If given, the audio tensor is moved to this device before STFT

        Returns
        -------
        torch.Tensor, shape = (80, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        if device is not None:
            audio = audio.to(device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        window = torch.hann_window(N_FFT).to(audio.device)
        stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_80
        filters = filters.to(device)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


class WhisperMel(Whisper):

    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.audio2mel = Mel()
        self.mel2txt = Whisper(dims)

    def forward(self, audio: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        mel = self.audio2mel(audio)
        return self.mel2txt(mel, tokens)


if __name__ == "__main__":
    audio_path = "../tests/jfk.flac"
    audio2, sr = librosa.load(audio_path, sr=16000)

    mel = Mel()
    mel_from_audio3 = mel(torch.from_numpy(audio2))

    print(mel_from_audio3.shape)
