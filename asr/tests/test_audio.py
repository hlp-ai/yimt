import os.path

import librosa
import numpy as np

from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram


def test_audio():
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    audio = load_audio(audio_path)
    assert audio.ndim == 1
    assert SAMPLE_RATE * 10 < audio.shape[0] < SAMPLE_RATE * 12
    assert 0 < audio.std() < 1

    mel_from_audio = log_mel_spectrogram(audio)
    mel_from_file = log_mel_spectrogram(audio_path)

    assert np.allclose(mel_from_audio, mel_from_file)
    assert mel_from_audio.max() - mel_from_audio.min() <= 2.0

    audio2, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    assert SAMPLE_RATE * 10 < audio.shape[0] < SAMPLE_RATE * 12
    assert sr==SAMPLE_RATE
    assert audio.ndim == 1
    assert 0 < audio.std() < 1

    mel_from_audio2 = log_mel_spectrogram(audio)
    assert np.allclose(mel_from_audio2, mel_from_audio)

