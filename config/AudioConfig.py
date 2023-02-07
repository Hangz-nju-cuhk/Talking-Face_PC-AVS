import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import lws


class AudioConfig:
    def __init__(self, frame_rate=25,
                       sample_rate=16000,
                       num_mels=80,
                       fft_size=1280,
                       hop_size=160,
                       num_frames_per_clip=5,
                       save_mel=True
                ):
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.num_bins_per_frame = int(sample_rate / hop_size / frame_rate)
        self.num_frames_per_clip = num_frames_per_clip
        self.silence_threshold = 2
        self.num_mels = num_mels
        self.save_mel = save_mel
        self.fmin = 125
        self.fmax = 7600
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.frame_shift_ms = None
        self.min_level_db = -100
        self.ref_level_db = 20
        self.rescaling = True
        self.rescaling_max = 0.999
        self.allow_clipping_in_normalization = True
        self.log_scale_min = -32.23619130191664
        self.norm_audio = True
        self.with_phase = False

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.sample_rate)[0]

    def audio_normalize(self, samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
        samples = samples * (desired_rms / rms)
        return samples

    def generate_spectrogram_magphase(self, audio):
        spectro = librosa.core.stft(audio, hop_length=self.get_hop_size(), n_fft=self.fft_size, center=True)
        spectro_mag, spectro_phase = librosa.core.magphase(spectro)
        spectro_mag = np.expand_dims(spectro_mag, axis=0)
        if self.with_phase:
            spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
            return spectro_mag, spectro_phase
        else:
            return spectro_mag

    def save_wav(self, wav, path):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, self.sample_rate, wav.astype(np.int16))

    def trim(self, quantized):
        start, end = self.start_and_end_indices(quantized, self.silence_threshold)
        return quantized[start:end]

    def adjust_time_resolution(self, quantized, mel):
        """Adjust time resolution by repeating features

        Args:
            quantized (ndarray): (T,)
            mel (ndarray): (N, D)

        Returns:
            tuple: Tuple of (T,) and (T, D)
        """
        assert len(quantized.shape) == 1
        assert len(mel.shape) == 2

        upsample_factor = quantized.size // mel.shape[0]
        mel = np.repeat(mel, upsample_factor, axis=0)
        n_pad = quantized.size - mel.shape[0]
        if n_pad != 0:
            assert n_pad > 0
            mel = np.pad(mel, [(0, n_pad), (0, 0)], mode="constant", constant_values=0)

        # trim
        start, end = self.start_and_end_indices(quantized, self.silence_threshold)

        return quantized[start:end], mel[start:end, :]

    adjast_time_resolution = adjust_time_resolution  # 'adjust' is correct spelling, this is for compatibility

    def start_and_end_indices(self, quantized, silence_threshold=2):
        for start in range(quantized.size):
            if abs(quantized[start] - 127) > silence_threshold:
                break
        for end in range(quantized.size - 1, 1, -1):
            if abs(quantized[end] - 127) > silence_threshold:
                break

        assert abs(quantized[start] - 127) > silence_threshold
        assert abs(quantized[end] - 127) > silence_threshold

        return start, end

    def melspectrogram(self, y):
        D = self._lws_processor().stft(y).T
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        if not self.allow_clipping_in_normalization:
            assert S.max() <= 0 and S.min() - self.min_level_db >= 0
        return self._normalize(S)

    def get_hop_size(self):
        hop_size = self.hop_size
        if hop_size is None:
            assert self.frame_shift_ms is not None
            hop_size = int(self.frame_shift_ms / 1000 * self.sample_rate)
        return hop_size

    def _lws_processor(self):
        return lws.lws(self.fft_size, self.get_hop_size(), mode="speech")

    def lws_num_frames(self, length, fsize, fshift):
        """Compute number of time frames of lws spectrogram
        """
        pad = (fsize - fshift)
        if length % fshift == 0:
            M = (length + pad * 2 - fsize) // fshift + 1
        else:
            M = (length + pad * 2 - fsize) // fshift + 2
        return M

    def lws_pad_lr(self, x, fsize, fshift):
        """Compute left and right padding lws internally uses
        """
        M = self.lws_num_frames(len(x), fsize, fshift)
        pad = (fsize - fshift)
        T = len(x) + 2 * pad
        r = (M - 1) * fshift + fsize - T
        return pad, pad + r


    def _linear_to_mel(self, spectrogram):
        global _mel_basis
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self):
        assert self.fmax <= self.sample_rate // 2
        return librosa.filters.mel(self.sample_rate, self.fft_size,
                                   fmin=self.fmin, fmax=self.fmax,
                                   n_mels=self.num_mels)

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db

    def read_audio(self, audio_path):
        wav = self.load_wav(audio_path)
        if self.norm_audio:
            wav = self.audio_normalize(wav)
        else:
            wav = wav / np.abs(wav).max()

        return wav

    def audio_to_spectrogram(self, wav):
        if self.save_mel:
            spectrogram = self.melspectrogram(wav).astype(np.float32).T
        else:
            spectrogram = self.generate_spectrogram_magphase(wav)
        return spectrogram
