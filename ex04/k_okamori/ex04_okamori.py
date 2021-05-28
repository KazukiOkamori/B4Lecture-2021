import numpy as np
import soundfile as sf  # 音声データの読み取り
import matplotlib.pyplot as plt  # 出力データのプロット
from scipy.linalg import solve_toeplitz

# 定数の定義
NFFT = 1024  # frame の大きさ
OVERLAP = int(NFFT // 2)  # half shift を採用
window = np.hamming(NFFT)  # 窓関数はハミング窓を使用


class SpectralEnvelope:
    def __init__(self, fname):
        self.wave, self.samplerate = sf.read(fname)

    def stft(self):
        self.spec = np.zeros([int(NFFT / 2) + 1, len(self.wave) // OVERLAP - 1],
                             dtype=np.complex)  # self.spec:出力データ

        for idx in range(0, len(self.wave), OVERLAP):
            frame = self.wave[idx: idx + NFFT]  # frame の切り出し
            if len(frame) == NFFT:  # frame が全部切り出せるときのみ変換
                windowed = window * frame  # 窓関数をかける
                result = np.fft.rfft(windowed)  # フーリエ変換
                for i in range(self.spec.shape[0]):
                    # 計算結果を出力データに追加
                    self.spec[i][int(idx / OVERLAP)] = result[i]

    def create_auto_correlation(self):
        self.ac = np.fft.irfft(np.square(np.abs(self.spec)), axis=0)

    def f0_by_auto_correlation(self, threshold):
        self.f0_ac = np.zeros(self.ac.shape[1])

        for i in range(self.ac.shape[1]):
            for j in range(1, self.ac.shape[0]-1):
                if self.ac[j, i] > self.ac[j-1, i] + threshold and self.ac[j, i] > self.ac[j+1, i] + threshold:
                    self.f0_ac[i] = self.ac[j, i]
                    print(self.f0_ac[i])
                    break

        self.f0_ac = self.samplerate / self.f0_ac

        pass

    def create_cepstrum(self):
        self.spec_dB = 20 * np.log10(abs(self.spec))
        self.cepstrum = np.fft.irfft(self.spec_dB, axis=0)

    def spectral_envelope_by_cepstrum(self, threshold):
        lifter = np.zeros_like(self.cepstrum)
        lifter[: threshold] = 1
        self.cep_env = np.fft.rfft(self.cepstrum * lifter, axis=0).real
        self.cep_micro = np.fft.rfft(self.cepstrum * (1 - lifter), axis=0).real

    def f0_by_cepstrum(self, threshold):
        self.f0_cep = np.zeros(self.cepstrum.shape[1])

        for i in range(self.cepstrum.shape[1]):
            for j in range(1, self.cepstrum.shape[0]-1):
                if self.cepstrum[j, i] > self.cepstrum[j-1, i] + threshold and self.cepstrum[j, i] > self.cepstrum[j+1, i] + threshold:
                    self.f0_cep[i] = self.cepstrum[j, i]
                    print(self.f0_cep[i])
                    break

    def spectral_envelope_by_lpc(self, deg):
        r = self.ac[:deg]

        a = np.zeros_like(r)
        a[0] = 1
        for i in range(r.shape[1]):
            a[1:, i] = solve_toeplitz(r[:-1, i], -r[1:, i])
            e = np.sqrt(np.sum(a * r, axis=0))
            self.lpc_env = 20 * np.log10(abs(e / np.fft.rfft(a, NFFT, axis=0)))


def main():
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    wave = SpectralEnvelope("aiueo.wav")
    wave.stft()
    wave.create_cepstrum()
    wave.spectral_envelope_by_cepstrum(100)
    wave.create_auto_correlation()
    wave.f0_by_auto_correlation(0.1)
    wave.f0_by_cepstrum(0.1)
    wave.spectral_envelope_by_lpc(100)
    ax1.plot(wave.f0_ac)
    ax1.plot(wave.f0_cep)
    ax2.plot(wave.spec_dB[:, 50])
    ax2.plot(wave.cep_env[:, 50])
    ax2.plot(wave.lpc_env[:, 50])
    plt.savefig("result.png")


if __name__ == "__main__":
    main()
