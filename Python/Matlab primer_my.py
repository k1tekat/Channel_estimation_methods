import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

# Параметры системы
Nfft = 32  # Размер FFT
Ng = Nfft // 8  # Длина циклического префикса
Nofdm = Nfft + Ng  # Общая длина OFDM символа
Nsym = 100  # Количество OFDM символов
Nps = 4  # Интервал между пилотами
Np = Nfft // Nps  # Количество пилотов на OFDM символ
Nbps = 4  # Количество бит на символ
M = 2 ** Nbps  # Размер QAM модуляции
Es = 1  # Энергия сигнала
A = np.sqrt(3 / 2 / (M - 1) * Es)  # Нормализующий коэффициент для QAM
SNR = 30  # Отношение сигнал/шум в дБ
MSE = np.zeros(6)  # Массив для хранения MSE
nose = 0  # Счетчик ошибок

# Функция для добавления шума
def awgn(signal, SNR):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (SNR / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# LS оценка канала
def LS_CE(Y, Xp, pilot_loc, Nfft, Nps, int_opt):
    Np = Nfft // Nps
    k = np.arange(Np)
    LS_est = Y[pilot_loc[:Np]] / Xp[:Np]  # Оценка канала в местах пилотов
    if int_opt.lower().startswith('l'):
        method = 'linear'
    else:
        method = 'spline'
    H_LS = interpolate(LS_est, pilot_loc[:Np], Nfft, method)
    return H_LS

# Интерполяция
def interpolate(LS_est, pilot_loc, Nfft, method):
    if method == 'linear':
        interp_func = interp1d(pilot_loc, LS_est, kind='linear', fill_value="extrapolate")
    elif method == 'spline':
        interp_func = interp1d(pilot_loc, LS_est, kind='cubic', fill_value="extrapolate")
    freq_indices = np.arange(Nfft)
    return interp_func(freq_indices)

# MMSE оценка канала
def MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, h, SNR):
    H_LS = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, 'linear')
    sigma_n_sq = 10 ** (-SNR / 10)
    H_MMSE = H_LS / (1 + sigma_n_sq / np.abs(H_LS) ** 2)
    return H_MMSE

# Генерация случайного канала
h = np.random.randn(2) + 1j * np.random.randn(2)
H_true = np.fft.fft(h, Nfft)  # Истинный канал в частотной области
ch_length = len(h)

# Основной цикл
for nsym in range(Nsym):
    # Генерация пилотных сигналов
    Xp = 2 * (np.random.randn(Np) > 0) - 1
    msgint = np.random.randint(0, M, Nfft - Np)  # Генерация данных
    Data = A * (2 * (msgint // np.sqrt(M)) - 1 + 1j * (2 * (msgint % np.sqrt(M)) - 1))

    # Формирование OFDM символа
    ip = 0
    pilot_loc = []
    X = np.zeros(Nfft, dtype=complex)
    for k in range(Nfft):
        if k % Nps == 0:
            X[k] = Xp[ip]
            pilot_loc.append(k)
            ip += 1
        else:
            X[k] = Data[k - ip]

    # IFFT и добавление циклического префикса
    x = np.fft.ifft(X)
    xt = np.concatenate((x[-Ng:], x))

    # Прохождение через канал
    y_channel = fftconvolve(xt, h, mode='full')[:Nofdm]
    yt = awgn(y_channel, SNR)
    y = yt[Ng:]
    Y = np.fft.fft(y)

    # Оценка канала
    methods = ['ls linear', 'ls spline', 'MMSE']
    for m, method in enumerate(methods):
        if method == 'ls linear' or method == 'ls spline':
            H_est = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, method)
        elif method == 'MMSE':
            H_est = MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, h, SNR)

        # DFT-based оценка канала
        h_est = np.fft.ifft(H_est)
        h_DFT = h_est[:ch_length]
        H_DFT = np.fft.fft(h_DFT, Nfft)

        # Вычисление MSE
        MSE[m] += np.sum(np.abs(H_true - H_est) ** 2)
        MSE[m + 3] += np.sum(np.abs(H_true - H_DFT) ** 2)

        # Визуализация (только для первого символа)
        if nsym == 0:
            H_true_dB = 10 * np.log10(np.abs(H_true) ** 2)
            H_est_dB = 10 * np.log10(np.abs(H_est) ** 2)
            H_DFT_dB = 10 * np.log10(np.abs(H_DFT) ** 2)

            plt.subplot(3, 2, 2 * m + 1)
            plt.plot(H_true_dB, 'b', label='True Channel')
            plt.plot(H_est_dB, 'r:', label=f'{method}')
            plt.legend()

            plt.subplot(3, 2, 2 * m + 2)
            plt.plot(H_true_dB, 'b', label='True Channel')
            plt.plot(H_DFT_dB, 'r:', label=f'{method} with DFT')
            plt.legend()

    # Демодуляция
    Y_eq = Y / H_est
    ip = 0
    Data_extracted = []
    for k in range(Nfft):
        if k % Nps == 0:
            ip += 1
        else:
            Data_extracted.append(Y_eq[k])

    Data_extracted = np.array(Data_extracted)
    msg_detected = np.round(((np.real(Data_extracted / A) + 1) / 2) * np.sqrt(M)) + \
                   np.round(((np.imag(Data_extracted / A) + 1) / 2) * np.sqrt(M)) * np.sqrt(M)
    nose += np.sum(msg_detected != msgint)

# Вывод результатов
MSEs = MSE / (Nfft * Nsym)
print("MSE:", MSEs)
print("Bit Error Rate:", nose / (Nsym * (Nfft - Np)))
plt.tight_layout()
plt.show()