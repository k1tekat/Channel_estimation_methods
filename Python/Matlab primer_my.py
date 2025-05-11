import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

# Параметры системы
Nfft = 32  # Размер FFT
Ng = Nfft // 8  # Длина циклического префикса
Nofdm = Nfft + Ng  # Общая длина OFDM символа
Nsym = 1  # Количество OFDM символов 100 было
Nps = 4  # Интервал между пилотами
Np = Nfft // Nps  # Количество пилотов на OFDM символ
Nbps = 4  # Количество бит на символ
M = 2 ** Nbps  # Размер QAM модуляции
Es = 1  # Энергия сигнала
A = np.sqrt(3 / 2 / (M - 1) * Es)  # Нормализующий коэффициент для QAM
SNR = 30  # Отношение сигнал/шум в дБ
MSE = np.zeros(6)  # Массив для хранения MSE
nose = 0  # Счетчик ошибок

print("Длина циклического префикса Ng =",Ng,"\n")
print("Общая длина OFDM символа Nofdm =",Nofdm,"\n")
print("Количество пилотов на OFDM символ Np =",Np,"\n")
print("Размер QAM модуляции M =",M,"\n")
print("Нормализующий коэффициент для QAM A =",A,"\n")



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

def Mapper(b,Nbps):
    mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}
     # Проверка: количество бит кратно Nbps
    if len(b) % Nbps != 0:
        raise ValueError("Длина входного списка бит должна быть кратна Nbps.")

    num_symbols = len(b) // Nbps
    n = 0
    N = np.zeros(num_symbols, dtype=complex)
    plt.figure("QAM-16",figsize=(10,10))
    for i in range(int(len(bits)/Nbps)):
        start = i * Nbps
        end = start + Nbps
        B = tuple(b[start:end])  # Получаем группу битов как кортеж
        Q = mapping_table[B]
        N[i] = Q
        print("N: ",N)
        print("Q: ",Q)
        print("Кол-во битов:",len(bits))
        print("Кол-во комлексных чисел на выходе:",int(len(bits)/Nbps))

        # Отрисовка точки
        plt.plot(Q.real, Q.imag, 'o', markersize=10, markerfacecolor='blue', markeredgecolor='black')
        # Добавление текста с битами
        plt.text(Q.real, Q.imag + 0.3, ''.join(str(bit) for bit in B), ha='center', fontsize=9, color='darkred')
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Constellation Diagram of 16-QAM')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
        
    plt.show()
    return N


# Генерация случайного канала
h = np.random.randn(2) + 1j * np.random.randn(2)
H_true = np.fft.fft(h, Nfft)  # Истинный канал в частотной области
ch_length = len(h)



# Основной цикл
for nsym in range(Nsym):
    # Генерация пилотных сигналов
    Xp = 2 * (np.random.randn(Np) > 0) - 1
    msgint = np.random.randint(0, M, Nfft - Np)  

    # Генерация данных
    bits = np.random.binomial(n=1, p=0.5, size = ((Nfft - Np) * 4) )# случайная генерация битовой последовательности
    Data = A * (2 * (msgint // np.sqrt(M)) - 1 + 1j * (2 * (msgint % np.sqrt(M)) - 1))
    print(Data)

    
    My_Data = Mapper(bits,Nbps)# Передаем битовую последовательность (96 бит) и размер бит на символ (Nbps)
    print(My_Data)

    """
    1. сгенерировать последовательность бит
    2. сделать маппер  и разбить биты, сконвертировать их в Комплексные сисла
    3. добавить опорные сигналы (1+0J) например
    """

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