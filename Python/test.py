import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.linalg import inv, toeplitz
from scipy.special import jv

# Параметры системы
Nfft = 64  # Размер FFT
Ng = Nfft // 8  # Длина циклического префикса
Nofdm = Nfft + Ng  # Общая длина OFDM символа
Nsym = 1  # Количество OFDM символов
Nps = 8  # Интервал между пилотами
Np = Nfft // Nps  # Количество пилотов на OFDM символ
Nbps = 4  # Количество бит на символ
M = 2 ** Nbps  # Размер QAM модуляции
Es = 1  # Энергия сигнала
A = np.sqrt(3 / 2 / (M - 1) * Es)  # Нормализующий коэффициент для QAM
SNR = 30  # Отношение сигнал/шум в дБ

print(f"Длина циклического префикса Ng = {Ng}")
print(f"Общая длина OFDM символа Nofdm = {Nofdm}")
print(f"Количество пилотов на OFDM символ Np = {Np}")
print(f"Размер QAM модуляции M = {M}")
print(f"Нормализующий коэффициент для QAM A = {A}")

# Функция для добавления шума
def awgn(signal, SNR):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (SNR / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# Оценка канала методом LS
def LS_CE(Y, Xp, pilot_loc, Nfft, Nps, int_opt='linear'):
    Np = len(pilot_loc)
    H_ls_pilots = Y[pilot_loc] / Xp
    return interpolate(H_ls_pilots, pilot_loc, Nfft, method=int_opt)

def compute_R_HH(N, df=1e6, tau_max=5e-6, f_max=100):
    l = np.arange(N)
    r_t = jv(0, 2 * np.pi * f_max * l / df)  # Jakes' модель
    r_f = np.zeros(N, dtype=complex)
    for k in range(N):
        r_f[k] = np.sum(r_t * np.exp(-1j * 2 * np.pi * k * np.arange(N) / N))
    return toeplitz(r_f)


def MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, SNR_dB):
    """
    MMSE оценка канала на основе пилотов
    """
    pilot_loc = np.array(pilot_loc)

    # LS-оценка на пилотах
    H_ls_pilots = Y[pilot_loc] / Xp[:len(pilot_loc)]

    # Мощность шума
    sigma_n_sq = 10 ** (-SNR_dB / 10)

    # Автокорреляционная матрица истинного канала
    R_HH = compute_R_HH(Nfft)

    # Ковариационная матрица наблюдаемого сигнала на пилотах
    R_yy_pilots = np.eye(len(H_ls_pilots)) * (np.mean(np.abs(H_ls_pilots)**2) + sigma_n_sq)

    # Вычисление весовой матрицы
    try:
        W = R_HH[:, pilot_loc] @ inv(R_yy_pilots)
    except np.linalg.LinAlgError:
        W = R_HH[:, pilot_loc] @ np.linalg.pinv(R_yy_pilots)

    # MMSE-оценка на пилотах
    H_mmse_pilots = W @ H_ls_pilots

    # Проверка: длина H_mmse_pilots == длина pilot_loc
    if len(H_mmse_pilots) != len(pilot_loc):
        print("⚠️ Длина H_mmse_pilots и pilot_loc не совпадает!")
        min_len = min(len(pilot_loc), len(H_mmse_pilots))
        pilot_loc = pilot_loc[:min_len]
        H_mmse_pilots = H_mmse_pilots[:min_len]

    # Интерполяция
    return interpolate(H_mmse_pilots, pilot_loc, Nfft, 'linear')
# Интерполяция
def interpolate(H, pilot_loc, Nfft, method='linear'):
    pilot_loc = np.array(pilot_loc)
    real_part = np.real(H)
    imag_part = np.imag(H)
    if method == 'linear':
        kind = 'linear'
    elif method in ['spline', 'cubic']:
        kind = 'cubic'
    else:
        raise ValueError("Неподдерживаемый метод интерполяции")
    freq_indices = np.arange(Nfft)
    interp_real = interp1d(pilot_loc, real_part, kind=kind, fill_value="extrapolate")
    interp_imag = interp1d(pilot_loc, imag_part, kind=kind, fill_value="extrapolate")
    H_real = interp_real(freq_indices)
    H_imag = interp_imag(freq_indices)
    return H_real + 1j * H_imag

# Модулятор QAM-16
def Mapper(bits, Nbps):
    mapping_table = {
        (0, 0, 0, 0): -3 - 3j,
        (0, 0, 0, 1): -3 - 1j,
        (0, 0, 1, 0): -3 + 3j,
        (0, 0, 1, 1): -3 + 1j,
        (0, 1, 0, 0): -1 - 3j,
        (0, 1, 0, 1): -1 - 1j,
        (0, 1, 1, 0): -1 + 3j,
        (0, 1, 1, 1): -1 + 1j,
        (1, 0, 0, 0): 3 - 3j,
        (1, 0, 0, 1): 3 - 1j,
        (1, 0, 1, 0): 3 + 3j,
        (1, 0, 1, 1): 3 + 1j,
        (1, 1, 0, 0): 1 - 3j,
        (1, 1, 0, 1): 1 - 1j,
        (1, 1, 1, 0): 1 + 3j,
        (1, 1, 1, 1): 1 + 1j,
    }
    num_symbols = len(bits) // Nbps
    symbols = []
    for i in range(num_symbols):
        start = i * Nbps
        end = start + Nbps
        B = tuple(bits[start:end])
        symbols.append(mapping_table[B])
    return np.array(symbols)

# Демодулятор QAM-16
def demapper(received_symbols):
    mapping_table = {
        (0, 0, 0, 0): -3 - 3j,
        (0, 0, 0, 1): -3 - 1j,
        (0, 0, 1, 0): -3 + 3j,
        (0, 0, 1, 1): -3 + 1j,
        (0, 1, 0, 0): -1 - 3j,
        (0, 1, 0, 1): -1 - 1j,
        (0, 1, 1, 0): -1 + 3j,
        (0, 1, 1, 1): -1 + 1j,
        (1, 0, 0, 0): 3 - 3j,
        (1, 0, 0, 1): 3 - 1j,
        (1, 0, 1, 0): 3 + 3j,
        (1, 0, 1, 1): 3 + 1j,
        (1, 1, 0, 0): 1 - 3j,
        (1, 1, 0, 1): 1 - 1j,
        (1, 1, 1, 0): 1 + 3j,
        (1, 1, 1, 1): 1 + 1j,
    }
    constellation = np.array(list(mapping_table.values()))
    demapped_bits = []
    for symbol in received_symbols:
        distances = np.abs(symbol - constellation)
        nearest_idx = np.argmin(distances)
        bits = list(mapping_table.keys())[nearest_idx]
        demapped_bits.extend(bits)
    return np.array(demapped_bits)

# Расчёт BER
def calculate_BER(tx_bits, rx_bits):
    return np.sum(tx_bits != rx_bits) / len(tx_bits)

# Генерация случайного канала
h = np.random.randn(2) + 1j * np.random.randn(2)
H_true = np.fft.fft(h, Nfft)
ch_length = len(h)

# === Основной цикл ===
for nsym in range(Nsym):
    # === Генерация QAM-16 пилотных символов ===
    bits_pilot = np.random.randint(0, 2, size=Np * Nbps)
    Xp = Mapper(bits_pilot, Nbps)

    # Генерация данных
    bits = np.random.binomial(n=1, p=0.5, size=(Nfft - Np) * Nbps)
    My_Data = Mapper(bits, Nbps)

    # Формирование OFDM символа
    ip = 0
    pilot_loc = []
    X = np.zeros(Nfft, dtype=complex)
    buf = 0
    for k in range(Nfft):
        if k % Nps == 0 and ip < len(Xp):
            X[k] = Xp[ip]
            pilot_loc.append(k)
            ip += 1
        else:
            X[k] = My_Data[k - ip] if (k - ip) < len(My_Data) else 0

    # IFFT и добавление CP
    x = np.fft.ifft(X)
    xt = np.concatenate((x[-Ng:], x))

    # Прохождение через канал
    y_channel = fftconvolve(xt, h, mode='full')[:Nofdm]
    yt = awgn(y_channel, SNR)
    y = yt[Ng:]
    Y = np.fft.fft(y)

    # Оценка канала
    H_est_ls = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, 'linear')
    H_est_mmse = MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, SNR)

    # Компенсация канала
    Y_eq_ls = Y / H_est_ls
    Y_eq_mmse = Y / H_est_mmse

    # Извлечение данных (без пилотов)
    Data_extracted_ls = []
    Data_extracted_mmse = []

    for k in range(Nfft):
        if k not in pilot_loc:
            Data_extracted_ls.append(Y_eq_ls[k])
            Data_extracted_mmse.append(Y_eq_mmse[k])

    # Демодуляция
    Data_extracted_ls = demapper(Data_extracted_ls)
    Data_extracted_mmse = demapper(Data_extracted_mmse)

    # Убедись, что длины совпадают перед calculate_BER
    expected_data_len = len(bits)
    actual_data_len = min(len(Data_extracted_ls), len(Data_extracted_mmse), expected_data_len)

    bits_trunc = bits[:actual_data_len]
    rx_ls_trunc = Data_extracted_ls[:actual_data_len]
    rx_mmse_trunc = Data_extracted_mmse[:actual_data_len]

    # Расчёт BER
    ber_ls = calculate_BER(bits_trunc, rx_ls_trunc)
    ber_mmse = calculate_BER(bits_trunc, rx_mmse_trunc)

    print(f"BER (LS): {ber_ls:.4f}")
    print(f"BER (MMSE): {ber_mmse:.4f}")

    # Constellation
    plt.figure("Constellation", figsize=(10, 8))
    plt.plot(Data_extracted_ls.real, Data_extracted_ls.imag, 'bo', markersize=6, label='LS')
    plt.plot(Data_extracted_mmse.real, Data_extracted_mmse.imag, 'go', markersize=4, label='MMSE')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.title("Constellation Diagram после компенсации канала (LS vs MMSE)")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()