import adi
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

# === Настройки SDR ===
sdr_ip = "ip:192.168.2.1"  # IP PlutoSDR по умолчанию
sample_rate = int(2.5e6)
center_freq = int(2.4e9)  # Рабочая частота, например 2.4 ГГц

# Инициализация SDR
sdr = adi.Pluto(sdr_ip)
sdr.sample_rate = sample_rate
sdr.tx_lo = center_freq
sdr.rx_lo = center_freq
sdr.tx_hardwaregain_control = 0  # Усиление TX: -89.75 до 0 дБ
sdr.gain_control_mode_chan0 = "fast_attack"  # AGC или manual
sdr.rx_hardwaregain_chan0 = 30   # Приемное усиление (0–71 дБ)

# === Параметры OFDM из твоего кода ===
Nfft = 64
Ng = Nfft // 8
Nofdm = Nfft + Ng
Np = Nfft // 8
Nbps = 4
M = 2 ** Nbps
A = np.sqrt(3 / 2 / (M - 1))  # Нормализующий коэффициент QAM
SNR = 30

# Параметры системы
Nfft = 64  # Размер FFT
Ng = Nfft // 8  # Длина циклического префикса
Nofdm = Nfft + Ng  # Общая длина OFDM символа
Nsym = 1  # Количество OFDM символов 100 было
Nps = 8  # Интервал между пилотами
Np = Nfft // Nps  # Количество пилотов на OFDM символ
Nbps = 4  # Количество бит на символ
M = 2 ** Nbps  # Размер QAM модуляции
Es = 1  # Энергия сигнала
A = np.sqrt(3 / 2 / (M - 1) * Es)  # Нормализующий коэффициент для QAM
SNR = 30  # Отношение сигнал/шум в дБ
MSE = np.zeros(6)  # Массив для хранения MSE
nose = 0  # Счетчик ошибок





def LS_CE(Y, Xp, pilot_loc, Nfft, Nps, int_opt):
    """
    Оценка канала методом наименьших квадратов (Least Squares, LS)
    
    Параметры:
        Y (np.ndarray)              -- Принятый сигнал в частотной области
        Xp (np.ndarray)             -- Переданные пилотные символы
        pilot_loc (list or np.array)-- Индексы пилотных поднесущих
        Nfft (int)                  -- Размер FFT
        Nps (int)                   -- Интервал между пилотами
        int_opt (str)               -- Метод интерполяции: 'linear' или 'spline'
        
    Возвращает:
        H_LS (np.ndarray)           -- Оценка канала для всех поднесущих
    """
    # Количество пилотных поднесущих
    Np = Nfft // Nps

    # Оценка канала в точках пилотов: Y / Xp
    LS_est = Y[pilot_loc[:Np]] / Xp[:Np]

    # Выбор метода интерполяции
    if isinstance(int_opt, str) and len(int_opt) > 0:
        if int_opt.lower()[0] == 'l':
            method = 'linear'
        elif int_opt.lower()[0] == 's':
            method = 'spline'
        else:
            raise ValueError("Неподдерживаемый метод интерполяции. Используйте 'linear' или 'spline'.")
    else:
        method = 'linear'

    # Интерполяция оценки канала на все поднесущие
    H_LS = interpolate(LS_est, pilot_loc[:Np], Nfft, method)

    return H_LS
def interpolate(H, pilot_loc, Nfft, method):
    """
    Интерполирует оценку канала H по всем поднесущим.
    
    Параметры:
        H (np.ndarray)             -- Оценка канала на пилотных поднесущих (комплексные числа)
        pilot_loc (list or np.array) -- Индексы пилотных поднесущих
        Nfft (int)                 -- Общее количество поднесущих (размер FFT)
        method (str)               -- Метод интерполяции: 'linear' или 'spline'
    
    Возвращает:
        H_interpolated (np.ndarray) -- Интерполированная оценка канала для всех поднесущих
    """
    # Преобразование в массив NumPy
    pilot_loc = np.array(pilot_loc)
    H = np.array(H)

    # Экстраполяция в начало, если первый пилот не равен 0
    if pilot_loc[0] > 0:
        if len(pilot_loc) >= 2:
            slope = (H[1] - H[0]) / (pilot_loc[1] - pilot_loc[0])
        else:
            slope = 0  # Если только один пилот — не делаем экстраполяцию
        H = np.insert(H, 0, H[0] - slope * pilot_loc[0])
        pilot_loc = np.insert(pilot_loc, 0, 0)

    # Экстраполяция в конец, если последний пилот < Nfft - 1
    if pilot_loc[-1] < Nfft - 1:
        if len(pilot_loc) >= 2:
            slope = (H[-1] - H[-2]) / (pilot_loc[-1] - pilot_loc[-2])
        else:
            slope = 0
        H = np.append(H, H[-1] + slope * (Nfft - 1 - pilot_loc[-1]))
        pilot_loc = np.append(pilot_loc, Nfft - 1)

    # Разделение на реальную и мнимую части для корректной интерполяции
    real_part = np.real(H)
    imag_part = np.imag(H)

    # Выбор метода интерполяции
    if method.lower() == 'linear':
        kind = 'linear'
    elif method.lower() in ['spline', 'cubic']:
        kind = 'cubic'
    else:
        raise ValueError("Неподдерживаемый метод интерполяции. Используйте 'linear' или 'spline'.")

    # Создание интерполяторов для вещественной и мнимой части
    interp_real = interp1d(pilot_loc, real_part, kind=kind, fill_value='extrapolate')
    interp_imag = interp1d(pilot_loc, imag_part, kind=kind, fill_value='extrapolate')

    # Применение интерполяции ко всем поднесущим
    freq_indices = np.arange(Nfft)
    H_real = interp_real(freq_indices)
    H_imag = interp_imag(freq_indices)
    print("H_real",(H_real))

    return H_real + 1j * H_imag

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
    for i in range(int(len(b)/Nbps)):
        start = i * Nbps
        end = start + Nbps
        B = tuple(b[start:end])  # Получаем группу битов как кортеж
        Q = mapping_table[B]
        N[i] = Q
        # print("N: ",N)
        # print("Q: ",Q)
        # print("Кол-во битов:",len(bits))
        # print("Кол-во комлексных чисел на выходе:",int(len(bits)/Nbps))

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

def calculate_EVM(tx_symbols, rx_symbols):
    if len(tx_symbols) != len(rx_symbols):
        raise ValueError("Количество переданных и принятых символов должно совпадать")

    error_vector = np.abs(tx_symbols - rx_symbols)
    signal_power = np.mean(np.abs(tx_symbols) ** 2)

    evm = np.sqrt(np.mean(error_vector ** 2) / signal_power) * 100  # в процентах
    return evm
def calculate_BER(tx_bits, rx_bits):
    if len(tx_bits) != len(rx_bits):
        raise ValueError("Длина переданных и принятых бит различается")
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits)
    return ber



def demapper(received_symbols):
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

    demapped_bits = []
    constellation = np.array(list(mapping_table.values()))
    for symbol in received_symbols:
        distances = np.abs(symbol - constellation)
        nearest_idx = np.argmin(distances)
        bits = list(mapping_table.keys())[nearest_idx]
        demapped_bits.extend(bits)
    return np.array(demapped_bits)





# === Передача одного OFDM символа через SDR ===
for nsym in range(Nsym):  # Nsym = 1 для тестирования
    # --- Генерация пилотов ---
    bits_pilot = np.random.randint(0, 2, size=Np * Nbps)
    Xp = Mapper(bits_pilot, Nbps)

    # --- Генерация данных ---
    bits = np.random.binomial(n=1, p=0.5, size=(Nfft - Np) * Nbps)
    My_Data = Mapper(bits, Nbps)

    # --- Формирование OFDM символа ---
    ip = 0
    pilot_loc = []
    X = np.zeros(Nfft, dtype=complex)
    buf = 0
    for k in range(Nfft):
        if buf == 0:
            X[k] = Xp[ip]
            pilot_loc.append(k)
            ip += 1
            buf = Nps
        else:
            X[k] = My_Data[k - ip]
            buf -= 1

    # --- IFFT и добавление CP ---
    x = np.fft.ifft(X)
    xt = np.concatenate((x[-Ng:], x))

    # --- Передача через SDR ---
    sdr.tx_cyclic_buffer = False  # Не повторяем сигнал
    sdr.tx(xt.astype(np.complex64))  # Отправка сигнала

    # --- Приём сигнала ---
    rx_data = sdr.rx()
    print("Принято от SDR:", len(rx_data), "сэмплов")

    # --- Удаление CP ---
    y = rx_data[Ng:Nofdm]

    # --- FFT ---
    Y = np.fft.fft(y)

    # --- Оценка канала LS ---
    H_est = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, 'linear')

    # --- Компенсация канала ---
    Y_eq = Y / H_est

    # --- Демаппинг данных ---
    Data_extracted = []
    ip = 0
    for k in range(Nfft):
        if k % Nps == 0:
            ip += 1
        else:
            Data_extracted.append(Y_eq[k])
    Data_extracted = np.array(Data_extracted)
    bits_rx = demapper(Data_extracted)

    # --- Расчёт BER/EVM ---
    ber = calculate_BER(bits, bits_rx)
    evm = calculate_EVM(X[~np.isin(np.arange(Nfft), pilot_loc)], Y_eq[~np.isin(np.arange(Nfft), pilot_loc)])
    print(f"BER: {ber:.4f}, EVM: {evm:.2f}%")