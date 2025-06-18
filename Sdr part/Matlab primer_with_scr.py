import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve






# === Настройки SDR ===
sdr_ip = "ip:192.168.2.1"  # IP PlutoSDR по умолчанию
sample_rate = int(1e6)
center_freq = int(2.4e9)  # Рабочая частота, например 2.4 ГГц

# Инициализация SDR
sdr = adi.Pluto(sdr_ip)
sdr.sample_rate = sample_rate
sdr.rx_buffer_size = 2048
sdr.tx_hardwaregain_control = -90  # Усиление TX: -89.75 до 0 дБ
#sdr.gain_control_mode_chan0 = "fast_attack"  # AGC или manual
sdr.rx_hardwaregain_chan0 = 0    # Приемное усиление (0–71 дБ)

# Параметры системы
Nfft = 64  # Размер FFT
Ng = Nfft // 2  # Длина циклического префикса
Nofdm = Nfft + Ng  # Общая длина OFDM символа
Nsym = 1  # Количество OFDM символов 100 было
Nps = 8  # Интервал между пилотами
Np = Nfft // Nps  # Количество пилотов на OFDM символ
Nbps = 4  # Количество бит на символ
M = 2 ** Nbps  # Размер QAM модуляции
Es = 1  # Энергия сигнала
A = np.sqrt(3 / 2 / (M - 1) * Es)  # Нормализующий коэффициент для QAM
MSE = np.zeros(6)  # Массив для хранения MSE
nose = 0  # Счетчик ошибок

print("Длина циклического префикса Ng =",Ng,"\n")
print("Общая длина OFDM символа Nofdm =",Nofdm,"\n")
print("Количество пилотов на OFDM символ Np =",Np,"\n")
print("Размер QAM модуляции M =",M,"\n")
print("Нормализующий коэффициент для QAM A =",A,"\n")

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

def detect_ofdm_symbol(rx_signal, cp_len, fft_len, threshold=0.7):
    """Улучшенный детектор OFDM символа с визуализацией"""
    N = len(rx_signal)
    corr = np.zeros(N - fft_len - cp_len)
    energy = np.zeros_like(corr)
    
    for i in range(len(corr)):
        cp = rx_signal[i:i+cp_len]
        tail = rx_signal[i+fft_len:i+fft_len+cp_len]
        corr[i] = np.abs(np.sum(cp * np.conj(tail)))**2
        energy[i] = 0.5*(np.sum(np.abs(cp)**2) + np.sum(np.abs(tail)**2))
    
    norm_corr = corr / (energy + 1e-10)
    peak_idx = np.argmax(norm_corr)
    peak_val = norm_corr[peak_idx]
    
    # Визуализация
    plt.figure(figsize=(12,4))
    plt.plot(norm_corr, label='Нормализованная корреляция')
    plt.axhline(threshold, color='r', linestyle='--', label='Порог')
    plt.axvline(peak_idx, color='g', linestyle=':', label='Обнаруженный пик')
    plt.title(f"Обнаружение CP (пик: {peak_val:.2f})")
    plt.xlabel("Отсчеты")
    plt.ylabel("Корреляция")
    plt.legend()
    plt.grid()
    plt.show()
    
    if peak_val < threshold:
        print(f"Внимание: слабый пик корреляции {peak_val:.2f} < {threshold}")
        return 0, norm_corr
    
    return peak_idx, norm_corr

def autocorrelation_cp_detection(received_signal, fft_size, cp_size, plot_result=True):
    """
    Обнаружение начала OFDM-символа с помощью автокорреляции циклического префикса.
    
    Параметры:
    received_signal - принятый сигнал
    fft_size - размер FFT (длина полезного символа)
    cp_size - размер циклического префикса
    plot_result - флаг для отображения графика корреляции
    
    Возвращает:
    index - индекс начала OFDM-символа
    correlation - массив значений корреляции
    """
    N = len(received_signal)
    correlation = np.zeros(N - fft_size - cp_size)
    
    # Вычисляем автокорреляцию между сигналом и его копией, сдвинутой на fft_size
    for n in range(len(correlation)):
        sum_corr = 0
        for k in range(cp_size):
            sum_corr += received_signal[n + k] * np.conj(received_signal[n + k + fft_size])
        correlation[n] = np.abs(sum_corr)
    
    # Нормализация (опционально)
    # correlation = correlation / np.max(correlation)
    
    # Находим пик корреляции
    peak_index = np.argmax(correlation)
    
    # Визуализация
    if plot_result:
        plt.figure(figsize=(12, 6))
        plt.plot(correlation, label='Автокорреляция')
        plt.axvline(x=peak_index, color='r', linestyle='--', label=f'Пик корреляции (индекс {peak_index})')
        plt.title('Автокорреляция для обнаружения OFDM символа')
        plt.xlabel('Индекс выборки')
        plt.ylabel('Уровень корреляции')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return peak_index, correlation

def find_signal_start_with_plot(x, cp_length):
    N = len(x)
    R = np.zeros(N - cp_length, dtype=complex)

    # Меняем диапазон: N - 2 * cp_length, чтобы не выйти за границу
    for n in range(N - 2 * cp_length + 1):  # <-- Здесь изменение
        corr = 0
        for k in range(cp_length):
            corr += x[n + k] * np.conj(x[n + k + cp_length])
        R[n] = corr

    start_index = np.argmax(np.abs(R))

    # Визуализация (остаётся без изменений)
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(R), label="Автокорреляция")
    plt.axvline(start_index, color='r', linestyle='--', label="Предполагаемое начало")
    plt.title("Модуль автокорреляции")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    zoom = 256
    start = max(0, start_index - 64)
    end = min(len(x), start + zoom)
    t = np.arange(start, end)
    plt.plot(t, np.real(x[start:end]), label="Re(Принятый сигнал)")
    plt.plot(t, np.imag(x[start:end]), label="Im(Принятый сигнал)")
    plt.axvline(start_index, color='r', linestyle='--', label="Начало сигнала")
    plt.title("Фрагмент сигнала вокруг точки синхронизации")
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return start_index, R

import numpy as np
import matplotlib.pyplot as plt

def find_ofdm_start_with_cp(rx_signal, cp_length, ofdm_length):
    signal_length = len(rx_signal)
    max_index = signal_length - (cp_length + ofdm_length)
    
    if max_index <= 0:
        raise ValueError("Сигнал слишком короткий для заданных параметров.")
    
    corr_values = np.zeros(max_index, dtype=np.complex128)

    for n in range(max_index):
        cp_segment = rx_signal[n : n + cp_length]
        sym_tail = rx_signal[n + ofdm_length : n + ofdm_length + cp_length]
        # Вычисляем корреляцию
        corr = np.sum(cp_segment * np.conj(sym_tail))
        corr_values[n] = corr

    # Найдём индекс максимального модуля автокорреляции
    start_index = np.argmax(np.abs(corr_values))

    # Визуализация
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(np.abs(corr_values), label="Модуль автокорреляции")
    plt.axvline(start_index, color='r', linestyle='--', label="Начало символа")
    plt.title("Автокорреляция для поиска начала OFDM-символа")
    plt.xlabel("Индекс")
    plt.ylabel("Значение корреляции")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    zoom_window = 128
    start = max(0, start_index - 32)
    end = min(len(rx_signal), start + zoom_window)
    t = np.arange(start, end)
    plt.plot(t, np.real(rx_signal[start:end]), label="Re(Принятый сигнал)")
    plt.plot(t, np.imag(rx_signal[start:end]), label="Im(Принятый сигнал)")
    plt.axvline(start_index, color='r', linestyle='--', label="Начало символа")
    plt.title("Фрагмент сигнала вокруг точки синхронизации")
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    return start_index, corr_values

# Основной цикл
for nsym in range(Nsym):
    


    # === Генерация QAM-16 пилотных символов ===

    # Шаг 1: Генерация случайных битов для пилотов
    bits_pilot = np.random.randint(0, 2, size=Np * Nbps)  # Np * 4 бита на Np пилотов

    # Шаг 2: Использование Mapper QAM-16
    Xp = Mapper(bits_pilot, Nbps)  # Теперь Xp — массив комплексных чисел QAM-16

    # Генерация данных
    bits = np.random.binomial(n=1, p=0.5, size = ((Nfft - Np) * 4) )# случайная генерация битовой последовательности
    print("Bits: ",bits)
    #Data = A * (2 * (msgint // np.sqrt(M)) - 1 + 1j * (2 * (msgint % np.sqrt(M)) - 1))
    My_Data =Mapper(bits,Nbps)# Передаем битовую последовательность (96 бит) и размер бит на символ (Nbps)
    #print(len(My_Data))

    # Формирование OFDM символа
    ip = 0
    pilot_loc = []
    X = np.zeros(Nfft, dtype=complex)
    buf = Nps

    buf = 0
    for k in range(Nfft):
        if buf == 0:
            X[k] = Xp[ip]
            pilot_loc.append(k)
            ip += 1
            buf = Nps
            print(k," ")
        else:
            X[k] = My_Data[k-ip]
            buf-=1
    print("\n",ip," np",Np,"\n"," X: ",len(X))

    for k in range(Nfft):
        if k % Np == 0:
            print("\n")

        if k % 9 == 0:
            print(X[k],end="")
        else:
            print("  0  ",end="")
    print("\n")

    for k in range(Nfft):
        if k % Np == 0:
            print("\n")

        if k % 9 == 0:
            print(X[k],end="")
        else:
            print(X[k],end="")
    print("\n")


   

    # IFFT и добавление циклического префикса
    #x = np.fft.ifft(X)
    x = np.fft.ifft(X) * np.sqrt(Nfft)
    xt = np.concatenate((x[-Ng:], x))
    Xt = np.concatenate((X[-Ng:], X))

  # Передача с повторением для надежности
    sdr.tx_cyclic_buffer = False
    sdr.tx(Xt)

    # Прием с задержкой
    plt.pause(0.3)
    rx_data = sdr.rx()
    print(rx_data)
    # --- Синхронизация ---
    #start_idx, corr = detect_ofdm_symbol(rx_data, Ng, Nfft)
    #start_idx, corr = autocorrelation_cp_detection(rx_data, Nfft, Ng)
    #start_idx, corr = find_signal_start_with_plot(rx_data,Ng)
    start_idx, corr =find_ofdm_start_with_cp(rx_data, Ng, Nfft)
    print(f"Начало OFDM-символа: {start_idx}")
    

    y = rx_data[start_idx+Ng : start_idx+Ng+Nfft]
    Y = np.fft.fft(y) / np.sqrt(Nfft)

    # --- Удаление CP и извлечение полезной части ---
    #y = rx_data[start_idx + Ng : start_idx + Ng + Nfft]

    # --- FFT ---
    #Y = np.fft.fft(y) / np.sqrt(Nfft)  # Нормализация
    #Y = np.fft.fft(y)

    # Вывод информации
    print("Длина принятого сигнала:", len(rx_data))
    print("Индекс начала OFDM-символа:", start_idx)
    print("Длина полезной части после удаления CP:", len(y))
    # # График корреляции
    # plt.figure(figsize=(12, 4))
    # plt.plot(corr, label="Значения корреляции")
    # plt.axvline(x=start_idx, color='r', linestyle="--", label=f"Начало OFDM-символа ({start_idx})")
    # plt.title("График корреляции для обнаружения циклического префикса")
    # plt.xlabel("Отсчёт")
    # plt.ylabel("Значение корреляции")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    

    # --- Удаление CP с учётом найденного начала ---
    #y = rx_data[start_idx + Ng : start_idx + Ng + Nfft]

    # --- FFT ---
    #Y = np.fft.fft(y) / np.sqrt(Nfft)  # нормализация
    # # --- Удаление CP ---
    # y = rx_data[Ng:Nofdm]

    # --- FFT ---
    #Y = np.fft.fft(y)
    #Y = np.fft.fft(y) / np.sqrt(Nfft)
    # #Y = np.fft.fft(y) / np.sqrt(Nfft)  # нормализация


    # === Визуализация IFFT / FFT ===
    plt.figure(figsize=(16, 10))

    # --- 1. Сигнал в частотной области (X[k]) ---
    plt.subplot(4, 1, 1)
    plt.plot(np.abs(X), 'b-o', label='Амплитуда')
    plt.title("1. Сигнал в частотной области $X[k]$")
    plt.xlabel("Поднесущая k")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()

    # --- 2. OFDM-символ во временной области (x[n]) ---
    plt.subplot(4, 1, 2)
    plt.plot(np.real(xt), 'b', label='Re(x[n])')
    plt.plot(np.imag(xt), 'r--', label='Im(x[n])')
    plt.title("2. OFDM-символ во временной области $x[n]$ (после IFFT)")
    plt.xlabel("Отсчёт n")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.legend()

    # --- 3. Принятый сигнал после удаления CP (y[n]) ---
    plt.subplot(4, 1, 3)
    plt.plot(np.real(y), 'b', label='Re(y[n])')
    plt.plot(np.imag(y), 'r--', label='Im(y[n])')
    plt.title("3. Принятый сигнал $y[n]$ (после удаления CP)")
    plt.xlabel("Отсчёт n")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.legend()

    # --- 4. После FFT — принятый сигнал в частотной области (Y[k]) ---
    plt.subplot(4, 1, 4)
    plt.plot(np.abs(Y), 'g-o', label='Амплитуда')
    plt.title("4. Принятый сигнал в частотной области $Y[k]$ (после FFT)")
    plt.xlabel("Поднесущая k")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("отправляем xt",xt,len(xt))
    print("приняли y после синхры",y,len(y))
    print("Y",Y,len(Y))

    # === Визуализация ===
    plt.figure(figsize=(12, 5))

    # --- 1. Исходный сигнал в частотной области ---
    plt.subplot(1, 2, 1)
    plt.stem(np.abs(X))
    plt.title("X[k] — переданный сигнал в частотной области")
    plt.xlabel("Поднесущая k")
    plt.ylabel("Амплитуда")

    # --- 2. После FFT (принятый сигнал Y[k]) ---
    plt.subplot(1, 2, 2)
    plt.stem(np.abs(Y))
    plt.title("Y[k] — принятый сигнал в частотной области")
    plt.xlabel("Поднесущая k")
    plt.ylabel("Амплитуда")

    plt.tight_layout()
    plt.show()

    my_y_Data = np.zeros(Nfft-Np, dtype=complex)
    buf = Nps

    buf = 0
    for k in range(Nfft):
        if k % 9 != 0:
            my_y_Data[buf] = Y[k]
            buf+=1

    plt.figure("QAM-16 приняли Y",figsize=(10,10))
    for k in range(len(my_y_Data)):
        # Отрисовка точки
        plt.plot(my_y_Data[k].real, my_y_Data[k].imag, 'o', markersize=10, markerfacecolor='blue', markeredgecolor='black')
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Constellation Diagram of 16-QAM')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


    H_est = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, 'linear')
    #H_est = MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, h, SNR)
    print("H_est: ",H_est)
    # DFT-based оценка канала
    h_est = np.fft.ifft(H_est)
    h_DFT = h_est[:len(rx_data)]
    H_DFT = np.fft.fft(h_DFT, Nfft)


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
    Data_extracted = demapper(Data_extracted)

    plt.figure("QAM-16_",figsize=(10,10))
    for k in range(Nfft-Np):
        # Отрисовка точки
        plt.plot(Y_eq[k].real, Y_eq[k].imag, 'o', markersize=10, markerfacecolor='blue', markeredgecolor='black')
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Constellation Diagram of 16-QAM')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

print("TX_Data: ",bits)
print("RX_Data: ",Data_extracted)


ber = calculate_BER(bits, Data_extracted)
print(f"BER: {ber:.4f}")
evm = calculate_EVM(bits, Data_extracted)
print(f"EVM: {evm:.2f}%")

plt.tight_layout()
plt.show()