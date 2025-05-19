import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

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
    #return signal

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


# Генерация случайного канала
h = np.random.randn(2) + 1j * np.random.randn(2)
H_true = np.fft.fft(h, Nfft)  # Истинный канал в частотной области
ch_length = len(h)


# Частотные индексы для отображения
freq_bins = np.arange(Nfft)

# === Визуализация ===

plt.figure(figsize=(12, 6))

# --- 1. Отображение канала во временной области ---
plt.subplot(2, 1, 1)
time_indices = np.arange(ch_length)
plt.stem(time_indices, np.abs(h), linefmt='b-', markerfmt='bo', basefmt=" ")
plt.title("Канал во временной области |h[n]|")
plt.xlabel("Индекс отсчета n")
plt.ylabel("Амплитуда")
plt.grid(True)

# --- 2. Отображение канала в частотной области ---
plt.subplot(2, 1, 2)
plt.plot(freq_bins, np.abs(H_true), 'r-', linewidth=2)
plt.title("Канал в частотной области |H[k]|")
plt.xlabel("Индекс поднесущей k")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.tight_layout()
plt.show()


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
    My_Data = Mapper(bits,Nbps)# Передаем битовую последовательность (96 бит) и размер бит на символ (Nbps)
    #print(len(My_Data))


    """
    1. сгенерировать последовательность бит                                   [true]
    2. сделать маппер  и разбить биты, сконвертировать их в Комплексные числа [true]
    3. добавить опорные сигналы (1+0J) например                               [true]
    """

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
    x = np.fft.ifft(X)
    xt = np.concatenate((x[-Ng:], x))

    # Прохождение через канал
    y_channel = fftconvolve(xt, h, mode='full')[:Nofdm]
    yt = awgn(y_channel, SNR)
    #print("y",yt,len(yt))




    y = yt[Ng:] #удаляем циклический префикс
    Y = np.fft.fft(y) 



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
    plt.plot(np.real(x), 'b', label='Re(x[n])')
    plt.plot(np.imag(x), 'r--', label='Im(x[n])')
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









    print("y",y,len(y))
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
    h_DFT = h_est[:ch_length]
    H_DFT = np.fft.fft(h_DFT, Nfft)


    # # Оценка канала
    # methods = ['ls linear', 'ls spline', 'MMSE']
    # for m, method in enumerate(methods):
    #     if method == 'ls linear' or method == 'ls spline':
    #         H_est = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, method)
    #     elif method == 'MMSE':
    #         H_est = MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, h, SNR)

    #     # DFT-based оценка канала
    #     h_est = np.fft.ifft(H_est)
    #     h_DFT = h_est[:ch_length]
    #     H_DFT = np.fft.fft(h_DFT, Nfft)

    #     # Вычисление MSE
    #     MSE[m] += np.sum(np.abs(H_true - H_est) ** 2)
    #     MSE[m + 3] += np.sum(np.abs(H_true - H_DFT) ** 2)

    #     # Визуализация (только для первого символа)
    #     if nsym == 0:
    #         H_true_dB = 10 * np.log10(np.abs(H_true) ** 2)
    #         H_est_dB = 10 * np.log10(np.abs(H_est) ** 2)
    #         H_DFT_dB = 10 * np.log10(np.abs(H_DFT) ** 2)

    #         plt.subplot(3, 2, 2 * m + 1)
    #         plt.plot(H_true_dB, 'b', label='True Channel')
    #         plt.plot(H_est_dB, 'r:', label=f'{method}')
    #         plt.legend()

    #         plt.subplot(3, 2, 2 * m + 2)
    #         plt.plot(H_true_dB, 'b', label='True Channel')
    #         plt.plot(H_DFT_dB, 'r:', label=f'{method} with DFT')
    #         plt.legend()

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
    print(Data_extracted)
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
    # msg_detected = np.round(((np.real(Data_extracted / A) + 1) / 2) * np.sqrt(M)) + \
    #                np.round(((np.imag(Data_extracted / A) + 1) / 2) * np.sqrt(M)) * np.sqrt(M)
    # nose += np.sum(msg_detected != My_Data)


    

# Вывод результатов
MSEs = MSE / (Nfft * Nsym)
print("MSE:", MSEs)
print("Bit Error Rate:", nose / (Nsym * (Nfft - Np)))
plt.tight_layout()
plt.show()