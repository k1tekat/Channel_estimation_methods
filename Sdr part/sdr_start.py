import adi
sdr = adi.Pluto('ip:192.168.2.1') # адрес PlutoSDR
sdr.sample_rate = int(2.5e6) # колчество временных отсчето в 1 [сек]
rx_data = sdr.rx()
print("rx data length: ",len(rx_data))

for i in rx_data:
    print(i)