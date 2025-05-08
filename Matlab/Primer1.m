% channel_estimation.m
% LS/DFT Channel Estimation with linear/spline interpolation
clear; close all; clc;

% Parameters
Nfft = 32; 
Ng = Nfft/8; 
Nofdm = Nfft + Ng; 
Nsym = 100;
Nps = 4; 
Np = Nfft/Nps; % Pilot spacing
Nbps = 4; 
M = 2^Nbps; % 16-QAM
Es = 1; 
A = sqrt(3/2/(M-1)*Es); % Normalization
SNR = 30; 

% Создаем QAM constellation (вручную)
constellation = A * qammod(0:M-1, M, 'UnitAveragePower', false);

% Initialize
MSE = zeros(1, 6); 
err_count = 0;

for nsym = 1:Nsym
    % Pilot sequence (BPSK)
    Xp = 2*(randi([0 1], 1, Np)) - 1;
    
    % Generate random symbols
    msgint = randi([0 M-1], 1, Nfft-Np);
    
    % Modulate data (используем qammod из Communications Toolbox или самодельную реализацию)
    Data = constellation(msgint + 1); % +1 потому что MATLAB индексирует с 1
    
    % Insert pilots
    X = zeros(1, Nfft);
    pilot_loc = zeros(1, Np);
    data_idx = 1;
    pilot_idx = 1;
    
    for k = 1:Nfft
        if mod(k-1, Nps) == 0
            X(k) = Xp(pilot_idx);
            pilot_loc(pilot_idx) = k;
            pilot_idx = pilot_idx + 1;
        else
            X(k) = Data(data_idx);
            data_idx = data_idx + 1;
        end
    end
    
    % OFDM modulation
    x = ifft(X, Nfft);
    xt = [x(end-Ng+1:end) x];
    
    % Channel (2-tap)
    h = [(randn + 1i*randn) (randn + 1i*randn)/2];
    H = fft(h, Nfft);
    ch_len = length(h);
    H_pow = 10*log10(abs(H).^2);
    
    % Channel transmission
    y_channel = conv(xt, h);
    yt = awgn(y_channel, SNR, 'measured');
    y = yt(Ng+1:Ng+Nfft);
    Y = fft(y);
    
    % Channel estimation methods
    methods = {'linear', 'spline', 'mmse'};
    for m = 1:3
        if strcmp(methods{m}, 'mmse')
            H_est = MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, h, SNR);
            method_name = 'MMSE';
        else
            H_est = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, methods{m});
            method_name = ['LS-' methods{m}];
        end
        
        % DFT processing
        h_est = ifft(H_est);
        h_dft = h_est(1:ch_len);
        H_dft = fft(h_dft, Nfft);
        
        % Calculate powers
        H_est_pow = 10*log10(abs(H_est).^2);
        H_dft_pow = 10*log10(abs(H_dft).^2);
        
        % Plot first symbol
        if nsym == 1
            subplot(3,2,2*m-1);
            plot(H_pow, 'b'); hold on;
            plot(H_est_pow, 'r:+');
            title([method_name ' Estimation']);
            legend('True', 'Estimated');
            
            subplot(3,2,2*m);
            plot(H_pow, 'b'); hold on;
            plot(H_dft_pow, 'r:+');
            title([method_name ' with DFT']);
            legend('True', 'Estimated');
        end
        
        % Calculate MSE
        MSE(m) = MSE(m) + sum(abs(H - H_est).^2);
        MSE(m+3) = MSE(m+3) + sum(abs(H - H_dft).^2);
    end
    
    % Equalization and demodulation
    Y_eq = Y ./ H_est;
    data_rx = Y_eq(setdiff(1:Nfft, pilot_loc));
    % Equalization and demodulation
    Y_eq = Y ./ H_est;
    data_rx = Y_eq(setdiff(1:Nfft, pilot_loc));
    msg_detected = qamdemod(data_rx / A, M, 'UnitAveragePower', false);
    err_count = err_count + sum(msg_detected ~= msgint);
end

% Final results
MSE = MSE / (Nfft * Nsym);
BER = err_count / (Nsym * (Nfft-Np) * Nbps);

disp(['Average MSE: ' num2str(mean(MSE))]);
disp(['Bit Error Rate: ' num2str(BER)]);

function H_interpolated = interpolate(H, pilot_loc, Nfft, method)
    % Interpolation function for channel estimation
    % Inputs:
    % H = Channel estimate using pilot sequence
    % pilot_loc = Location of pilot sequence (indices)
    % Nfft = FFT size
    % method = 'linear' or 'spline'
    % Output:
    % H_interpolated = Interpolated channel

    % Extend the pilot locations and estimates to cover the full range [1, Nfft]
    if pilot_loc(1) > 1
        % Extrapolate to the left (before the first pilot)
        slope = (H(2) - H(1)) / (pilot_loc(2) - pilot_loc(1));
        H = [H(1) - slope * (pilot_loc(1) - 1), H];
        pilot_loc = [1, pilot_loc];
    end

    if pilot_loc(end) < Nfft
        % Extrapolate to the right (after the last pilot)
        slope = (H(end) - H(end-1)) / (pilot_loc(end) - pilot_loc(end-1));
        H = [H, H(end) + slope * (Nfft - pilot_loc(end))];
        pilot_loc = [pilot_loc, Nfft];
    end

    % Perform interpolation based on the specified method
    freq_indices = 1:Nfft; % All subcarrier indices
    if strcmpi(method, 'linear')
        H_interpolated = interp1(pilot_loc, H, freq_indices, 'linear', 'extrap');
    elseif strcmpi(method, 'spline')
        H_interpolated = interp1(pilot_loc, H, freq_indices, 'spline', 'extrap');
    else
        error('Invalid interpolation method. Use ''linear'' or ''spline''.');
    end
end

function H_LS = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, int_opt)
    % LS channel estimation function
    % Inputs:
    % Y = Frequency-domain received signal
    % Xp = Pilot signal
    % pilot_loc = Pilot location indices
    % Nfft = FFT size
    % Nps = Pilot spacing
    % int_opt = 'linear' or 'spline'
    % Output:
    % H_LS = LS Channel estimate

    % Number of pilots
    Np = Nfft / Nps;
    k = 1 : Np;

    % LS channel estimation at pilot locations
    LS_est = Y(pilot_loc(k)) ./ Xp(k);

    if lower(int_opt(1))=='l', method='linear'; else method='spline'; end
    % Linear/Spline interpolation
    H_LS = interpolate(LS_est,pilot_loc,Nfft,method);
end

function [H_MMSE] = MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, h, SNR)
    % MMSE channel estimation function
    %
    % Inputs:
    % Y          = Frequency-domain received signal
    % Xp         = Pilot signal
    % pilot_loc  = Pilot location indices
    % Nfft       = FFT size
    % Nps        = Pilot spacing
    % h          = Channel impulse response
    % SNR        = Signal-to-Noise Ratio [dB]
    %
    % Output:
    % H_MMSE     = MMSE channel estimate

    % Convert SNR from dB to linear scale
    snr = 10^(SNR * 0.1);

    % Number of pilots
    Np = Nfft / Nps;
    k = 1:Np;

    % LS estimate at pilot locations (Eq. 6.12 or 6.8)
    H_tilde = Y(1, pilot_loc(k)) ./ Xp(k);


    k=0:length(h)-1; %k_ts = k*ts;
    hh = h*h'; tmp = h.*conj(h).*k; %tmp = h.*conj(h).*k_ts;
    r = sum(tmp)/hh; r2 = tmp*k.'/hh; %r2 = tmp*k_ts.’/hh;


    % Calculate RMS delay spread
    tau_rms = sqrt(r2-r^2); % rms delay

    % Frequency-domain parameters
    df = 1/Nfft; %1/(ts*Nfft);
    j2pi_tau_df = 1j * 2 * pi * tau_rms * df; % Complex exponential factor

    % Correlation matrices
    K1 = repmat((0:Nfft-1)', 1, Np); % Row vector for subcarriers
    K2 = repmat(0:Np-1, Nfft, 1); % Column vector for pilots
    rf = 1 ./ (1 + j2pi_tau_df * Nps * (K1 - K2)); % Eq. (6.17a)

    K3 = repmat((0:Np-1)', 1, Np); % Row vector for pilots
    K4 = repmat(0:Np-1, Np, 1); % Column vector for pilots
    rf2 = 1 ./ (1 + j2pi_tau_df * Nps * (K3 - K4)); % Eq. (6.17a)

    Rhp = rf; % Cross-correlation matrix between channel and pilots
    Rpp = rf2 + eye(length(H_tilde), length(H_tilde)) / snr; % Auto-correlation matrix of pilots (Eq. 6.14)

    % MMSE estimate (Eq. 6.15)
    H_MMSE = transpose(Rhp * inv(Rpp) * H_tilde'); % Final MMSE channel estimate
end