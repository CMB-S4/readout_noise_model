import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
import matplotlib.pylab as plt
import time

import sys
sys.path.append('Scripts/')
from colorednoise import powerlaw_psd_gaussian

#of = open("estimate_alias.dat", 'w')
#of.write(f"# {'f3dB':20}\t{'num_downsamples':20}\t{'adc_freq':20}\t{'row_len':20}\t{'num_rows':20}\t\t{'num_array_visits':20}\t{'AFdB':20}\n")

def aliased_noise_fft(num_array_visits, num_downsamples, adc_freq, row_len, num_rows, analog_f3db_hz, do_plot, v_white_noise, v_pink_noise):
    # Number of samples per array visit
    array_len = row_len*num_rows
    start = time.time()
    
    # Generate high bw white noise
    mean = 0
    std = v_white_noise * np.sqrt(adc_freq/2) #in units of volts, assuming v_noise in v/rt(hz)
    num_samples = int(num_array_visits*array_len)
    adc_dt = 1./adc_freq
    adc_samples = np.random.normal(mean, std, size=num_samples)
    adc_sample_times = np.arange(num_samples)*adc_dt
    
    #Generate pink noise
    adc_pink_noise = powerlaw_psd_gaussian(1, num_samples)*v_pink_noise#sample rate is 1, amplitude is 1, both need to be scaled

    adc_samples = adc_samples + adc_pink_noise
    
    # Filter to desired analog bw
    b, a = signal.butter(N=1, Wn=analog_f3db_hz, btype='low', fs=adc_freq)
    zi = signal.lfilter_zi(b, a)
    adc_samples_filt, _ = signal.lfilter(
        b, a, adc_samples, zi=zi*adc_samples[0])
    
    #
    # Scale by Nyquist freq so that power spectral density before
    # downsampling is ~1
    pxx_raw_den = rfft(adc_samples)
    f_raw = rfftfreq(num_samples, 1 / adc_freq)
    pxx_filt_den = rfft(adc_samples_filt)
    f_filt = rfftfreq(num_samples, 1 / adc_freq)

    v_noise_raw = np.mean(pxx_raw_den)
    
    # Simulate tmux sampling
    downsamples = []
    for array_visit in range(int(num_array_visits)):
        row_visit = adc_samples_filt[array_visit *
                                     array_len:array_visit*array_len+row_len]
        row_visit_downsamples = row_visit[row_len-num_downsamples:]
        downsamples.append(np.average(row_visit_downsamples))

    pxx_downsample_den = rfft(downsamples)
    f_downsample = rfftfreq(len(downsamples), float(row_len*num_rows) / adc_freq)

    #AFdB = 10.*np.log10(np.mean(pxx_downsample_den)*(adc_freq/2.))
    #print(f'Calculating from dB {10**(AFdB/20)}')
    #print(f'* AFdB = {AFdB} dBpwr')
    v_noise_aliased = np.sqrt(np.max(pxx_downsample_den))
    print(f'Aliased white noise level is {v_noise_aliased} V/rt(Hz)')
    
    #of.write(
    #    f"{analog_f3db_hz:.4e}\t{num_downsamples:20}\t{adc_freq:.3e}\t{row_len:20}\t{num_rows:20}\t{num_array_visits:.3e}\t{AFdB:.2f}\n")

    if do_plot:
        plt.figure()
        plt.title(
            f"F3dB={analog_f3db_hz:.4e} nds={num_downsamples} adcf={adc_freq:.3e} rl={row_len} nr={num_rows} nav={num_array_visits:.3e} aliased_noise={v_noise_aliased:.2f} V/rt(Hz)\n", fontsize=8)
        # plt.semilogx(f, 10.*np.log10(pxx_den))
        plt.xlim(1, adc_freq/2.)
        plt.ylim(5e-10, 1e-6)
        #plt.semilogx(f_raw, 10.*np.log10(pxx_raw_den*(adc_freq/2.)))
        #plt.semilogx(f_filt, 10.*np.log10(pxx_filt_den*(adc_freq/2.)))
        plt.loglog(f_raw, np.sqrt(pxx_raw_den))
        plt.loglog(f_filt, np.sqrt(pxx_filt_den))

        #plt.semilogx(f_downsample, 10. *
        #             np.log10(pxx_downsample_den*(adc_freq/2.)))
        plt.loglog(f_downsample, np.sqrt(pxx_downsample_den))

    #of.close()
    end = time.time()
    print(end-start)
    return(v_noise_aliased, f_downsample, pxx_downsample_den)
    plt.show()