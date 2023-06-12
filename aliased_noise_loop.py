import numpy as np
from scipy import signal
import matplotlib.pylab as plt
import time

import sys
sys.path.append('Scripts/')
from colorednoise import powerlaw_psd_gaussian

#of = open("estimate_alias.dat", 'w')
#of.write(f"# {'f3dB':20}\t{'num_downsamples':20}\t{'adc_freq':20}\t{'row_len':20}\t{'num_rows':20}\t\t{'num_array_visits':20}\t{'AFdB':20}\n")

def aliased_noise_loop(num_array_visits, num_downsamples, adc_freq, row_len, num_rows, analog_f3db_hz, do_plot, v_white_noise, f_rolloff, samplequotient): 
    # Number of samples per array visit
    array_len = row_len*num_rows
    start = time.time()

    cycles = 20
    num_samples = int(num_array_visits*array_len)
    
    spectrum_length_downsamp = int((num_array_visits//samplequotient)/2) + 1
    v_noise_aliased_array = np.zeros(cycles)
    #f_downsample_array = np.zeros((cycles, spectrum_length_downsamp))
    pxx_downsample_den_array = np.zeros((cycles, spectrum_length_downsamp))
    
    spectrum_length = int((num_samples//samplequotient)/2) + 1
    #f_raw_array = np.zeros(cycles)
    pxx_raw_den_array = np.zeros((cycles, spectrum_length))
    #f_filt_array = np.zeros((cycles, spectrum_length))
    pxx_filt_den_array = np.zeros((cycles, spectrum_length))
    
    for i in range(cycles):

    
        # Generate high bw white noise
        mean = 0
        std = v_white_noise * np.sqrt(adc_freq/2) #in units of volts, assuming v_noise in v/rt(hz)
        adc_dt = 1./adc_freq
        adc_samples = np.random.normal(mean, std, size=num_samples)
        adc_sample_times = np.arange(num_samples)*adc_dt
    
        #Generate pink noise
        v_pink_noise = np.sqrt(f_rolloff) * v_white_noise#Pink noise amplitude at 1 Hz is the product of the rolloff level and the rolloff frequency (in the same units)
        adc_pink_noise = powerlaw_psd_gaussian(1, num_samples)*v_pink_noise#sample rate is 1, amplitude is 1, both need to be scaled

        adc_samples = adc_samples + adc_pink_noise
    
        # Filter to desired analog bw
        b, a = signal.butter(N=1, Wn=analog_f3db_hz, btype='low', fs=adc_freq)
        zi = signal.lfilter_zi(b, a)
        adc_samples_filt, _ = signal.lfilter(
            b, a, adc_samples, zi=zi*adc_samples[0])
    
        if do_plot:
            # Calculate noise spectra for plotting purposes
            # Scale by Nyquist freq so that power spectral density before
            # downsampling is ~1
            f_raw, pxx_raw_den_array[i, :] = signal.welch(adc_samples, adc_freq,
                                          nperseg=num_samples//samplequotient)
            f_filt, pxx_filt_den_array[i, :] = signal.welch(adc_samples_filt, adc_freq,
                                            nperseg=num_samples//samplequotient)
            v_noise_raw = np.mean(pxx_raw_den_array[i, :]) 
        
    
        # Simulate tmux sampling
        downsamples = []
        for array_visit in range(int(num_array_visits)):
            row_visit = adc_samples_filt[array_visit *
                                         array_len:array_visit*array_len+row_len]
            row_visit_downsamples = row_visit[row_len-num_downsamples:]
            downsamples.append(np.average(row_visit_downsamples))

        f_downsample, pxx_downsample_den_array[i, :] = signal.welch(downsamples, adc_freq/float(row_len*num_rows),
                                                        nperseg=len(downsamples)//samplequotient)
        #AFdB = 10.*np.log10(np.mean(pxx_downsample_den)*(adc_freq/2.))
        #print(f'Calculating from dB {10**(AFdB/20)}')
        #print(f'* AFdB = {AFdB} dBpwr')
        v_noise_aliased_array[i] = np.sqrt(np.max(pxx_downsample_den_array[i, :]))
        #print(f'Aliased white noise level is {v_noise_aliased} V/rt(Hz)')
    
        #of.write(
            #    f"{analog_f3db_hz:.4e}\t{num_downsamples:20}\t{adc_freq:.3e}\t{row_len:20}\t{num_rows:20}\t{num_array_visits:.3e}\t{AFdB:.2f}\n")


        #of.close()
        end = time.time()
    print(end-start)
    
    pxx_downsample_den = np.mean(pxx_downsample_den_array, axis=0)
    pxx_raw_den = np.mean(pxx_raw_den_array, axis=0)
    pxx_filt_den = np.mean(pxx_filt_den_array, axis=0)
    
    
    v_noise_aliased = np.sqrt(np.max(pxx_downsample_den))#evaluated at 400 Hz, high end of JAMA requirement
    
    v_noise_pink #evaluated at 0.1 Hz, low end of JAMA requirement
        
    if do_plot:
        #Plot spectra
        plt.figure()
        # plt.semilogx(f, 10.*np.log10(pxx_den))
        plt.xlim(1, adc_freq/2.)
        plt.ylim(5e-10, 1e-6)
        #plt.semilogx(f_raw, 10.*np.log10(pxx_raw_den*(adc_freq/2.)))
        #plt.semilogx(f_filt, 10.*np.log10(pxx_filt_den*(adc_freq/2.)))
        plt.loglog(f_raw, np.sqrt(pxx_raw_den_array[i, :]), label = 'Amplifier Input')
        plt.loglog(f_filt, np.sqrt(pxx_filt_den_array[i, :]), label = 'Bandwidth Limited Noise')
        plt.axvline(f_rolloff)#Show the intended rolloff frequency

        figure_title = 'Aliased Amplifier Noise Spectra'
        plt.title(figure_title)
        plt.grid(True, which = 'both')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Voltage Noise [V/rt(Hz)]')

        
        #plt.title(
                #f"F3dB={analog_f3db_hz:.4e} nds={num_downsamples} adcf={adc_freq:.3e} rl={row_len} nr={num_rows} nav={num_array_visits:.3e} aliased_noise={v_noise_aliased:.2f} V/rt(Hz)\n", fontsize=8)

        plt.loglog(f_downsample, np.sqrt(pxx_downsample_den_array[i, :]), label = 'Multiplexing Aliased Noise')
        plt.legend(loc='best')

    
        plt.show()
        
    return(v_noise_aliased, f_downsample, pxx_downsample_den)
