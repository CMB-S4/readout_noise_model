import numpy as np
from scipy import signal
import matplotlib.pylab as plt
import time

import sys
sys.path.append('Scripts/')
from powerlawnoise import powerlaw_psd_gaussian

#of = open("estimate_alias.dat", 'w')
#of.write(f"# {'f3dB':20}\t{'num_downsamples':20}\t{'adc_freq':20}\t{'row_len':20}\t{'num_rows':20}\t\t{'num_array_visits':20}\t{'AFdB':20}\n")

def aliased_noise(num_array_visits, num_downsamples, adc_freq, row_len, num_rows, analog_f3db_hz, rolloff_pole, analog_f3db_hz2, rolloff_pole2, do_plot, v_white_noise, f_corner, samplequotient, save_plot):
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
    v_pink_noise = np.sqrt(f_corner) * v_white_noise#Pink noise amplitude at 1 Hz is the product of the rolloff level and the rolloff frequency (in the same units)
    adc_pink_noise = powerlaw_psd_gaussian(1, num_samples)*v_pink_noise*4#sample rate is 1, amplitude is 1, both need to be scaled, I do not understand this factor of 4

    adc_samples = adc_samples + adc_pink_noise
    
    # Filter to desired analog bw
    b, a = signal.butter(N=rolloff_pole, Wn=analog_f3db_hz, btype='low', fs=adc_freq)
    zi = signal.lfilter_zi(b, a)
    adc_samples_filt, _ = signal.lfilter(
        b, a, adc_samples, zi=zi*adc_samples[0])
    b2, a2 = signal.butter(N=rolloff_pole2, Wn=analog_f3db_hz2, btype='low', fs=adc_freq)
    zi = signal.lfilter_zi(b, a)
    adc_samples_filt, _ = signal.lfilter(
        b, a, adc_samples_filt, zi=zi*adc_samples_filt[0])
    
    f_filt, pxx_filt_den = signal.welch(adc_samples_filt, adc_freq,
                                   nperseg=num_samples//samplequotient)

    if do_plot:
        # Calculate noise spectra for plotting purposes
        # Scale by Nyquist freq so that power spectral density before
        # downsampling is ~1
        f_raw, pxx_raw_den = signal.welch(adc_samples, adc_freq,
                                      nperseg=num_samples//samplequotient)
        f_pink, pxx_pink_den = signal.welch(adc_pink_noise, adc_freq,
                                      nperseg=num_samples//samplequotient)
        
        v_noise_raw = np.mean(pxx_raw_den) 
        
        plt.figure()
        plt.loglog(f_pink, np.sqrt(pxx_pink_den)*np.sqrt(f_pink)/v_pink_noise)
        #print(np.mean(np.sqrt(pxx_pink_den)*np.sqrt(f_pink)/v_pink_noise))
        #print(1/np.mean(np.sqrt(pxx_pink_den)*np.sqrt(f_pink)/v_pink_noise))
        
        #Plot spectra
        plt.figure()
        # plt.semilogx(f, 10.*np.log10(pxx_den))
        plt.xlim(0.1, adc_freq/2.)
        plt.ylim(5e-12, 1e-6)
        #plt.semilogx(f_raw, 10.*np.log10(pxx_raw_den*(adc_freq/2.)))
        #plt.semilogx(f_filt, 10.*np.log10(pxx_filt_den*(adc_freq/2.)))
        #plt.loglog(f_raw, np.sqrt(pxx_raw_den), label = 'Amplifier Input')
        plt.loglog(f_filt, np.sqrt(pxx_filt_den), label = 'Bandwidth Limited Noise')
        plt.loglog(f_pink, np.sqrt(pxx_pink_den), label = '1/f Noise Component')
        
        f_full = np.logspace(-1, np.log(adc_freq/2))
        plt.loglog(f_full, v_pink_noise/np.sqrt(f_full))
        plt.axvline(f_corner)#Show the intended rolloff frequency

        figure_title = 'Aliased Amplifier Noise Spectra'
        plt.title(figure_title)
        plt.grid(True, which = 'both')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Voltage Noise [V/rt(Hz)]')

    
    # Simulate tmux sampling
    downsamples = []
    for array_visit in range(int(num_array_visits)):
        row_visit = adc_samples_filt[array_visit *
                                     array_len:array_visit*array_len+row_len]
        row_visit_downsamples = row_visit[row_len-num_downsamples:]
        downsamples.append(np.average(row_visit_downsamples))

    f_downsample, pxx_downsample_den = signal.welch(downsamples, adc_freq/float(row_len*num_rows),
                                                    nperseg=len(downsamples)//samplequotient)

    #Evaluate voltage noise
    science_band = np.where(np.logical_and(f_downsample>=100, f_downsample<=400))
    v_noise_aliased= np.sqrt(np.mean(pxx_downsample_den[science_band]))
    v_noise_1f = v_pink_noise/np.sqrt(0.1) #evaluated at 0.1 Hz, low end of JAMA requirement

        

    
    #of.write(
    #    f"{analog_f3db_hz:.4e}\t{num_downsamples:20}\t{adc_freq:.3e}\t{row_len:20}\t{num_rows:20}\t{num_array_visits:.3e}\t{AFdB:.2f}\n")
    f_multiplexed = np.logspace(-1, np.log10(f_downsample[-1]))    
    total_noise_spectrum = np.sqrt(np.square(v_pink_noise/np.sqrt(f_multiplexed)) + np.square(v_noise_aliased))

    if do_plot:
        #plt.title(
            #f"F3dB={analog_f3db_hz:.4e} nds={num_downsamples} adcf={adc_freq:.3e} rl={row_len} nr={num_rows} nav={num_array_visits:.3e} aliased_noise={v_noise_aliased:.2f} V/rt(Hz)\n", fontsize=8)

        plt.loglog(f_downsample, np.sqrt(pxx_downsample_den), label = 'Multiplexing Aliased Noise')
        f_multiplexed = np.logspace(-1, np.log10(f_downsample[-1]))
        plt.loglog(f_multiplexed, total_noise_spectrum, label='2 component multiplexed noise result')
        plt.legend(loc='best')
        if save_plot:
            plt.savefig('figures/aliased_noise_example.png')#, bbox_inches='tight

    #of.close()
    end = time.time()
    #print(end-start)
    return(v_noise_aliased, v_noise_1f, f_filt, np.sqrt(pxx_filt_den), f_multiplexed, total_noise_spectrum)
    plt.show()
