import numpy as np
from scipy import signal
from scipy.integrate import quad
import matplotlib.pylab as plt
import time
import configparser

import sys
import json
sys.path.append('Scripts/')
from powerlawnoise import powerlaw_psd_gaussian
from aliased_noise import aliased_noise

def Butterworth_Transfer_Function(f, fc, n):
    V_out = 1/np.sqrt(1 + (f/fc)**(2*n))
    return V_out

def k_manganin(T):
    T_list = np.array([0.1, 0.4, 1, 4, 10, 20, 80, 150, 300])
    k_list = np.array([6e-3, 2e-2, 6e-2, 5e-1, 2, 3.3, 13, 16, 22])
    k_out = np.interp(T, T_list, k_list)
    return k_out

def kT_manganin(T):
    T_list = np.array([0.1, 0.4, 1, 4, 10, 20, 80, 150, 300])
    k_list = np.array([6e-3, 2e-2, 6e-2, 5e-1, 2, 3.3, 13, 16, 22])
    k_out = np.interp(T, T_list, k_list)
    return k_out*T

def kTi_manganin(T):
    T_list = np.array([0.1, 0.4, 1, 4, 10, 20, 80, 150, 300])
    k_list = np.array([6e-3, 2e-2, 6e-2, 5e-1, 2, 3.3, 13, 16, 22])
    k_out = np.interp(T, T_list, k_list)
    return k_out/T

def noise_output(noisetag, configfilename, squidfilename):
    configfile = configparser.ConfigParser()
    configfile.read(configfilename)
    squidfile = configparser.ConfigParser()
    squidfile.read(squidfilename)
    kB = 1.38E-23 #SI units
    
    if noisetag == 'sq1_shunt_johnson_rs_closed':
        #Provides johnson noise from the SQ1 shunt, assuming all row select switches are closed (for SSA only noise)
        T_SSA = float(configfile['SSA']['SSA_TEMP_KELVIN'])
        R_SQ1_Shunt = float(configfile['SQ1']['SQ1_SHUNT_OHM'])
        R_SQ1_Para = float(configfile['SQ1']['SQ1_R_par'])
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        L_SSA_in = float(configfile['SSA']['SSA_L_IN_HENRY'])
        L_NbTi_cable = float(configfile['CRYOCABLE']['NBTI_ROUNDTRIP_INDUCTANCE_HENRY'])
        R_tot = R_SQ1_Shunt + R_SQ1_Para
        
        I_Johnson_SQ1_Shunt = np.sqrt(4*kB*T_SSA/R_tot)
        V_Johnson_SQ1_Shunt = I_Johnson_SQ1_Shunt * dV_SSA_dI_SSAin
        
        L_tot = L_SSA_in + L_NbTi_cable
        f_rolloff = R_tot/(2*np.pi*L_tot)
        
        n_pole = 1
        
        return(V_Johnson_SQ1_Shunt, f_rolloff, 0, n_pole)
    if noisetag == 'sq1_shunt_johnson_rs_open':
        #Provides johnson noise from the SQ1 shunt, assuming one row select switches is open (for SQ1 noise)
        T_SSA = float(configfile['SSA']['SSA_TEMP_KELVIN'])
        T_SQ1 = float(configfile['SQ1']['SQ1_TEMP_KELVIN'])
        R_SQ1_Shunt = float(configfile['SQ1']['SQ1_SHUNT_OHM'])
        R_SQ1_Para = float(configfile['SQ1']['SQ1_R_par'])
        R_SQ1_Series = float(squidfile['SQ1']['R_SERIES'])
        R_SQ1_Dyn = float(squidfile['SQ1']['R_DYN_OPERATING_UPSLOPE'])
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        L_SSA_in = float(configfile['SSA']['SSA_L_IN_HENRY'])
        L_NbTi_cable = float(configfile['CRYOCABLE']['NBTI_ROUNDTRIP_INDUCTANCE_HENRY'])
        L_SQ1 = float(squidfile['SQ1']['L_SQ1'])
        R_4K = R_SQ1_Shunt + R_SQ1_Para
        R_100mK = R_SQ1_Series + R_SQ1_Dyn
        
        V_Johnson_SQ1_Stage = np.sqrt(4*kB*T_SSA*R_4K + 4*kB*T_SQ1*R_100mK)
        I_Johnson_SQ1_Stage = V_Johnson_SQ1_Stage/(R_4K + R_100mK)
        V_Johnson_Amp = I_Johnson_SQ1_Stage * dV_SSA_dI_SSAin

        R_tot = R_4K + R_100mK
        L_tot = L_SSA_in + L_NbTi_cable + L_SQ1
        f_rolloff = R_tot/(2*np.pi*L_tot)
        
        n_pole = 1
       
        return(V_Johnson_Amp, f_rolloff, 0, n_pole)
    if noisetag == 'tes_shunt_johnson':
        T_TES = float(configfile['SQ1']['SQ1_TEMP_KELVIN']) #same temp for TES and SQ1
        R_Shunt = float(configfile['TES']['TES_R_SHUNT'])
        R_TES = float(configfile['TES']['TES_R_OP'])
        L_NYQ = float(configfile['TES']['L_NYQUIST'])

        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        dI_SSAin_dI_SQ1in = float(squidfile['SQ1']['dI_SSA_IN_dI_SQ1_IN_upslope'])
        
        I_Johnson_TES_Shunt = np.sqrt(4*kB*T_TES/(R_Shunt+R_TES))
        V_out = I_Johnson_TES_Shunt * dV_SSA_dI_SSAin * dI_SSAin_dI_SQ1in

        f_nyquist = (R_TES + R_Shunt)/(2*np.pi*L_NYQ)
        
        print('tes shunt johnson')
        print(V_out)
        print(f_nyquist)
        n_pole = 1
       
        return(V_out, f_nyquist, 0, n_pole)
    
    if noisetag == 'ssa_bias_cryocable_johnson':
        T_base = float(configfile['SSA']['SSA_TEMP_KELVIN'])
        T_warm = 300
        kint = quad(k_manganin, T_base, T_warm, full_output=1)[0]
        kTint = quad(kT_manganin, T_base, T_warm, full_output=1)[0]
        
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        v_out = np.sqrt(4*kB*R_cable*kTint/(kint))
        
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])
        f_rolloff = 1/(2*np.pi*R_cable*C_cable)

        n_pole = 5
        
        return(v_out, f_rolloff, 0, n_pole)
        
    if noisetag == 'sq1_bias_cryocable_johnson_rs_closed':
        T_base = float(configfile['SSA']['SSA_TEMP_KELVIN'])
        T_warm = 300
        kint = quad(k_manganin, T_base, T_warm, full_output=1)[0]
        kTint = quad(kT_manganin, T_base, T_warm, full_output=1)[0]
        
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        R_SQ1_BIAS_LIST = np.array(json.loads(configfile.get("SQ1","SQ1B_BACKPLANE_RESISTANCE_OHMS")))
        if isinstance(R_SQ1_BIAS_LIST, np.ndarray):
            R_backplane = np.sum(R_SQ1_BIAS_LIST)
        else:
            R_backplane = R_SQ1_BIAS_LIST
        R_board = float(configfile['SQ1']['SQ1B_BC_RESISTANCE_OHM'])
        R_tot = R_cable + R_backplane + R_board
        
        R_SQ1_Shunt = float(configfile['SQ1']['SQ1_SHUNT_OHM'])
        R_SQ1_Para = float(configfile['SQ1']['SQ1_R_par'])
                
        V_Johnson_SQ1_Bias_Cable = np.sqrt(4*kB*R_cable*kTint/(kint))
        V_Johnson_SQ1_Bias_R = np.sqrt(4*kB*T_warm*(R_backplane + R_board))
        I_Johnson_SQ1_Bias_Cable = (V_Johnson_SQ1_Bias_Cable+V_Johnson_SQ1_Bias_R)/R_tot
        I_Johnson_SQ1_In = I_Johnson_SQ1_Bias_Cable/(1 + (R_SQ1_Para/R_SQ1_Shunt))
        V_Johnson_Out = I_Johnson_SQ1_In * dV_SSA_dI_SSAin

        L_SSA_in = float(configfile['SSA']['SSA_L_IN_HENRY'])
        L_NbTi_cable = float(configfile['CRYOCABLE']['NBTI_ROUNDTRIP_INDUCTANCE_HENRY'])
        L_cold = L_SSA_in + L_NbTi_cable
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])  
        f_rolloff = R_cable/(2*np.pi*L_cold)
        f_cable = 1/(2*np.pi*R_cable*C_cable)
        f_rolloff = np.append(f_rolloff, f_cable)   

        n_pole = 1
        
        return(V_Johnson_Out, f_rolloff, 0, n_pole)  
    
    if noisetag == 'sq1_bias_cryocable_johnson_rs_open':
        T_base = float(configfile['SSA']['SSA_TEMP_KELVIN'])
        T_warm = 300
        kint = quad(k_manganin, T_base, T_warm, full_output=1)[0]
        kTint = quad(kT_manganin, T_base, T_warm, full_output=1)[0]
        
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        dV_SSA_dI_SSAin = 1/float(configfile['SSA']['dI_SSA_IN_dV_SSA']) *1e3 #V per A
        R_SQ1_BIAS_LIST = np.array(json.loads(configfile.get("SQ1","SQ1B_BACKPLANE_RESISTANCE_OHMS")))
        if isinstance(R_SQ1_BIAS_LIST, np.ndarray):
            R_backplane = np.sum(R_SQ1_BIAS_LIST)
        else:
            R_backplane = R_SQ1_BIAS_LIST
        R_board = float(configfile['SQ1']['SQ1B_BC_RESISTANCE_OHM'])
        R_tot = R_cable + R_backplane + R_board
        
        R_SQ1_Shunt = float(configfile['SQ1']['SQ1_SHUNT_OHM'])
        R_SQ1_Dyn = float(squidfile['SQ1']['R_DYN_OPERATING_UPSLOPE'])
        R_SQ1_Para = float(configfile['SQ1']['SQ1_R_par'])
        R_SQ1_Series = float(squidfile['SQ1']['R_SERIES'])
        
        R_IN_LEG = R_SQ1_Dyn + R_SQ1_Para + R_SQ1_Series
                
        V_Johnson_SQ1_Bias_Cable = np.sqrt(4*kB*R_cable*kTint/(kint))
        V_Johnson_SQ1_Bias_R = np.sqrt(4*kB*T_warm*(R_backplane + R_board))
        I_Johnson_SQ1_Bias_Cable = (V_Johnson_SQ1_Bias_Cable+V_Johnson_SQ1_Bias_R)/R_tot
        I_Johnson_SQ1_In = I_Johnson_SQ1_Bias_Cable/(1 + (R_IN_LEG/R_SQ1_Shunt))
        V_Johnson_Out = I_Johnson_SQ1_In * dV_SSA_dI_SSAin

        L_SSA_in = float(configfile['SSA']['SSA_L_IN_HENRY'])
        L_NbTi_cable = float(configfile['CRYOCABLE']['NBTI_ROUNDTRIP_INDUCTANCE_HENRY'])
        L_SQ1 = float(squidfile['SQ1']['L_SQ1'])
        L_cold = L_SSA_in + L_NbTi_cable + L_SQ1
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])  
        f_rolloff = R_cable/(2*np.pi*L_cold)
        f_cable = 1/(2*np.pi*R_cable*C_cable)
        f_rolloff = np.append(f_rolloff, f_cable)   
        
        n_pole = 1
        
        return(V_Johnson_Out, f_rolloff, 0, n_pole)  
    
    if noisetag == 'ssa_fb_cryocable_johnson':
        T_base = float(configfile['SSA']['SSA_TEMP_KELVIN'])
        T_warm = 300
        kint = quad(k_manganin, T_base, T_warm, full_output=1)[0]
        kint = quad(k_manganin, T_base, T_warm, full_output=1)[0]
        kTint = quad(kT_manganin, T_base, T_warm, full_output=1)[0]
        
        ADU_conversion = float(configfile['PREAMPADC']['ADU_TO_VOLTS_AT_PREAMP_INPUT'])
        SSAFB_conversion = float(configfile['SSA']['SSAFB_AMP_DAC'])
        dV_SSA_dFB_SSA = float(configfile['SSA']['dV_SSA_ADU_dFB_SSA_ADC'])
        
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        turns_ratio = float(squidfile['SSA']['M_SSA_IN'])/float(squidfile['SSA']['M_SSA_FB'])
        
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        R_backplane = float(configfile['SSA']['SSA_FB_BACKPLANE_RESISTANCE_OHMS'])
        R_board = np.array(json.loads(configfile.get("SSA","SSA_FB_BC_RESISTANCE_OHM")))
        R_tot = R_cable + R_backplane + np.sum(R_board)
        
        V_Johnson_SSA_FB_Cable = np.sqrt(4*kB*R_cable*kTint/(kint))
        V_Johnson_SSA_FB_R = np.sqrt(4*kB*T_warm*(R_backplane + np.sum(R_board)))
        I_Johnson_SSA_FB_Cable = (V_Johnson_SSA_FB_Cable + V_Johnson_SSA_FB_R)/R_tot
        V_Johnson_Out = I_Johnson_SSA_FB_Cable * dV_SSA_dI_SSAin/turns_ratio

        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])        
        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            C_filt = np.array(json.loads(configfile.get("SSA","SSA_FB_LPF_FARAD")))
            f_rolloff = 1/(2*np.pi*R_board*C_filt)
            f_cable = 1/(2*np.pi*R_cable*C_cable)
            f_rolloff = np.append(f_rolloff, f_cable)
        else:
            f_rolloff = 1/(2*np.pi*R_cable*C_cable)        

        #f_rolloff = R_cable/(2*np.pi*L_SSA_FB)
        
        n_pole = 1
        
        return(V_Johnson_Out, f_rolloff, 0, n_pole)
    if noisetag == 'sq1_fb_cryocable_johnson':
        T_base = float(configfile['SSA']['SSA_TEMP_KELVIN'])
        T_warm = 300
        kint = quad(k_manganin, T_base, T_warm, full_output=1)[0]
        kint = quad(k_manganin, T_base, T_warm, full_output=1)[0]
        kTint = quad(kT_manganin, T_base, T_warm, full_output=1)[0]
        
        R_SQ1_FB_BACKPLANE = float(configfile['SQ1']['SQ1_FB_BACKPLANE_RESISTANCE_OHMS'])
        R_Cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        C_Cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])
        
        R_Filt = np.array(json.loads(configfile.get("SQ1","SQ1_FB_LPF_OHM")))
        C_Filt = np.array(json.loads(configfile.get("SQ1","SQ1_FB_LPF_FARAD")))        
        if isinstance(R_Filt, np.ndarray):
            R_Tot = np.sum(R_Filt) + R_SQ1_FB_BACKPLANE + R_Cable
        else:
            R_Tot = R_SQ1_FB_BACKPLANE + R_Filt + R_Cable

        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        dI_SSAin_dI_SQ1in = float(squidfile['SQ1']['dI_SSA_IN_dI_SQ1_IN_upslope'])
        turns_ratio = float(squidfile['SQ1']['M_SQ1_IN'])/float(squidfile['SQ1']['M_SQ1_FB'])
        
        V_Johnson_SQ1_FB_Cable = np.sqrt(4*kB*R_Cable*kTint/(kint))
        V_Johnson_SQ1_FB_R = np.sqrt(4*kB*T_warm*(R_SQ1_FB_BACKPLANE + np.sum(R_Filt)))
        I_Johnson_SQ1_FB_Cable = (V_Johnson_SQ1_FB_Cable + V_Johnson_SQ1_FB_R)/R_Tot
        V_Johnson_Out = I_Johnson_SQ1_FB_Cable * dV_SSA_dI_SSAin * dI_SSAin_dI_SQ1in/turns_ratio

        f_filt = 1/(2*np.pi*R_Filt*C_Filt)
        f_cable = 1/(2*np.pi*R_Cable*C_Cable)
        f_rolloff = np.append(f_filt, f_cable)     
        
        n_pole = 1
        
        return(V_Johnson_Out, f_rolloff, 0, n_pole)

    if noisetag == 'tes_bias_cryocable_johnson':
        T_base = float(configfile['SSA']['SSA_TEMP_KELVIN'])
        T_warm = 300
        kint = quad(k_manganin, T_base, T_warm, full_output=1)[0]
        kTint = quad(kT_manganin, T_base, T_warm, full_output=1)[0]
        
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        dI_SSAin_dI_SQ1in = float(squidfile['SQ1']['dI_SSA_IN_dI_SQ1_IN_upslope'])
        
        TES_BC_RESISTANCE_OHM = np.array(json.loads(configfile.get("TES","TES_BIAS_BC_RESISTANCE_OHM")))
        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            R_Filt = np.array(json.loads(configfile.get("TES","TES_BIAS_LPF_OHM")))
            V_Johnson_TES_Bias_R = np.sqrt(4*kB*T_warm*np.sum(R_Filt))
            R_tot = TES_BIAS_BACKPLANE_RESISTANCE_OHMS + np.sum(R_Filt) + R_cable
        else:
            TES_BIAS_BACKPLANE_RESISTANCE_OHMS = float(configfile['TES']['TES_BIAS_BACKPLANE_RESISTANCE_OHMS'])
            R_tot = TES_BIAS_BACKPLANE_RESISTANCE_OHMS + np.sum(TES_BC_RESISTANCE_OHM) + R_cable
            V_Johnson_TES_Bias_R = np.sqrt(4*kB*T_warm*(TES_BIAS_BACKPLANE_RESISTANCE_OHMS + np.sum(TES_BC_RESISTANCE_OHM)))

        R_TES = float(configfile['TES']['TES_R_OP'])
        R_SHUNT = float(configfile['TES']['TES_R_SHUNT'])
        L_NYQ = float(configfile['TES']['L_NYQUIST'])
                
        V_Johnson_TES_Bias_Cable = np.sqrt(4*kB*R_cable*kTint/(kint))
        I_Johnson_TES_Bias_Cable = (V_Johnson_TES_Bias_Cable+V_Johnson_TES_Bias_R)/R_tot
        I_Johnson_TES_In = I_Johnson_TES_Bias_Cable/(1 + (R_TES/R_SHUNT))
        V_Johnson_Out = I_Johnson_TES_In * dV_SSA_dI_SSAin * dI_SSAin_dI_SQ1in

        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            R_Filt = np.array(json.loads(configfile.get("TES","TES_BIAS_LPF_OHM")))
            C_Filt = np.array(json.loads(configfile.get("TES","TES_BIAS_LPF_FARAD")))        
            f_rolloff = 1/(2*np.pi*R_Filt*C_Filt)
        else:        
            C_filt = np.array(json.loads(configfile.get("TES","TES_BIAS_LPF_FARAD")))
            f_rolloff = 1/(2*np.pi*TES_BC_RESISTANCE_OHM*C_filt)
        f_cable = 1/(2*np.pi*R_cable*C_cable)
        f_rolloff = np.append(f_rolloff, f_cable)   
        f_nyquist = R_cable/(2*np.pi*L_NYQ)
        f_rolloff = np.append(f_rolloff, f_nyquist)
        print(noisetag)
        print('Rolloff freq:')
        print(f_rolloff)
        
        n_pole = 1
                
        return(V_Johnson_Out, f_rolloff, 0, n_pole)  
    
    if noisetag == 'ssa_offset_bias_johnson':
        #We are neglecting the Johnson noise of the SA Bias resistor since it is much larger than the cold resistance (15 kΩ >> 500 Ω) and the effective Johnson noise comes from those two resistors in parallel, so the smaller will thoroughly dominate.
        T_warm = 300
        R_Offset = float(configfile['PREAMPADC']['SSA_OFFSET_GROUND'])         
        R_Gain = float(configfile['PREAMPADC']['SSA_AMP_GAIN_RESISTOR'])
        R_Offset_Bias = float(configfile['PREAMPADC']['SA_OFFSET_BIAS_RESISTOR'])         
        R_Offset_Parallel = R_Offset*R_Offset_Bias/(R_Offset + R_Offset_Bias)
                                              
        V2_Johnson_offset = 4*kB*T_warm*R_Offset_Parallel*(R_Gain/(R_Gain+R_Offset_Parallel))**2
        V2_Johnson_gain = 4*kB*T_warm*R_Gain*(R_Offset_Parallel/(R_Gain+R_Offset_Parallel))**2
        
        V_Johnson = np.sqrt(V2_Johnson_offset + V2_Johnson_gain)

        f_rolloff = 1e15#Essentially infinite, this is a resistor directly to ground.  Will be filtered by room temp
        
        n_pole = 1
        
        return(V_Johnson, f_rolloff, 0, n_pole)  
    if noisetag == 'ssa_amplifier':
        v_noise = float(configfile['PREAMPADC']['SA_AMPLIFIER_NOISE_VOLTS'])
        i_noise = float(configfile['PREAMPADC']['SA_AMPLIFIER_NOISE_AMPS'])
        
        R_bias = float(configfile['SSA']['SSA_BIAS_RC_RESISTANCE_OHMS'])
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        R_SSA = float(configfile['SSA']['SSA_R_DYN'])
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])
        R_total = R_cable + R_SSA
        R_Offset = float(configfile['PREAMPADC']['SSA_OFFSET_GROUND'])         
        R_Gain = float(configfile['PREAMPADC']['SSA_AMP_GAIN_RESISTOR'])
        R_Offset_Bias = float(configfile['PREAMPADC']['SA_OFFSET_BIAS_RESISTOR'])         
        R_Offset_Parallel = R_Offset*R_Offset_Bias/(R_Offset + R_Offset_Bias)
        
        v_current_plus = i_noise * R_total
        v_current_minus = i_noise * R_Offset_Parallel * R_Gain / (R_Offset_Parallel + R_Gain)
        v_tot = np.sqrt(v_noise**2 + v_current_plus**2 + v_current_minus**2)
        f_rolloff = float(configfile['PREAMPADC']['AMPLIFIER_ROLLOFF'])#solely limited by room temp bandwidth
        
        corner_freq = float(configfile['PREAMPADC']['SA_AMPLIFIER_CORNER'])
        
        n_pole = 5
        
        return(v_tot, f_rolloff, corner_freq, n_pole)
    if noisetag == 'ssa_bias_dac':
        R_bias = float(configfile['SSA']['SSA_BIAS_RC_RESISTANCE_OHMS'])
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        R_SSA = float(configfile['SSA']['SSA_R_DYN'])
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])
        R_total = R_cable + R_bias + R_SSA

        v_out = float(configfile['PREAMPADC']['SA_BIAS_AMPLIFIER_NOISE'])*R_cable/R_total     
        
        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            f_rolloff = float(configfile['PREAMPADC']['SA_BIAS_AMPLIFIER_ROLLOFF'])
        else:
            f_rolloff = 1/(2*np.pi*R_cable*C_cable)
        
        corner_freq = float(configfile['PREAMPADC']['SA_BIAS_AMPLIFIER_CORNER'])
       
        n_pole = 1
        
        return(v_out, f_rolloff, corner_freq, n_pole)    
    if noisetag == 'ssa_offset_dac':
        #While it looks like ssa bias enters here as well in SLAC electronics schematic, this is unpopulated
        v_out = float(configfile['PREAMPADC']['SA_OFFSET_AMPLIFIER_NOISE'])*float(configfile['PREAMPADC']['SSA_OFFSET_GROUND'])/(float(configfile['PREAMPADC']['SSA_OFFSET_GROUND']) + float(configfile['PREAMPADC']['SA_OFFSET_BIAS_RESISTOR']))
                
        f_rolloff = float(configfile['PREAMPADC']['SA_OFFSET_AMPLIFIER_ROLLOFF'])
        
        corner_freq = float(configfile['PREAMPADC']['SA_BIAS_AMPLIFIER_CORNER'])
       
        n_pole = 1
        
        return(v_out, f_rolloff, corner_freq, n_pole)
    if noisetag == 'ssa_fb_amp_in':
        SSA_FB_BACKPLANE_RESISTANCE_OHMS = float(configfile['SSA']['SSA_FB_BACKPLANE_RESISTANCE_OHMS'])
        SSA_FB_BC_RESISTANCE_OHM = np.array(json.loads(configfile.get("SSA","SSA_FB_BC_RESISTANCE_OHM")))
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])
        
        R_total = SSA_FB_BACKPLANE_RESISTANCE_OHMS + np.sum(SSA_FB_BC_RESISTANCE_OHM) + R_cable
        
        I_SSA_FB = float(configfile['PREAMPADC']['SSA_FB_AMPLIFIER_NOISE'])/R_total
        
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        turns_ratio = float(squidfile['SSA']['M_SSA_IN'])/float(squidfile['SSA']['M_SSA_FB'])
        
        v_out = I_SSA_FB * dV_SSA_dI_SSAin/turns_ratio

        
        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            C_filt = np.array(json.loads(configfile.get("SSA","SSA_FB_LPF_FARAD")))
            f_rolloff = 1/(2*np.pi*SSA_FB_BC_RESISTANCE_OHM*C_filt)
            f_cable = 1/(2*np.pi*R_cable*C_cable)
            f_rolloff = np.append(f_rolloff, f_cable)
        else:
            f_rolloff = 1/(2*np.pi*R_cable*C_cable)        
        
        corner_freq = float(configfile['PREAMPADC']['SSA_FB_AMPLIFIER_CORNER'])

        n_pole = 1
        
        return(v_out, f_rolloff, corner_freq, n_pole)
    if noisetag == 'sq1_bias_amp_in_rs_closed':  
        R_SQ1_BIAS_LIST = np.array(json.loads(configfile.get("SQ1","SQ1B_BACKPLANE_RESISTANCE_OHMS")))
        if isinstance(R_SQ1_BIAS_LIST, np.ndarray):
            R_SQ1_BACKPLANE_BIAS = np.sum(R_SQ1_BIAS_LIST)
        else:
            R_SQ1_BACKPLANE_BIAS = R_SQ1_BIAS_LIST
        R_SQ1_RC_BIAS = float(configfile['SQ1']['SQ1B_BC_RESISTANCE_OHM'])
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])
        
        R_bias = R_SQ1_BACKPLANE_BIAS + R_SQ1_RC_BIAS + R_cable
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])

        R_SQ1_Shunt = float(configfile['SQ1']['SQ1_SHUNT_OHM'])
        R_SQ1_Para = float(configfile['SQ1']['SQ1_R_par'])
        
        I_SSA_IN = float(configfile['PREAMPADC']['SQ1_BIAS_AMPLIFIER_NOISE'])/(R_bias * (1 + (R_SQ1_Para/R_SQ1_Shunt)))
        
        v_out = I_SSA_IN * dV_SSA_dI_SSAin
        
        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            C_filt = np.array(json.loads(configfile.get("SQ1","SQ1B_LPF_FARAD")))
            f_rolloff = 1/(2*np.pi*R_SQ1_BIAS_LIST*C_filt)
            f_cable = 1/(2*np.pi*R_cable*C_cable)
            f_rolloff = np.append(f_rolloff, f_cable)
        else:
            f_rolloff = 1/(2*np.pi*R_cable*C_cable)
        L_SSA_in = float(configfile['SSA']['SSA_L_IN_HENRY'])
        L_NbTi_cable = float(configfile['CRYOCABLE']['NBTI_ROUNDTRIP_INDUCTANCE_HENRY'])
        L_cold = L_SSA_in + L_NbTi_cable
        f_inductor = R_cable/(2*np.pi*L_cold)
        f_rolloff = np.append(f_rolloff, f_inductor)
       
        corner_freq = float(configfile['PREAMPADC']['SQ1_BIAS_AMPLIFIER_CORNER'])
        
        n_pole = 1

        return(v_out, f_rolloff, corner_freq, n_pole)        
    if noisetag == 'sq1_bias_amp_in_rs_open':  
        R_SQ1_BIAS_LIST = np.array(json.loads(configfile.get("SQ1","SQ1B_BACKPLANE_RESISTANCE_OHMS")))
        if isinstance(R_SQ1_BIAS_LIST, np.ndarray):
            R_SQ1_BACKPLANE_BIAS = np.sum(R_SQ1_BIAS_LIST)
        else:
            R_SQ1_BACKPLANE_BIAS = R_SQ1_BIAS_LIST
            
        R_SQ1_RC_BIAS = float(configfile['SQ1']['SQ1B_BC_RESISTANCE_OHM'])        
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])

        R_bias = R_SQ1_BACKPLANE_BIAS + R_SQ1_RC_BIAS + R_cable
            #Dynamic resistance, shunt resistance, parasitic resistance are all 3 orders of magnitude smaller than the backplane resistors, so not going to trouble with the voltage divider.
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        
        R_SQ1_Shunt = float(configfile['SQ1']['SQ1_SHUNT_OHM'])
        R_SQ1_Dyn = float(squidfile['SQ1']['R_DYN_OPERATING_UPSLOPE'])
        R_SQ1_Para = float(configfile['SQ1']['SQ1_R_par'])
        R_SQ1_Series = float(squidfile['SQ1']['R_SERIES'])
        
        R_IN_LEG = R_SQ1_Dyn + R_SQ1_Para + R_SQ1_Series
        I_SSA_IN = float(configfile['PREAMPADC']['SQ1_BIAS_AMPLIFIER_NOISE'])/(R_bias * (1 + (R_IN_LEG/R_SQ1_Shunt)))
        
        v_out = I_SSA_IN * dV_SSA_dI_SSAin
        
        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            C_filt = np.array(json.loads(configfile.get("SQ1","SQ1B_LPF_FARAD")))
            f_rolloff = 1/(2*np.pi*R_SQ1_BIAS_LIST*C_filt)
            f_cable = 1/(2*np.pi*R_cable*C_cable)
            f_rolloff = np.append(f_rolloff, f_cable)
        else:
            f_rolloff = 1/(2*np.pi*R_cable*C_cable) 
        L_SSA_in = float(configfile['SSA']['SSA_L_IN_HENRY'])
        L_NbTi_cable = float(configfile['CRYOCABLE']['NBTI_ROUNDTRIP_INDUCTANCE_HENRY'])
        L_SQ1 = float(squidfile['SQ1']['L_SQ1'])
        L_cold = L_SSA_in + L_NbTi_cable + L_SQ1
        f_inductor = R_cable/(2*np.pi*L_cold)
        f_rolloff = np.append(f_rolloff, f_inductor)
            
        corner_freq = float(configfile['PREAMPADC']['SQ1_BIAS_AMPLIFIER_CORNER'])
        
        n_pole = 1

        return(v_out, f_rolloff, corner_freq, n_pole)        
    
    if noisetag == 'sq1_fb_amp_in':
        turns_ratio = float(squidfile['SQ1']['M_SQ1_IN'])/float(squidfile['SQ1']['M_SQ1_FB'])
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        dI_SSAin_dI_SQ1in = float(squidfile['SQ1']['dI_SSA_IN_dI_SQ1_IN_upslope'])

        R_SQ1_FB_BACKPLANE = float(configfile['SQ1']['SQ1_FB_BACKPLANE_RESISTANCE_OHMS'])
        R_Cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        C_Cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])
        
        R_Filt = np.array(json.loads(configfile.get("SQ1","SQ1_FB_LPF_OHM")))
        C_Filt = np.array(json.loads(configfile.get("SQ1","SQ1_FB_LPF_FARAD")))        
        if isinstance(R_Filt, np.ndarray):
            R_Tot = np.sum(R_Filt) + R_SQ1_FB_BACKPLANE + R_Cable
        else:
            R_Tot = R_SQ1_FB_BACKPLANE + R_Filt + R_Cable


        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            I_SQ1FB = float(configfile['PREAMPADC']['SQ1_FB_AMPLIFIER_NOISE'])/R_Tot
        else:
            R_33 = float(configfile['SQ1']['SQ1_FB_RC_R33_OHM'])        
            I_SQ1FB = float(configfile['PREAMPADC']['SQ1_FB_AMPLIFIER_NOISE_AMPS'])*R_33/(R_33 + R_Tot)
        
        v_out = I_SQ1FB * dI_SSAin_dI_SQ1in * dV_SSA_dI_SSAin / turns_ratio

        f_filt = 1/(2*np.pi*R_Filt*C_Filt)
        f_cable = 1/(2*np.pi*R_Cable*C_Cable)
        f_rolloff = np.append(f_filt, f_cable)
        
        corner_freq = float(configfile['PREAMPADC']['SQ1_FB_AMPLIFIER_NOISE_CORNER'])
        
        n_pole = 1

        return(v_out, f_rolloff, corner_freq, n_pole)  
    if noisetag == 'tes_bias_amp_in':
        TES_BC_RESISTANCE_OHM = np.array(json.loads(configfile.get("TES","TES_BIAS_BC_RESISTANCE_OHM")))
        R_cable = float(squidfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
        C_cable = float(configfile['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_CAPACITANCE_FARAD'])
        
        R_TES = float(configfile['TES']['TES_R_OP'])
        R_SHUNT = float(configfile['TES']['TES_R_SHUNT'])
        L_NYQ = float(configfile['TES']['L_NYQUIST'])
        
        dV_SSA_dI_SSAin = float(squidfile['SSA']['dV_SSA_dI_SSA_IN_UPSLOPE'])
        dI_SSAin_dI_SQ1in = float(squidfile['SQ1']['dI_SSA_IN_dI_SQ1_IN_upslope'])
        
        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            I_TES_BIAS = 0.5*float(configfile['PREAMPADC']['TES_BIAS_AMPLIFIER_NOISE'])/TES_BC_RESISTANCE_OHM
        else:
            TES_BIAS_BACKPLANE_RESISTANCE_OHMS = float(configfile['TES']['TES_BIAS_BACKPLANE_RESISTANCE_OHMS'])
            R_total = TES_BIAS_BACKPLANE_RESISTANCE_OHMS + np.sum(TES_BC_RESISTANCE_OHM) + R_cable
            I_TES_BIAS = float(configfile['PREAMPADC']['TES_BIAS_AMPLIFIER_NOISE_AMPS'])/R_total
        I_TES = I_TES_BIAS/(1 + (R_TES/R_SHUNT))
                
        v_out = I_TES * dV_SSA_dI_SSAin * dI_SSAin_dI_SQ1in
        
        if configfile['PREAMPADC']['CIRCUIT_TYPE']=='S4':
            R_Filt = np.array(json.loads(configfile.get("TES","TES_BIAS_LPF_OHM")))
            C_Filt = np.array(json.loads(configfile.get("TES","TES_BIAS_LPF_FARAD")))        
            f_filt = 1/(2*np.pi*R_Filt*C_Filt)
            amp_rolloff = float(configfile['PREAMPADC']['TES_BIAS_AMPLIFIER_NOISE_ROLLOFF'])
            f_rolloff = np.append(f_filt, amp_rolloff)
        else:        
            C_filt = np.array(json.loads(configfile.get("TES","TES_BIAS_LPF_FARAD")))
            f_rolloff = 1/(2*np.pi*TES_BC_RESISTANCE_OHM*C_filt)
        f_cable = 1/(2*np.pi*R_cable*C_cable)
        f_rolloff = np.append(f_rolloff, f_cable)        
        f_nyquist = (R_TES + R_SHUNT)/(2*np.pi*L_NYQ)
        f_rolloff = np.append(f_rolloff, f_nyquist)
        
        corner_freq = float(configfile['PREAMPADC']['TES_BIAS_AMPLIFIER_NOISE_CORNER'])

        n_pole = 1

        return(v_out, f_rolloff, corner_freq, n_pole)  
    else:
        print('%s is not a valid tag' %noisetag)
            
def full_system_noise(noise_list, sample_tag, configfilename, squidfilename, figure_title, plot_tag):
    configfile = configparser.ConfigParser()
    configfile.read(configfilename)
    
    if sample_tag == 'full bandwidth':
        adc_freq = float(configfile['PREAMPADC']['ADC_FREQ'])
        f_grid = np.logspace(1, np.log10(adc_freq/2), 32768)
    elif sample_tag == 'multiplexing':
        num_array_visits = int(1e4)
        num_downsamples = int(configfile['PREAMPADC']['NUM_SAMPLES'])
        adc_freq = float(configfile['PREAMPADC']['ADC_FREQ'])
        row_len = int(configfile['PREAMPADC']['ROW_LEN'])
        num_rows = int(configfile['PREAMPADC']['NUM_ROWS'])
        sampling_freq = adc_freq/(row_len*num_rows)
        f_grid = np.linspace(0.1, sampling_freq/2, 32768)

    v_total = np.zeros_like(f_grid)
    f_amp_rolloff = float(configfile['PREAMPADC']['AMPLIFIER_ROLLOFF'])
    f_amp_poles = float(configfile['PREAMPADC']['AMPLIFIER_POLES'])
    
    for noise_source in noise_list:
        noise_level, noise_rolloff, corner_freq, n_pole = noise_output(noise_source, configfilename, squidfilename)
        if sample_tag == 'full bandwidth':
            v_grid = noise_level * np.ones(np.shape(f_grid))
            if isinstance(noise_rolloff, np.ndarray):
                for f in noise_rolloff:
                    v_grid = v_grid*Butterworth_Transfer_Function(f_grid, f, 1)
                v_grid = v_grid*Butterworth_Transfer_Function(f_grid, f_amp_rolloff, f_amp_poles)
            else:
                v_grid = noise_level*Butterworth_Transfer_Function(f_grid, noise_rolloff, n_pole)*Butterworth_Transfer_Function(f_grid, f_amp_rolloff, f_amp_poles)
          
            v_total = np.vstack((v_total, v_grid))
            v_total[0, :] = np.sqrt(np.square(v_total[0, :]) + np.square(v_grid))
        elif sample_tag == 'multiplexing':
            if noise_rolloff < f_grid[-1]:
                #If the noise rolls off within the multiplexed band, we'll trust that little is aliased and save the computation
                v_grid = noise_level*Butterworth_Transfer_Function(f_grid, noise_rolloff, n_pole)*Butterworth_Transfer_Function(f_grid, f_amp_rolloff, f_amp_poles)
                v_total = np.vstack((v_total, v_grid))
                v_total[0, :] = np.sqrt(np.square(v_total[0, :]) + np.square(v_grid))
            else:
                v_aliased, v_noise_1f, f_downsampled, v_spectrum = aliased_noise(num_array_visits, num_downsamples, adc_freq, row_len, num_rows, noise_rolloff, n_pole, f_amp_rolloff, f_amp_poles, False, noise_level, corner_freq, 32)
                v_grid = v_noise_1f/np.sqrt(f_grid/0.1) + v_aliased
                v_total = np.vstack((v_total, v_grid))
                v_total[0, :] = np.sqrt(np.square(v_total[0, :]) + np.square(v_grid))
        else:
            print('Check your sampling tag')
    
    if plot_tag:
        plt.figure()
        plt.loglog(f_grid, v_total[0, :], label = 'Total Noise')
        for i, noise_source in enumerate(noise_list):
            plt.loglog(f_grid, v_total[i+1, :], label=noise_source)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Voltage Noise [V/rt(Hz)]')
        plt.title(figure_title)
        plt.grid(True, which = 'both')
        plt.legend(loc='best')
    
        plt.show()
    return f_grid, v_total