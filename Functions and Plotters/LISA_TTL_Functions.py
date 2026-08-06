#Trying to set up a file to put functions built during the TTL initial phase


def sangria_source(filename, catpath, index):
    import h5py
    import numpy as np
    f = h5py.File(filename, 'r')
    data_raw = np.array(f[catpath])
    names = np.array(data_raw.dtype.names)

    data = data_raw[index]
    data_list = list(data[0])
    data_arr = np.array(data_list)

    pMBHB_snr = {k: v for k, v in zip(names, data_arr)}
    return(pMBHB_snr)

def compute_sangria_snr(filename, catpath):
    #imports just in case
    import numpy as np
    import pandas as pd
    import h5py
    from ldc.waveform.lisabeta import FastBHB
    from ldc.lisa.noise import AnalyticNoise
    import lisabeta.lisa.lisa as lisa

    source_snrs = []

    #set up stuff for source
    dt = 0.25

    waveform_params_smbh = {
        "minf": 1e-5,
        "maxf": 1,
        "t0": 0.0,
        "timetomerger_max": 1.0,
        "tmax": 1.0,
        "TDI": "TDIAET",
        "LISAconst": "Proposal",
        "responseapprox": "full",
        "frozenLISA": False,
        "TDIrescaled": False
        }
    
    #import data
    f = h5py.File(filename, 'r')
    data_raw = np.array(f[catpath])
    names = np.array(data_raw.dtype.names)

    for i in range(0, len(data_raw)):
        data = data_raw[i]
        data_list = list(data[0])
        data_arr = np.array(data_list)

        pMBHB_snr = {k: v for k, v in zip(names, data_arr)}

        #set up source
        t_max = pMBHB_snr["CoalescenceTime"]+1000
        df = 1/t_max

        #Put everything in terms of lisabeta
        FBH = FastBHB("MBHB", T=t_max, delta_t=dt, approx="IMRPhenomD")
        pMBHB = FBH.rename_as_lisabeta(pMBHB_snr)

        #generate signal
        tdisignal = lisa.GenerateLISATDISignal_SMBH(pMBHB, **waveform_params_smbh)
        tdi = tdisignal['tdi']
        mbh_lb = tdi[(2,2)]
        #generate frequency grid
        freq = mbh_lb['freq']
        freq = freq[freq>0]
        #generate data
        A_full = np.conjugate((mbh_lb['amp_real_chan1'] + 1.j*mbh_lb['amp_imag_chan1'])*np.exp(1j* mbh_lb['phase']))
        #generate noise PSD
        Nmodel = AnalyticNoise(freq, model="SciRDv1") # or whatever version you prefer. In the previous notebook we used "sangria"
        Npsd = Nmodel.psd(option='A', tdi2=True)

        #compute snr
        snr_a = (4.0*df) * np.sum(np.abs(A_full)**2/Npsd)
        #print("Computed Noiseless SNR is " + str(np.sqrt(float(snr_a))))
        source_snrs.append(np.sqrt(snr_a))
        snrs = {k: v for k, v in zip(range(0,len(source_snrs)), source_snrs)}
    
    return(snrs)

def sky_separation_from_chains(chain_filepath1, chain_filepath2):
    import pandas as pd
    import numpy as np
    import astropy
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    names = ['beta', 'lambda', 'chi1', 'chi2', 'm1', 'm2', 'Deltat', 'phi', 'dist', 'psi', 'inc']
    #first we need to convert the chain for lambda and beta into its value
    beta1_file = pd.read_csv(chain_filepath1)
    data1 = pd.DataFrame(beta1_file, columns = names)
    beta1 = np.average(data1['beta'])
    lambda1 = np.average(data1['lambda'])

    beta2_file = pd.read_csv(chain_filepath2)
    data2 = pd.DataFrame(beta2_file, columns = names)
    beta2 = np.average(data2['beta'])
    lambda2 = np.average(data2['lambda'])

    print(beta1, lambda1)
    print(beta2, lambda2)

    noiseless_loc = SkyCoord(l=lambda1*u.deg, b=-beta1*u.deg, frame='galactic')
    noisy_loc = SkyCoord(l=lambda2*u.deg, b=-beta2*u.deg, frame='galactic')
    # Convert to ICRS (RA/Dec)
    noiseless_loc_icrs = noiseless_loc.icrs
    noisy_loc_icrs = noisy_loc.icrs
    print(noiseless_loc_icrs)
    print(noisy_loc_icrs)

    separation = noiseless_loc_icrs.separation(noisy_loc_icrs)
    print(separation, separation.degree)
    return(separation, separation.degree)