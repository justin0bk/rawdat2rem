#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:31:56 2017

Note:
Matplotlib constantly changes how to plot figures interactively.
In case, if no figures are not plotted, there might be three options to try out:
(1) First, start python interpreter using "ipython --matplotlib"
(2) Replace plt.plot(block=False) with plt.plot()
(3) Or, try to turn on interactive mode before any plotting (plt.show()) happens, 
    using plt.ion()

@author: Franz
"""
import scipy.signal
import numpy as np
import scipy.io as so
import os.path
import re
import matplotlib.pylab as plt
import h5py
import matplotlib.patches as patches
import pdb


class Mouse :    
    def __init__(self, idf, list=None, typ='') :
        self.recordings = []
        self.recordings.append(list)
        self.typ = typ
        self.idf = idf
    
    def add(self, rec) :
        self.recordings.append(rec)

    def __len__(self) :
        return len(self.recordings)

    def __repr__(self) :
        return ", ".join(self.recordings)



### FILE PROCESSING OF RECORDING DATA #################################################
def load_stateidx(ppath, name, ann_name=''):
    """ load the sleep state file of recording (folder) $ppath/$name
    @Return:
        M,K         sequence of sleep states, sequence of 
                    0'1 and 1's indicating non- and annotated states
    """   
    
    if ann_name == '':
        ann_name = name
        
    file = os.path.join(ppath, name, 'remidx_' + ann_name + '.txt')
    
    f = open(file, 'r')    
    lines = f.readlines()
    f.close()
    
    n = 0
    for l in lines:
        if re.match('\d', l):
            n = n+1
            
    M = np.zeros(n)
    K = np.zeros(n)
    
    i = 0
    for l in lines :
        
        if re.search('^\s+$', l) :
            continue
        if re.search('\s*#', l) :
            continue
        
        if re.match('\d+\s+\d+', l) :
            a = re.split('\s+', l)
            M[i] = int(a[0])
            K[i] = int(a[1])
            i = i+1
            
    return M,K



def load_recordings(ppath, rec_file) :
    """
    load_recordings(ppath, rec_file)
    
    load recording listing with syntax:
    [E|C] \s+ recording_name
    
    #COMMENT
    
    @RETURN:
        (list of controls, lis of experiments)
    """
    exp_list = []
    ctr_list = []    
    
    file = os.path.join(ppath, rec_file)
    f = open(file, 'r')    
    lines = f.readlines()
    f.close()

    for l in lines :
        if re.search('^\s+$', l) :
            continue
        if re.search('^\s*#', l) :
            continue
        
        a = re.split('\s+', l)
        
        if re.search('E', a[0]) :
            exp_list.append(a[1])
            
        if re.search('C', a[0]) :
            ctr_list.append(a[1])
            
    return (ctr_list, exp_list)




def load_dose_recordings(ppath, rec_file):
    """
    load recording list with following syntax:
    A line is either control or experiments; Control recordings look like:

    C \s recording_name

    Experimental recordings also come with an additional dose parameter 
    (allowing for comparison of multiple doses with controls)
    
    E \s recording_name \s dose_1
    E \s recording_name \s dose_2
    """
    
    file = os.path.join(ppath, rec_file)
    f = open(file, 'r')    
    lines = f.readlines()
    f.close()

    # first get all potential doses
    doses = {}
    ctr_list = []
    for l in lines :
        if re.search('^\s+$', l):
            continue
        if re.search('^\s*#', l):
            continue        
        a = re.split('\s+', l)
        
        if re.search('E', a[0]):
            if doses.has_key(a[2]):
                doses[a[2]].append(a[1])
            else:
                doses[a[2]] = [a[1]]

        if re.search('C', a[0]):
            ctr_list.append(a[1])

            
    return (ctr_list, doses)
    


def get_snr(ppath, name) :
    """
    read and return SR from file $ppath/$name/info.txt 
    """
    fid = open(os.path.join(ppath, name, 'info.txt'), 'r')    
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + 'SR' + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))            
    return float(values[0])



def get_infoparam(ifile, field) :
    """
    NOTE: field is a single string
    and the function does not check for the type
    of the values for field.
    In fact, it just returns the string following field
    """
    fid = open(ifile, 'r')    
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + field + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))
            
    return values
    



def laser_start_end(laser, SR=1525.88, intval=5):
    """laser_start_end(ppath, name) ...
    print start and end index of laser stimulation trains: For example,
    if you was stimulated for 2min every 20 min with 20 Hz, return the
    start and end index of the each 2min stimulation period (train)

    returns the tuple (istart, iend), both indices are inclusive,
    i.e. part of the sequence
    @Param:
    laser    -    laser, vector of 0s and 1s
    intval   -    minimum time separation [s] between two laser trains
    @Return:
    (istart, iend) - tuple of two arrays with laser start and end indices
    """
    idx = np.where(laser > 0.5)[0]
    if len(idx) == 0 :
        #return (None, None)
        return ([], [])
    
    idx2 = np.nonzero(np.diff(idx)*(1./SR) > intval)[0]
    istart = np.hstack([idx[0], idx[idx2+1]])
    iend   = np.hstack([idx[idx2], idx[-1]])    

    return (istart, iend)



def load_laser(ppath, name):
    """
    load laser from recording ppath/name ...
    @RETURN: 
    @laser, vector of 0's and 1's 
    """ 
    # laser might be .mat or h5py file
    # perhaps we could find a better way of testing that
    file = os.path.join(ppath, name, 'laser_'+name+'.mat')
    try:
        laser = np.array(h5py.File(file,'r').get('laser'))
    except:
        laser = so.loadmat(file)['laser']
    return np.squeeze(laser)



def laser_protocol(ppath, name):
    """
    What was the stimulation frequency and the inter-stimulation interval for recording
    $ppath/$name?
    
    @Return:
        avg. inter-stimulation interval, frequency
    """
    laser = load_laser(ppath, name)
    SR = get_snr(ppath, name)
    
    # first get inter-stimulation interval
    (istart, iend) = laser_start_end(laser, SR)
    intv = np.diff(np.array(istart/float(SR)))
    d = intv/60.0
    print "The laser was turned on in average every %.2f min," % (np.mean(d))
    print "with a min. interval of %.2f min and max. interval of %.2f min." % (np.min(d), np.max(d))    
    print "Laser stimulation lasted for %f s." % (np.mean(np.array(iend/float(SR)-istart/float(SR)).mean()))    

    # for each laser stimulation interval, check laser stimulation frequency
    dt = 1/float(SR)
    freq = []
    laser_up = []
    laser_down = []
    for (i,j) in zip(istart, iend):
        part = laser[i:j+1]
        (a,b) = laser_start_end(part, SR, 0.005)
        
        dur = (j-i+1)*dt
        freq.append(len(a) / dur)
        up_dur = (b-a+1)*dt*1000
        down_dur = (a[1:]-b[0:-1]-1)*dt*1000
        laser_up.append(np.mean(up_dur))
        laser_down.append(down_dur)
        
    
    print os.linesep + "Laser stimulation freq. was %.2f Hz," % np.mean(np.array(freq))
    print "with laser up and down duration of %.2f and %.2f ms." % (np.mean(np.array(laser_up)), np.mean(np.array(laser_down)))
        
    return np.mean(d), np.mean(np.array(freq))


def swap_eeg(ppath, rec, ch='EEG'):
    """
    swap EEG and EEG2
    """
    if ch == 'EEG':
        name = 'EEG'
    else:
        name = ch
    
    EEG = so.loadmat(os.path.join(ppath, rec, name+'.mat'))[name]
    EEG2 = so.loadmat(os.path.join(ppath, rec, name+'2.mat'))[name + '2']
        
    tmp = EEG
    EEG = EEG2
    EEG2 = tmp
    
    file_eeg1 = os.path.join(ppath, rec, '%s.mat' % name)
    file_eeg2 = os.path.join(ppath, rec, '%s2.mat' % name)
    so.savemat(file_eeg1, {name : EEG})        
    so.savemat(file_eeg2, {name+'2' : EEG2})



def video_pulse_detection(ppath, rec, SR=1000, iv = 0.01):
    """
    return index of each video frame onset
    ppath/rec  -  recording
    
    @Optional
    SR     -      sampling rate of EEG(!) recording
    iv     -      minimum time inverval (in seconds) between two frames
    
    @Return
    index of each video frame onset
    """
    
    V = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'videotime_' + rec + '.mat'))['laser'])
    TS = np.arange(0, len(V))
    # indices where there's a jump in the signal
    t = TS[np.where(V<0.5)];
    if len(t) == 0:
        idx = []
        return idx
    
    # time points where the interval between jumps is longer than iv
    t2  = np.where(np.diff(t)*(1.0/SR)>=iv)[0]
    idx = np.concatenate(([t[0]],t[t2+1]))
    return idx



# SIGNAL PROCESSING ###########################################################
def my_lpfilter(x, w0, N=4):
    """
    create a lowpass Butterworth filter with a cutoff of w0 * the Nyquist rate. 
    The nice thing about this filter is that is has zero-phase distortion. 
    A conventional lowpass filter would introduce a phase lag.
    
    w0   -    filter cutoff; value between 0 and 1, where 1 corresponds to nyquist frequency.
              So if you want a filter with cutoff at x Hz, the corresponding w0 value is given by
                                w0 = 2 * x / sampling_rate
    N    -    order of filter 
    @Return:
        low-pass filtered signal
        
    See also my hp_filter, or my_bpfilter
    """    
    from scipy import signal
    
    b,a = signal.butter(N, w0)
    y = signal.filtfilt(b,a, x)
    
    return y


def my_hpfilter(x, w0, N=4):
    """
    create an N-th order highpass Butterworth filter with cutoff frequency w0 * sampling_rate/2
    """
    from scipy import signal
    # use scipy.signal.firwin to generate filter
    #taps = signal.firwin(numtaps, w0, pass_zero=False)
    #y = signal.lfilter(taps, 1.0, x)

    b,a = signal.butter(N, w0, 'high')
    y = signal.filtfilt(b,a, x)
        
    return y


def my_bpfilter(x, w0, w1, N=4):
    """
    create N-th order bandpass Butterworth filter with corner frequencies 
    w0*sampling_rate/2 and w1*sampling_rate/2
    """
    #from scipy import signal
    #taps = signal.firwin(numtaps, w0, pass_zero=False)
    #y = signal.lfilter(taps, 1.0, x)
    #return y
    from scipy import signal
    b,a = signal.butter(N, [w0, w1], 'bandpass')
    y = signal.filtfilt(b,a, x)
        
    return y
    


def downsample_vec(x, nbin):
    '''
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive \
    bin by their mean \
    @RETURN: the downsampled vector 
    '''
    n_down = int(np.floor(len(x) / nbin))
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))

    # 0 1 2 | 3 4 5 | 6 7 8 
    for i in range(nbin) :
        idx = range(i, int(n_down*nbin), int(nbin))
        x_down += x[idx]

    return x_down / nbin



def smooth_data(x, sig):
    """
    y = smooth_data(x, sig)
    smooth data vector @x with gaussian kernal
    with standard deviation $sig
    """
    sig = float(sig)
    if sig == 0.0:
        return x
        
    # gaussian:
    gauss = lambda (x, sig) : (1/(sig*np.sqrt(2.*np.pi)))*np.exp(-(x*x)/(2.*sig*sig))

    p = 1000000000
    L = 10.
    while (p > p):
        L = L+10
        p = gauss((L, sig))

    F = map(lambda (x): gauss((x, sig)), np.arange(-L, L+1.))
    F = F / np.sum(F)
    
    return scipy.signal.fftconvolve(x, F, 'same')



def power_spectrum(data, length, dt):
    """
    scipy's implementation of Welch's method using hanning window to estimate
    the power spectrum
    @Parameters
        data    -   time series; float vector!
        length  -   length of hanning window, even integer!
    
    @Return:
        powerspectrum, frequencies
    """
    f, pxx = scipy.signal.welch(data, fs=1/dt, window='hanning', nperseg=int(length), noverlap=int(length/2))
    return pxx, f



def spectral_density(data, length, nfft, dt):
    """
    calculate the spectrogram for the time series given by data with time resolution dt
    The powerspectrum for each window of length $length is computed using 
    Welch's method.
    data    -     time series
    length  -     window length of data used to calculate powerspectrum. 
                  Note that the time resolution of the spectrogram is length/2
    nfft    -     size of the window used to calculate the powerspectrum. 
                  determines the frequency resolution.
    @RETURN:
        Powspectrum, frequencies, time-axis
    0 - 5
     2.5 - 7.5
         5 -  10  
    """  
    n = len(data)
    k = int(np.ceil((1.0*n)/length))
    data = np.concatenate((data, np.zeros((length*k-n,))))
    fdt = length*dt/2; # time step for spectrogram
    t = np.arange(0, fdt*(2*k-2)+fdt/2.0, fdt);

    #the power spectrum is calculated for k-1 time points
    j=0;
    # frequency axis of spectrogram
    f = np.linspace(0, 1, int(np.ceil(nfft/2.0))+1) * (0.5/dt)
    Pow = np.zeros((len(f), k*2-1));
    print k
    for i in range(0, k-2+1):
        w1=data[(length*i):(i+1)*length]
        w2=data[length*i+length/2:(i+1)*length+length/2]
        Pow[:,j]   = power_spectrum(w1, nfft, dt)[0]
        Pow[:,j+1] = power_spectrum(w2, nfft, dt)[0]

        j = j+2

    (Pow[:,j],f) = power_spectrum(data[length*(k-1):k*length], nfft, dt)
    
    return Pow, f, t



def calculate_spectrum(ppath, name, fres=0.5):
    """
    calculate powerspectrum used for sleep stage detection.
    Function assumes that data vectors EEG.mat and EMG.mat exist in recording
    folder ppath/name; these are used to calculate the powerspectrum
    fres   -   resolution of frequency axis
    
    all data saved in "true" mat files
    """
    
    SR = get_snr(ppath, name)
    swin = round(SR)*5
    fft_win = round(swin/5)
    if (fres == 1.0) or (fres == 1):
        fft_win = int(fft_win)
    elif fres == 0.5:
        fft_win = 2*int(fft_win)
    else:
        print "Resolution %f not allowed; please use either 1 or 0.5" % fres
    
    (peeg2, pemg2) = (False, False)
    
    # Calculate EEG spectrogram
    EEG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EEG.mat'))['EEG'])
    Pxx, f, t = spectral_density(EEG, int(swin), int(fft_win), 1/SR)
    if os.path.isfile(os.path.join(ppath, name, 'EEG2.mat')):
        peeg2 = True
        EEG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EEG2.mat'))['EEG2'])
        Pxx2, f, t = spectral_density(EEG, int(swin), int(fft_win), 1/SR)        
    #save the stuff to a .mat file
    spfile = os.path.join(ppath, name, 'sp_' + name + '.mat')
    if peeg2 == True:
        so.savemat(spfile, {'SP':Pxx, 'SP2':Pxx2, 'freq':f, 'dt':t[1]-t[0],'t':t})
    else:
        so.savemat(spfile, {'SP':Pxx, 'freq':f, 'dt':t[1]-t[0],'t':t})


    # Calculate EMG spectrogram
    EMG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EMG.mat'))['EMG'])
    Qxx, f, t = spectral_density(EMG, int(swin), int(fft_win), 1/SR)
    if os.path.isfile(os.path.join(ppath, name, 'EMG2.mat')):
        pemg2 = True
        EMG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EMG2.mat'))['EMG2'])
        Qxx2, f, t = spectral_density(EMG, int(swin), int(fft_win), 1/SR)
    # save the stuff to .mat file
    spfile = os.path.join(ppath, name, 'msp_' + name + '.mat')
    if pemg2 == True:
        so.savemat(spfile, {'mSP':Qxx, 'mSP2':Qxx2, 'freq':f, 'dt':t[1]-t[0],'t':t})
    else:
        so.savemat(spfile, {'mSP':Qxx, 'freq':f, 'dt':t[1]-t[0],'t':t})
    
    return Pxx, Qxx, f, t



def recursive_spectrogram(ppath, name, sf=0.3, alpha=0.3, pplot=True):
    """
    calculate EEG/EMG spectrogram in a way that can be implemented by a closed-loop system.
    The spectrogram is temporally filtered using a recursive implementation of a lowpass filter
    @Parameters:
        ppath/name   -    mouse EEG recording
        sf           -    smoothing factor along frequency axis
        alpha        -    temporal lowpass filter time constant
        pplot        -    if pplot==True, plot figure 
    @Return:
        SE, SM       -    EEG, EMG spectrogram

    """
    
    EEG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EEG.mat'))['EEG'])
    EMG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EMG.mat'))['EMG'])
    len_eeg = len(EEG)
    fdt = 2.5
    SR = get_snr(ppath, name)
    # we calculate the powerspectrum for 5s windows
    swin = int(np.round(SR) * 5.0)
    # but we sample new data each 2.5 s
    swinh = int(swin/2.0)
    fft_win = int(swin / 5.0)
    # number of 2.5s long samples
    spoints = int(np.floor(len_eeg / swinh))

    SE = np.zeros((fft_win/2+1, spoints))
    SM = np.zeros((fft_win/2+1, spoints))
    print "Starting calculating spectrogram for %s..." % name
    for i in range(2, spoints):
        # we take the last two swinh windows (the new 2.5 s long sample and the one from
        # the last iteration)
        x = EEG[(i-2)*swinh:i*swinh]
        
        [p, f] = power_spectrum(x.astype('float'), fft_win, 1.0/SR)
        p = smooth_data(p, sf)
        # recursive low pass filtering of spectrogram:
        # the current state is an estimate of the current sample and the previous state
        SE[:,i] = alpha*p + (1-alpha) * SE[:,i-1]

        # and the same of EMG        
        x = EMG[(i-2)*swinh:i*swinh]
        [p, f] = power_spectrum(x.astype('float'), fft_win, 1.0/SR)
        p = smooth_data(p, sf)
        SM[:,i] = alpha*p + (1-alpha) * SM[:,i-1]

    if pplot==True:        
        # plot EEG spectrogram
        t = np.arange(0, SM.shape[1])*fdt
        plt.figure()
        ax1 = plt.subplot(211)
        im = np.where((f>=0) & (f<=30))[0]
        med = np.median(SE.max(axis=0))
        ax1.imshow(np.flipud(SE[im,:]), vmin=0, vmax=med*2)
        plt.xticks(())
        ix = range(0, 30, 10)
        fi = f[im][::-1]
        plt.yticks(ix, map(int, fi[ix]))
        box_off(ax1)
        plt.axis('tight')
        plt.ylabel('Freq (Hz)')
        
        # plot EMG amplitude
        ax2 = plt.subplot(212)
        im = np.where((f>=10) & (f<100))[0]
        df = np.mean(np.diff(f))
        # amplitude is the square root of the integral
        ax2.plot(t, np.sqrt(SM[im,:].sum(axis=0)*df)/1000.0)
        plt.xlim((0, t[-1]))
        plt.ylabel('EMG Ampl (mV)')
        plt.xlabel('Time (s)')
        box_off(ax2)
        plt.show()

    return SE, SM, f



def recursive_sleepstate_rem(ppath, recordings, sf=0.3, alpha=0.3, past_mu=0.2, std_thdelta = 1.5, past_len=120, sdt=2.5, psave=False, xemg=False):
    """
    predict a REM period only based on EEG/EMG history; the same algorithm is also used for 
    closed-loop REM sleep manipulation.
    The algorithm uses for REM sleep detection a threshold on delta power, EMG power, and theta/delta power.
    For theta/delta I use two thresholds: A hard (larger) threshold and a soft (lower) threshold. Initially,
    theta/delta has to cross the hard threshold to initiate a REM period. Then, as long as,
    theta/delta is above the soft threshold (and EMG power stays low) REM sleep continues.
    
    @Parameters:
        ppath        base folder with recordings
        recordings   list of recordings
        sf           smoothing factor for each spectrogram
        past_mu      percentage (0 .. 1) of brain states that are allowed to have EMG power larger than threshold
                     during the last $past_len seconds
        past_len     window to calculate $past_mu
        std_thdelta  the hard theta/delta threshold is given by, mean(theta/delta) + $std_thdelta * std(theta/delta)  
        sdt          time bin for brain sttate, typically 2.5s
        psave        if True, save threshold parameters to file.
    
    """        
    idf = re.split('_', recordings[0])[0]
    past_len = int(np.round(past_len/sdt))
    
    # calculate spectrogram
    (SE, SM) = ([],[])
    for rec in recordings:
        A,B, freq = recursive_spectrogram(ppath, rec, sf=sf, alpha=alpha)       
        SE.append(A)
        SM.append(B)
        
    # fuse lists SE and SM
    SE = np.squeeze(reduce(lambda x,y: np.concatenate((x,y)), SE))
    if not(xemg):
        SM = np.squeeze(reduce(lambda x,y: np.concatenate((x,y)), SM))
    else:
        SM = SE

    # EEG, EMG bands
    ntbins = SE.shape[1]
    r_delta = [0.5, 4]
    r_theta = [5,12]
    # EMG band
    r_mu = [300, 500]

    i_delta = np.where((freq >= r_delta[0]) & (freq <= r_delta[1]))[0]
    i_theta = np.where((freq >= r_theta[0]) & (freq <= r_theta[1]))[0]
    i_mu    = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    
    pow_delta = np.sum(SE[i_delta,:], axis=0)
    pow_theta = np.sum(SE[i_theta,:], axis=0)
    pow_mu = np.sum(SM[i_mu,:], axis=0)
    # theta/delta
    th_delta = np.divide(pow_theta, pow_delta)
    thr_th_delta1 = np.nanmean(th_delta) + std_thdelta*np.nanstd(th_delta)
    thr_th_delta2 = np.nanmean(th_delta) +  0.0*np.nanstd(th_delta)
    thr_delta = pow_delta.mean()
    thr_mu = pow_mu.mean() + 0.5*np.nanstd(pow_mu)


    ### The actual algorithm for REM detection
    rem_idx = np.zeros((ntbins,))
    prem = 0 # whether or not we are in REM
    for i in range(ntbins):
        
        if prem == 0 and pow_delta[i] < thr_delta and pow_mu[i] < thr_mu:
            ### could be REM
            
            if th_delta[i] > thr_th_delta1:
                ### we are potentially entering REM
                if (i - past_len) >= 0:
                    sstart = i-past_len
                else:
                    sstart = 0
                # count the percentage of brainstate bins with elevated EMG power
                c_mu = np.sum( np.where(pow_mu[sstart:i]>thr_mu)[0] ) / past_len
               
                if c_mu < past_mu:
                    ### we are in REM
                    prem = 1  # turn laser on
                    rem_idx[i] = 1
        
        # We are currently in REM; do we stay there?
        if prem == 1:
            ### REM continues, if theta/delta is larger than soft threshold and if there's
            ### no EMG activation
            if (th_delta[i] > thr_th_delta2) and (pow_mu[i] < thr_mu):
                rem_idx[i] = 1
            else:
                prem = 0 #turn laser off

    # Determine which channel is EEG, EMG
    ch_alloc = get_infoparam(os.path.join(ppath, recordings[0], 'info.txt'),  'ch_alloc')[0]
      
    # plot the whole stuff:
    # (1) spectrogram
    # (2) EMG Power
    # (3) Delta
    # (4) TH_Delta
    plt.figure()
    t = np.arange(0, sdt*(ntbins-1)+sdt/2.0, sdt)
    ax1 = plt.subplot(411)
    im = np.where((freq>=0) & (freq<=30))[0]
    med = np.median(SE.max(axis=0))
    ax1.imshow(np.flipud(SE[im,:]), vmin=0, vmax=med*2)
    plt.yticks(range(0, 31, 10), range(30, -1, -10))
    plt.ylabel('Freq. (Hz)')
    plt.axis('tight')

    ax2 = plt.subplot(412)
    ax2.plot(t, pow_mu, color='black')
    ax2.plot(t, np.ones((len(t),))*thr_mu, color='red')
    plt.ylabel('EMG Pow.')
    plt.xlim((t[0], t[-1]))

    ax3 = plt.subplot(413, sharex=ax2)
    ax3.plot(t, pow_delta, color='black')
    ax3.plot(t, np.ones((len(t),))*thr_delta, color='red')
    plt.ylabel('Delta Pow.')
    plt.xlim((t[0], t[-1]))

    ax4 = plt.subplot(414, sharex=ax3)
    ax4.plot(t, th_delta, color='black')
    ax4.plot(t, np.ones((len(t),))*thr_th_delta1, color='red')
    ax4.plot(t, np.ones((len(t),))*thr_th_delta2, color='pink')
    ax4.plot(t, rem_idx*thr_th_delta1, color='blue')
    plt.ylabel('Theta/Delta')
    plt.xlabel('Time (s)')
    plt.xlim((t[0], t[-1]))
    plt.show(block=False)
    
    # write config file
    if psave:
        cfile = os.path.join(ppath, idf + '_rem.txt')
        fid = open(cfile, 'w')
        fid.write(('IDF: %s'+os.linesep) % idf)
        fid.write(('ch_alloc: %s'+os.linesep) % ch_alloc)
        fid.write(('THR_DELTA: %.2f'+os.linesep) % thr_delta)
        fid.write(('THR_MU: %.2f'+os.linesep) % thr_mu)
        fid.write(('THR_TH_DELTA: %.2f %.2f'+os.linesep) % (thr_th_delta1, thr_th_delta2))
        fid.write(('STD_THDELTA: %.2f'+os.linesep) % std_thdelta)
        fid.write(('PAST_MU: %.2f'+os.linesep) % past_mu)
        fid.write(('SF: %.2f'+os.linesep) % sf)
        fid.write(('ALPHA: %.2f'+os.linesep) % alpha)
        if xemg:
            fid.write(('XEMG: %d'+os.linesep), 1)
        else:
            fid.write(('XEMG: %d' + os.linesep), 0)
        fid.close()
        print 'wrote file %s' % cfile


    
def load_sleep_params(path, param_file):
    """
    load parameter file generated by &recursive_sleepstate_rem
    @Return:
        Dictionary: Parameter --> Value
    """    
    fid = open(os.path.join(path, param_file), 'r')
    lines = fid.readlines()
    params = {}
    for line in lines:
        if re.match('^[\S_]+:', line):
            a = re.split('\s+', line)
            key = a[0][:-1]
            params[key] = a[1:-1]
            
    # transform number strings to floats
    for k in params.keys():
        vals = params[k] 
        new_vals = []
        for v in vals:
            if re.match('^[\d\.]+$', v):
                new_vals.append(float(v))
            else:
                new_vals.append(v)
        params[k] = new_vals
                    
    return params
            
        
        

def recursive_sleepstate(ppath, recordings, sf=0.3, alpha=0.3, past_mu=0.2, std_thdelta = 1.5, past_len=120, sdt=2.5, psave=False):
    from sklearn import mixture
    
    idf = re.split('_', recordings[0])[0]
    past_len = int(np.round(past_len/sdt))
    
    # calculate spectrogram
    (SE, SM) = ([],[])
    for rec in recordings:
        A,B, freq = recursive_spectrogram(ppath, rec, sf=sf, alpha=alpha)       
        SE.append(A)
        SM.append(B)
        
    # fuse lists SE and SM
    SE = np.squeeze(reduce(lambda x,y: np.concatenate((x,y)), SE))
    SM = np.squeeze(reduce(lambda x,y: np.concatenate((x,y)), SM))
    
    # EEG, EMG bands        
    ntbins = SE.shape[1]
    r_delta = [0.5, 4]
    r_theta = [5,12]
    # EMG band
    r_mu = [300, 500]

    i_delta = np.where((freq >= r_delta[0]) & (freq <= r_delta[1]))[0]
    i_theta = np.where((freq >= r_theta[0]) & (freq <= r_theta[1]))[0]
    i_mu    = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    
    pow_delta = np.sum(SE[i_delta,:], axis=0)
    pow_theta = np.sum(SE[i_theta,:], axis=0)
    pow_mu = np.sum(SM[i_mu,:], axis=0)
    # theta/delta
    th_delta = np.divide(pow_theta, pow_delta)
    thr_th_delta1 = np.nanmean(th_delta) + std_thdelta*np.nanstd(th_delta)
    thr_th_delta2 = np.nanmean(th_delta) +  0.0*np.nanstd(th_delta)
    thr_delta = pow_delta.mean()
    thr_mu = pow_mu.mean() + 0.5*np.nanstd(pow_mu)

    
    med_delta = np.median(pow_delta)
    pow_delta_fit = pow_delta[np.where(pow_delta<=3*med_delta)]
    
    # fit Gaussian mixture model to delta power
    # see http://www.astroml.org/book_figures/chapter4/fig_GMM_1D.html
    gm = mixture.GMM(n_components=2)
    fit = gm.fit(pow_delta_fit)
    means = np.squeeze(fit.means_)
    means.sort()
    #stds  = fit.covars_
    
    x = np.arange(0, med_delta*3, 100)
    plt.figure()
    plt.hist(pow_delta_fit, 100, normed=True, histtype='stepfilled', alpha=0.4)
    
    logprob, responsibilities = fit.eval(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.plot(x, pdf, '-k')
    plt.plot(x, pdf_individual, '--k')
    plt.xlim((0, med_delta*3))
    plt.show()

    plt.ylabel('p(x)')
    plt.xlabel('x = Delta power')
    
    # get point where curves cut each other
    idx = np.where((x>= means[0]) & (x<= means[1]))[0]
    imin = np.argmin(pdf[idx])
    xcut = x[idx[0]+imin]
    
    plt.plot(xcut, pdf[idx[0]+imin], 'ro')
    
    ilow = np.argmin(np.abs(x-means[0]))
    plt.plot(x[ilow], pdf[ilow], 'bo')

    ihigh = np.argmin(np.abs(x-means[1]))
    plt.plot(x[ihigh], pdf[ihigh], 'go')


    pdb.set_trace()

    thr_delta1 = x[ihigh]
    thr_delta2 = x[ilow]

    # NREM yes or no according to thresholds
    pnrem = 0
    # grace_count = 120
    # NREM stays on after thresholds are NOT fulfilled to avoid interruptions by microarousals

    # nrem_delay: NREM only starts with some delay


    for i in range(ntbins):
        if pnrem == 0:
            ### Entering NREM
            if (pow_delta[i] > thr_delta1 and pow_mu[i] < thr_mu  and th_delta[i] < thr_th_delta1):
                ### NOT-NREM -> NREM
                pnrem = 1
                nrem_idx[i] = 0
                delay_count = delay_count-1
                
                grace_count = grace_period
            else:
                ### NOT-NREM -> NOT-NREM
                if grace_count > 0:
                    grace_count = grace_count - 1
                    nrem_idx[i] = 1
                else:
                    nrem_idx[i] = 0
        else:
            ### pnrem == 1
            if (pow_delta(i) > thr_delta2 and pow_mu(i) < thr_mu and th_delta(i) < thr_th_delta1):
                if delay_count > 0:
                    delay_count = delay_count - 1
                    nrem_idx[i] = 0
                else :
                    nrem_idx[i] = 1
            else:
                ### Exit NREM -> NOT-NREM
                delay_count = nrem_delay
                pnrem = 0
                
                if grace_count > 0:
                    grace_count = grace_count - 1
                    nrem_idx[i] = 1
        



def iirnotch(data):
    pass
### END SIGNAL PROCESSING #####################################################

### FUNCTIONS USED BY SLEEP_STATE #####################################################
def get_sequences(idx, ibreak=1) :  
    """
    get_sequences(idx, ibreak=1)
    idx     -    np.vector of indices
    @RETURN:
    seq     -    list of np.vectors
    """
    diff = idx[1:] - idx[0:-1]
    breaks = np.nonzero(diff>ibreak)[0]
    breaks = np.append(breaks, len(idx)-1)
    
    seq = []    
    iold = 0
    for i in breaks:
        r = range(iold, i+1)
        seq.append(idx[r])
        iold = i+1
        
    return seq


def threshold_crossing(data, th, ilen, ibreak, m):
    """
    seq = threshold_crossing(data, th, ilen, ibreak, m)
    """

    if m>=0:
        idx = np.where(data>=th)[0]
    else:
        idx = np.where(data<=th)[0]


    # gather sequences
    j = 0;
    seq = []
    while (j <= len(idx)-1):
        s = [idx[j]]
        
        for k in range(j+1,len(idx)):
            if (idx[k] - idx[k-1]-1) <= ibreak:
                # add j to sequence
                s.append(idx[k])
            else:
                break

        if (s[-1] - s[0]+1) >= ilen and not(s[0] in [i[1] for i in seq]):
            seq.append((s[0], s[-1]))
        
        if j == len(idx)-1:
            break           
        j=k
        
    return seq



def closest_precessor(seq, i):
    """
    find the preceding element in seq which is closest to i
    helper function for sleep_state
    """
    tmp = seq-i;
    d = np.where(tmp<0)[0]
    
    if len(d)>0:
        id = seq[d[-1]];
    else:
        id = 0;
    
    return id


def write_remidx(M, K, ppath, name, mode=1) :
    """
    rewrite_remidx(idx, states, ppath, name)
    replace the indices idx in the remidx file of recording name
    with the assignment given in states
    """
   
    if mode == 0 :
        outfile = os.path.join(ppath, name, 'remidx_' + name + '.txt')
    else :
        outfile = os.path.join(ppath, name, 'remidx_' + name + '_corr.txt')

    f = open(outfile, 'w')
    s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M[0,:],K)]
    f.writelines(s)
    f.close()

#######################################################################################


### MANIPULATING FIGURES ##############################################################
def set_fontsize(fs):
    import matplotlib
    matplotlib.rcParams.update({'font.size': fs})



def save_figure(fig_file):
    import matplotlib
    # alternative way of setting nice fonts:
    
    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.rcParams['ps.fonttype'] = 42
    #matplotlib.pylab.savefig(fig_file, dpi=300)

    matplotlib.rcParams['text.usetex'] = False 
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.pylab.savefig(fig_file, bbox_inches="tight", dpi=300)
    matplotlib.rcParams['text.usetex'] = False   


def box_off(ax):
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  

#######################################################################################



def sleep_state(ppath, name, th_delta_std=1, mu_std=0, sf=1, sf_delta=3, pwrite=0, pplot=1, pemg=1):
    """
    sleep_state(ppath, name, th_delta_std=1, gamma_std=1, sf=1, sf2=1, pwrite=0, pplot=1):
    New: use also sigma band: that's very helpful to classify pre-REM periods
    as NREM; xfshotherwise they tend to be classified as qwake.
    Gamma peaks nicely pick up microarousals. My strategy is the following:
    I smooth delta band a lot to avoid strong fragmentation of sleep; but to 
    still pick up microarousals I use the gamma power.
    
    spectrogram data has to be calculated before by calculate_spectrum
    """
    
    PRE_WAKE_REM = 30.0
    
    # Minimum Duration and Break in 
    # high theta/delta, high emg, and high delta sequences
    # Synatax: duration(i,0) is the minimum duration of sequency i
    # duration(i,2) is maximal break duration allowed in a sequence
    # of state i
    duration = np.zeros((5,2))
    # high theta/delta
    duration[0,:] = [5,15]
    # high emg
    duration[1,:] = [0, 5]
    # high delta
    duration[2,:] = [10, 10]
    # high sigma
    duration[3,:] = [10, 10]
    # gamma
    duration[4,:] = [0, 5]
    
    # Frequency Bands/Ranges for delta, theta, and, gamma
    r_delta = [0.5, 4]
    r_sigma = [12, 20]
    r_theta = [5,12]
    # EMG band
    r_mu = [50, 500]
    if pemg==0: 
        r_mu = [250, 500]
    # high gamma power
    r_gamma = [100, 150] #100, 150

    #load EEG and EMG spectrum, calculated by calculate_spectrum
    P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'))
    if pemg == 1:
        Q = so.loadmat(os.path.join(ppath, name, 'msp_' + name + '.mat'))
    else:
        #Q = so.loadmat(os.path.join(ppath, name, 'sp_' + name + '.mat'))
        pass
    
    SPEEG = np.squeeze(P['SP'])
    if pemg == 1:
        SPEMG = np.squeeze(Q['mSP'])
    else:
        SPEMG = np.squeeze(P['SP'])
        
    freq  = np.squeeze(P['freq'])
    t     = np.squeeze(P['t'])
    dt    = float(np.squeeze(P['dt']))
    N     = len(t)
    duration = np.divide(duration,dt)
    
    # get indices for frequency bands
    i_delta = np.where((freq >= r_delta[0]) & (freq <= r_delta[1]))[0]
    i_theta = np.where((freq >= r_theta[0]) & (freq <= r_theta[1]))[0]
    i_mu    = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    i_sigma = np.where((freq >= r_sigma[0]) & (freq <= r_sigma[1]))[0]
    i_gamma = np.where((freq >= r_gamma[0]) & (freq <= r_gamma[1]))[0]

    p_delta = smooth_data( SPEEG[i_delta,:].mean(axis=0), sf_delta );
    p_theta = smooth_data( SPEEG[i_theta,:].mean(axis=0), 0 );    
    # now filtering for EMG to pick up microarousals
    p_mu    = smooth_data( SPEMG[i_mu,:].mean(axis=0), sf );
    p_sigma = smooth_data( SPEEG[i_sigma,:].mean(axis=0), sf );
    p_gamma = smooth_data( SPEEG[i_gamma,:].mean(axis=0), 0 );

    th_delta = np.divide(p_theta, p_delta)
    #th_delta = smooth_data(th_delta, 2);

    seq = {}
    seq['high_theta'] = threshold_crossing(th_delta, np.nanmean(th_delta)+th_delta_std*np.nanstd(th_delta), \
       duration[0,1], duration[0,1], 1)
    seq['high_emg'] = threshold_crossing(p_mu, np.nanmean(p_mu)+mu_std*np.nanstd(p_mu), \
       duration[1,0], duration[1,1], 1)
    seq['high_delta'] = threshold_crossing(p_delta, np.nanmean(p_delta), duration[2,0], duration[2,1], 1)
    seq['high_sigma'] = threshold_crossing(p_sigma, np.nanmean(p_sigma), duration[3,0], duration[3,1], 1)
    seq['high_gamma'] = threshold_crossing(p_gamma, np.nanmean(p_gamma), duration[4,0], duration[4,1], 1)
    
    #pdb.set_trace()
    # Sleep-State Rules
    idx = {}
    for k in seq.keys():
        tmp = [range(i,j+1) for (i,j) in seq[k]]
        # now idea why this works to flatten a list
        # idx[k] = sum(tmp, [])
        # alternative that I understand:
        print k
        idx[k] = np.array(reduce(lambda x,y: x+y, tmp))

    idx['low_emg']    = np.setdiff1d(np.arange(0,N), np.array(idx['high_emg']))
    idx['low_delta'] = np.setdiff1d(np.arange(0,N), np.array(idx['high_delta']))
    idx['low_theta'] = np.setdiff1d(np.arange(0,N), np.array(idx['high_theta']))
        
        
    #REM Sleep: thdel up, emg down, delta down    
    a = np.intersect1d(idx['high_theta'], idx['low_delta'])
    # non high_emg phases
    b = np.setdiff1d(a, idx['high_emg'])
    rem = get_sequences(b, duration[0,1])
    rem_idx = reduce(lambda x,y: np.concatenate((x,y)), rem)


    # SWS Sleep
    # delta high, no theta, no emg
    a = np.setdiff1d(idx['high_delta'], idx['high_emg']) # no emg activation
    b = np.setdiff1d(a, idx['high_theta'])               # no theta;
    sws = get_sequences(b)
    sws_idx = reduce(lambda x,y: np.concatenate((x,y)), sws)
    #print a

    # Wake
    # low delta + high emg and not rem
    a = np.unique(np.union1d(idx['low_delta'], idx['high_emg']))
    b = np.setdiff1d(a, rem_idx)
    wake = get_sequences(b)
    wake_idx = reduce(lambda x,y: np.concatenate((x,y)), wake)

    # sequences with low delta, high sigma and low emg are NREM
    a = np.intersect1d(np.intersect1d(idx['high_sigma'], idx['low_delta']), idx['low_emg'])
    a = np.setdiff1d(a, rem_idx)
    sws_idx = np.unique(np.union1d(a, sws_idx))
    wake_idx = np.setdiff1d(wake_idx, a)

    #NREM sequences with high gamma are wake
    a = np.intersect1d(sws_idx, idx['high_gamma'])    
    sws_idx = np.setdiff1d(sws_idx, a)
    wake_idx = np.unique(np.union1d(wake_idx,a))

    # Wake and Theta
    wake_motion_idx = np.intersect1d(wake_idx, idx['high_theta'])

    # Wake w/o Theta
    wake_nomotion_idx = np.setdiff1d(wake_idx, idx['low_theta'])

    # Are there overlapping sequences?
    a = np.intersect1d(np.intersect1d(rem_idx, wake_idx), sws_idx);

    # Are there undefined sequences?
    undef_idx = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.arange(0,N), rem_idx), wake_idx), sws_idx);
    
    # Wake wins over SWS
    sws_idx = np.setdiff1d(sws_idx, wake_idx);

    
    # Special rules
    # if there's a REM sequence directly following a short wake sequence (PRE_WAKE_REM),
    # this wake sequence goes to SWS
    # NREM to REM transitions are sometimes mistaken as quite wake periods
    for rem_seq in rem:
        if len(rem_seq) > 0:
            irem_start = rem_seq[0]
            # is there wake in the preceding bin?
            if irem_start-1 in wake_idx:
                # get the closest sws bin in the preceding history
                isws_end = closest_precessor(sws_idx, irem_start);
                print "%d %d" % (isws_end, irem_start)
                if (irem_start - isws_end)*dt < PRE_WAKE_REM:
                    new_rem = np.arange(isws_end+1,irem_start)
                    rem_idx = np.union1d(rem_idx, new_rem);
                    wake_idx = np.setdiff1d(wake_idx, new_rem);


    # two different representations for the results:
    S = {}
    S['rem']    = rem_idx
    S['nrem']   = sws_idx
    S['wake']   = wake_idx
    S['awake']  = wake_motion_idx
    S['qwake']  = wake_nomotion_idx
    
    M = np.zeros((N,))
    M[rem_idx]           = 1
    M[wake_idx]          = 2
    M[sws_idx]           = 3
    M[undef_idx]         = 0
    
    # write sleep annotation to file
    if pwrite==1:
        outfile = os.path.join(ppath, name, 'remidx_' + name + '.txt')
        print "writing annotation to %s" % outfile
        f = open(outfile, 'w')
        s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M,np.zeros((N,)))]
        f.writelines(s)
        f.close()
        
    # nice plotting
    if pplot==1:        
        plt.figure(figsize=(18,9))
        axes1=plt.axes([0.1, 0.9, 0.8, 0.05])
        A = np.zeros((1,len(M)))
        A[0,:] = M
        cmap = plt.cm.jet
        my_map = cmap.from_list('ha', [[0,0,0], [0,1,1],[1,0,1], [0.8, 0.8, 0.8]], 4)
        tmp = axes1.imshow(A, vmin=0, vmax=3)
        tmp.set_cmap(my_map)
        axes1.axis('tight')
        tmp.axes.get_xaxis().set_visible(False)
        tmp.axes.get_yaxis().set_visible(False)
        box_off(axes1)
        
        # show spectrogram
        axes2=plt.axes([0.1, 0.75, 0.8, 0.1], sharex=axes1)
        #axes2.pcolor(t,freq[0:30],SPEEG[0:30,:])
        ifreq = np.where(freq <= 30)[0]
        axes2.imshow(np.flipud(SPEEG[ifreq,:]))
        axes2.axis('tight')        
        plt.ylabel('Freq (Hz)')
        box_off(axes2)

        # show
        axes3=plt.axes([0.1, 0.6, 0.8, 0.1])
        axes3.plot(t,p_delta, color='gray')
        plt.ylabel('Delta (a.u.)')
        plt.xlim((t[0], t[-1]))
        seq = get_sequences(S['nrem'])
        #for s in seq:
        #    plt.plot(t[s],p_delta[s], color='red')
        s = idx['high_delta']
        seq = get_sequences(s)
        for s in seq:
            plt.plot(t[s],p_delta[s], color='red')
        box_off(axes3)

        axes4=plt.axes([0.1, 0.45, 0.8, 0.1], sharex=axes3)
        axes4.plot(t,p_sigma, color='gray')
        plt.ylabel('Sigma (a.u.)')
        plt.xlim((t[0], t[-1]))
        s = idx['high_sigma']
        seq = get_sequences(s)
        for s in seq:
            plt.plot(t[s],p_sigma[s], color='red')
        box_off(axes4)

        axes5=plt.axes([0.1, 0.31, 0.8, 0.1], sharex=axes4)
        axes5.plot(t,th_delta, color='gray')
        plt.ylabel('Th/Delta (a.u.)')
        plt.xlim((t[0], t[-1]))
        s = idx['high_theta']
        seq = get_sequences(s)
        for s in seq:            
            plt.plot(t[s],th_delta[s], color='red')
        box_off(axes5)

        axes6=plt.axes([0.1, 0.17, 0.8, 0.1], sharex=axes5)
        axes6.plot(t,p_gamma, color='gray')
        plt.ylabel('Gamma (a.u.)')
        plt.xlim((t[0], t[-1]))
        s = idx['high_gamma']
        seq = get_sequences(s)
        for s in seq:
            plt.plot(t[s],p_gamma[s], color='red')
        box_off(axes6)

        axes7=plt.axes([0.1, 0.03, 0.8, 0.1], sharex=axes6)
        axes7.plot(t,p_mu, color='gray')        
        plt.xlabel('Time (s)')
        plt.ylabel('EMG (a.u.)')
        plt.xlim((t[0], t[-1]))
        s = idx['high_emg']
        seq = get_sequences(s)
        for s in seq:
            plt.plot(t[s],p_mu[s], color='red')
        box_off(axes7)
        plt.show()

        
        # 2nd figure showing distribution of different bands
        plt.figure(figsize=(20,3))
        axes1 = plt.axes([0.05, 0.1, 0.13, 0.8])
        plt.hist(p_delta, bins=100)
        plt.plot(np.nanmean(p_delta), 10, 'ro')
        plt.title('delta')
        plt.ylabel('# Occurances')
        box_off(axes1)
        
        axes1 = plt.axes([0.25, 0.1, 0.13, 0.8])
        plt.hist(th_delta, bins=100)
        plt.plot(np.nanmean(th_delta)+th_delta_std*np.nanstd(th_delta), 10, 'ro')
        plt.title('theta/delta')
        box_off(axes1)

        axes1 = plt.axes([0.45, 0.1, 0.13, 0.8])
        plt.hist(p_sigma, bins=100)
        plt.plot(np.nanmean(p_sigma), 10, 'ro')
        plt.title('sigma')
        box_off(axes1)
                
        axes1 = plt.axes([0.65, 0.1, 0.13, 0.8])
        plt.hist(p_gamma, bins=100)
        plt.plot(np.nanmean(p_gamma), 10, 'ro')
        plt.title('gamma')
        box_off(axes1)
        
        axes1 = plt.axes([0.85, 0.1, 0.13, 0.8])
        plt.hist(p_mu, bins=100)
        plt.plot(np.nanmean(p_mu)+np.nanstd(p_mu), 10, 'ro')
        plt.title('EMG')
        plt.show(block=False)
        box_off(axes1)
        
        plt.show()
    
    return M,S
    


def laser_triggered_eeg(ppath, name, pre, post, f_max, pnorm=2, pplot=False, psave=False, peeg2=False):
    """
    calculate laser triggered, averaged EEG and EMG spectrum
    ppath   -    base folder containing mouse recordings
    name    -    recording
    pre     -    time before laser
    post    -    time after laser
    f_max   -    calculate/plot frequencies up to frequency f_max
    p_norm  -    normalization: 
                 pnorm = 0, no normalization
                 pnorm = 1, normalize each frequency band by its average power
                 pnorm = 2, normalize each frequency band by the average power 
                            during the preceding baseline period
    pplot   -    plot figure yes=True, no=False
    psave   -    save the figure, yes=True, no = False
    """
    SR = get_snr(ppath, name)
    NBIN = np.round(2.5*SR)
    lsr = load_laser(ppath, name)
    idxs, idxe = laser_start_end(lsr)
    laser_dur = np.mean((idxe-idxs)/SR)
    print('Average laser duration: %f; Number of trials %d' % (laser_dur, len(idxs)))

    # downsample EEG time to spectrogram time    
    idxs = [int(i/NBIN) for i in idxs]
    idxe = [int(i/NBIN) for i in idxe]
    #load EEG and EMG
    if peeg2==False:
        P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'))
    else:
        P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'))
    Q = so.loadmat(os.path.join(ppath, name, 'msp_' + name + '.mat'))

    if peeg2==False:
        SPEEG = np.squeeze(P['SP'])
    else:
        SPEEG = np.squeeze(P['SP2'])
    SPEMG = np.squeeze(Q['mSP'])
    freq  = np.squeeze(P['freq'])
    t     = np.squeeze(P['t'])
    dt    = float(np.squeeze(P['dt']))

    speeg_mean = SPEEG.mean(axis=1)
    spemg_mean = SPEMG.mean(axis=1)

    # Spectrogram for EEG and EMG normalized by average power in each frequency band
    if pnorm == 1:
        SPEEG = np.divide(SPEEG, np.repeat(speeg_mean, len(t)).reshape(len(speeg_mean), len(t)))
        SPEMG = np.divide(SPEMG, np.repeat(spemg_mean, len(t)).reshape(len(spemg_mean), len(t)))
    
    ifreq = np.where(freq<=f_max)[0]
    ipre  = int(np.round(pre/dt))
    ipost = int(np.round(post/dt))


    speeg_parts = []
    spemg_parts = []
    for (i,j) in zip(idxs, idxe):
        if i>=ipre and j+ipost < len(t):
            speeg_parts.append(SPEEG[ifreq,i-ipre:i+ipost+1])
            spemg_parts.append(SPEMG[ifreq,i-ipre:i+ipost+1])
            
    
    EEGLsr = np.array(speeg_parts).mean(axis=0)
    EMGLsr = np.array(spemg_parts).mean(axis=0)
    
    # smooth spectrogram
    nfilt = 3
    filt = np.ones((nfilt,nfilt))
    filt = np.divide(filt, filt.sum())
    EEGLsr = scipy.signal.convolve2d(EEGLsr, filt, boundary='symm', mode='same')
    EMGLsr = scipy.signal.convolve2d(EMGLsr, filt, boundary='symm', mode='same')

    if pnorm == 2:    
        for i in range(EEGLsr.shape[0]):
            EEGLsr[i,:] = np.divide(EEGLsr[i,:], np.sum(np.abs(EEGLsr[i,0:ipre]))/(1.0*ipre))
            EMGLsr[i,:] = np.divide(EMGLsr[i,:], np.sum(np.abs(EMGLsr[i,0:ipre]))/(1.0*ipre))
    
    # get time axis    
    dt = (1.0/SR)*NBIN
    #t = np.arange(-ipre,ipost+1)*dt
    t = np.linspace(-ipre*dt, ipost*dt, ipre+ipost+1)
    f = freq[ifreq]

    
    if pplot==True:
        set_fontsize(14)
        # get rid of boxes around matplotlib plots
        def box_off(ax):
            ax.spines["top"].set_visible(False)    
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()  
            ax.get_yaxis().tick_left() 
                
        plt.ion()
        plt.figure(figsize=(12,10))
        ax = plt.axes([0.1, 0.55, 0.4, 0.4])
        plt.pcolormesh(t,f,EEGLsr, vmin=0, vmax=np.median(EEGLsr)*3)
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        box_off(ax)
        plt.title('EEG')
        cbar = plt.colorbar()
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        
        ax = plt.axes([0.6, 0.55, 0.3, 0.4])
        #ipre = np.where(t<0)[0]
        ilsr = np.where((t>=0) & (t<=120))[0]        
        plt.plot(f,EEGLsr[:,0:ipre].mean(axis=1), color='gray', label='baseline', lw=2)
        plt.plot(f,EEGLsr[:,ilsr].mean(axis=1), color='blue', label='laser', lw=2)
        box_off(ax)
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('Power (uV^2)')
        #plt.legend(loc=0)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        
        ax = plt.axes([0.1, 0.05, 0.4, 0.4])
        plt.pcolormesh(t,f,EMGLsr)
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        box_off(ax)    
        plt.title('EMG')
        cbar = plt.colorbar()
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')

        
        ax = plt.axes([0.6, 0.05, 0.3, 0.4])
        mf = np.where((f>=10) & (f <= 50))[0]
        df = f[1]-f[0]
        # amplitude is square root of (integral over each frequency)
        avg_emg = np.sqrt(EMGLsr[mf,:].sum(axis=0)*df)    
        m = np.max(avg_emg)*1.5
        plt.plot([0,0], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.plot([laser_dur,laser_dur], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.xlim((t[0], t[-1]))
        plt.ylim((0,m))
        plt.plot(t,avg_emg, color='black', lw=2)
        box_off(ax)     
        plt.xlabel('Time (s)')
        plt.ylabel('EMG ampl. (uV)')
        
        plt.show()
        
        if psave==True:
            img_file = os.path.join(ppath, name, 'fig_'+name+'_spec.png')
            save_figure(img_file)
                
    return EEGLsr, EMGLsr, freq[ifreq], t
    


def laser_triggered_eeg_avg(ppath, recordings, pre, post, f_max, laser_dur, pnorm=1, pplot=True, psave=False):
    """
    calculate average spectrogram for all recordings listed in @recordings; for averaging take 
    mouse identity into account
    """
    EEGSpec = {}
    EMGSpec = {}
    mice = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not(idf in mice):
            mice.append(idf)
        EEGSpec[idf] = []
        EMGSpec[idf] = []
    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        EEG, EMG, f, t = laser_triggered_eeg(ppath, rec, pre, post, f_max, pnorm=pnorm, pplot=False, psave=False)
        EEGSpec[idf].append(EEG)
        EMGSpec[idf].append(EMG)
    
    for idf in mice:
        EEGSpec[idf] = np.array(EEGSpec[idf]).mean(axis=0)
        EMGSpec[idf] = np.array(EMGSpec[idf]).mean(axis=0)
        
    EEGLsr = np.array([EEGSpec[k] for k in mice]).mean(axis=0)
    EMGLsr = np.array([EMGSpec[k] for k in mice]).mean(axis=0)
    

    if pplot==True:
        set_fontsize(14)

        plt.ion()
        plt.figure(figsize=(12,10))
        ax = plt.axes([0.1, 0.55, 0.4, 0.4])
        plt.pcolormesh(t,f,EEGLsr)
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        box_off(ax)
        plt.title('EEG')
        cbar = plt.colorbar()
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        
        ax = plt.axes([0.6, 0.55, 0.3, 0.4])
        ipre = np.where(t<0)[0]
        ilsr = np.where((t>=0) & (t<=120))[0]        
        plt.plot(f,EEGLsr[:,ipre].mean(axis=1), color='gray', label='baseline', lw=2)
        plt.plot(f,EEGLsr[:,ilsr].mean(axis=1), color='blue', label='laser', lw=2)
        box_off(ax)
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('Power (uV^2)')
        #plt.legend(loc=0)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        
        ax = plt.axes([0.1, 0.05, 0.4, 0.4])
        plt.pcolormesh(t,f,EMGLsr, cmap='jet')
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        box_off(ax)    
        plt.title('EMG')
        cbar = plt.colorbar()
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')

        
        ax = plt.axes([0.6, 0.05, 0.3, 0.4])
        mf = np.where((f>=10) & (f <= 200))[0]
        df = f[1]-f[0]
        # amplitude is square root of (integral over each frequency)
        avg_emg = np.sqrt(EMGLsr[mf,:].sum(axis=0)*df)    
        m = np.max(avg_emg)*1.5
        plt.plot([0,0], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.plot([laser_dur,laser_dur], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.xlim((t[0], t[-1]))
        plt.ylim((0,m))
        plt.plot(t,avg_emg, color='black', lw=2)
        box_off(ax)     
        plt.xlabel('Time (s)')
        plt.ylabel('EMG ampl. (uV)')
        
        plt.show()
        
        if psave==True:
            img_file = os.path.join(ppath, name, 'fig_'+name+'_spec.png')
            save_figure(img_file)




    
def laser_brainstate(ppath, recordings, pre, post, pplot=True, fig_file=''):
    """
    calculate laser triggered probability of REM, Wake, NREM
    ppath        -    base folder holding all recording
    recordings   -    list of recording
    pre          -    time before laser onset
    post         -    time after laser onset
    @Optional:
    pplot        -    pplot==True: plot figure
    fig_file     -    specify filename including ending, if you wish to save figure
    """

    if type(recordings) != list:
        recordings = [recordings]

    BrainstateDict = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        BrainstateDict[idf] = []
    nmice = len(BrainstateDict.keys())

    for rec in recordings:
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        dt = NBIN * 1/SR

        M = load_stateidx(ppath, rec)[0]
        (idxs, idxe) = laser_start_end(load_laser(ppath, rec))
        idf = re.split('_', rec)[0]

        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        ipre  = int(pre/dt)
        ipost = int(post/dt)

        idxs = [int(i/NBIN) for i in idxs]
        idxe = [int(i/NBIN) for i in idxe]
        laser_dur = np.mean((np.array(idxe) - np.array(idxs))) * dt
        
        for (i,j) in zip(idxs, idxe):
            if i>=ipre and j+ipost<=len(M)-1:
                bs = M[i-ipre:i+ipost+1]                
                BrainstateDict[idf].append(bs) 

    # I assume here that every recording has same dt
    t = np.linspace(-ipre*dt, ipost*dt, ipre+ipost+1)
    BS = np.zeros((nmice, len(t), 3))
    Trials = []
    imouse = 0
    for mouse in BrainstateDict.keys():
        M = np.array(BrainstateDict[mouse])
        Trials.append(M)
        for state in range(1,4):
            C = np.zeros(M.shape)
            C[np.where(M==state)] = 1
            BS[imouse,:,state-1] = C.mean(axis=0)
        imouse += 1
                
    # flatten Trials
    Trials = reduce(lambda x,y: np.concatenate((x,y), axis=0),  Trials)
    
    if pplot == True:
        def box_off(ax):
            ax.spines["top"].set_visible(False)    
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()  
            ax.get_yaxis().tick_left() 
        
        state_label = {0:'REM', 1:'Wake', 2:'NREM'}
        set_fontsize(20)
        plt.figure()
        plt.ion()
        #ax = plt.subplot(111)
        ax = plt.axes([0.1, 0.15, 0.8, 0.7])
        colors = [[0, 1, 1 ],[0.5, 0, 1],[0.6, 0.6, 0.6]];
        for state in range(3):
            plt.plot(t, BS[:,:,state].mean(axis=0), color=colors[state], lw=3, label=state_label[state])
        
        plt.xlim([-pre, post])
        plt.ylim([0,1])
        ax.add_patch(patches.Rectangle((0,0), laser_dur, 1, facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))
        box_off(ax)
        plt.xlabel('Time (s)')
        plt.ylabel('Probability')
        #plt.legend(bbox_to_anchor=(0., 1.02, 0.5, .102), loc=3, ncol=3, borderaxespad=0.)
        plt.draw()
                
        plt.figure(figsize=(7,9))
        plt.ion()
        ax = plt.subplot(111)        
        cmap = plt.cm.jet
        my_map = cmap.from_list('ha', [[0,1,1],[0.5,0,1], [0.6, 0.6, 0.6]], 3)
        x = range(Trials.shape[0])
        plt.pcolormesh(t,np.array(x), np.flipud(Trials), cmap=my_map, vmin=1, vmax=3)
        plt.plot([0,0], [0, len(x)-1], color='white')
        plt.plot([laser_dur,laser_dur], [0, len(x)-1], color='white')
        ax.axis('tight')
        plt.draw()
        plt.xlabel('Time (s)')
        plt.ylabel('Trial No.')
        box_off(ax)
        
        plt.show()
        
        if len(fig_file)>0:
            plt.savefig(os.path.join(ppath, fig_file))
        
    return BS,t




def sleep_stats(ppath, recordings, ma_thr=10.0, tstart=0, tend=-1, pplot=True):
    """
    Calculate average percentage of each brain state,
    average duration and average frequency
    plot histograms for REM, NREM, and Wake durations
    @PARAMETERS:
    ppath      -   base folder
    recordings -   single string specifying recording or list of recordings

    @OPTIONAL:
    ma_thr     -   threshold for wake periods to be considered as microarousals
    tstart     -   only consider recorded data starting from time tstart, default 0s
    tend       -   only consider data recorded up to tend s, default -1, i.e. everything till the end
    pplot      -   generate plot in the end; True or False

    @RETURN:
        ndarray of percentages (# mice x [REM,Wake,NREM])
        ndarray of state durations
        ndarray of transition frequency / hour
    """
    if type(recordings) != list:
        recordings = [recordings]

    Percentage = {}
    Duration = {}
    Frequency = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        Percentage[idf] = {1:[], 2:[], 3:[]}
        Duration[idf] = {1:[], 2:[], 3:[]}
        Frequency[idf] = {1:[], 2:[], 3:[]}
    nmice = len(Frequency.keys())    

    for rec in recordings:
        idf = re.split('_', rec)[0]
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        dt = NBIN * 1/SR

        # load brain state
        M = load_stateidx(ppath, rec)[0]        
        if tend==-1:
            iend = len(M)-1
        else:
            iend = int(np.round((1.0*tstart) / dt))
        istart = int(np.round((1.0*tstart) / dt))

        M[np.where(M==5)] = 2        
        # polish out microarousals
        seq = get_sequences(np.where(M==2)[0])
        for s in seq:
            if len(s)*dt <= ma_thr:
                M[s] = 3
        
        Mcut = M[istart:iend+1]
        nm = len(Mcut)*1.0
        
        # get percentage of each state
        for s in [1,2,3]:
            Percentage[idf][s].append(len(np.where(Mcut==s)[0]) / nm)
            
        # get frequency of each state
        for s in [1,2,3]:
            Frequency[idf][s].append( len(get_sequences(np.where(Mcut==s)[0])) * (3600. / (nm*dt)) )
            
        # get average duration for each state
        for s in [1,2,3]:
            seq = get_sequences(np.where(Mcut==s)[0])
            Duration[idf][s] += [len(i)*dt for i in seq] 
        
    PercMx = np.zeros((nmice,3))
    i=0
    for k in Percentage.keys():
        for s in [1,2,3]:
            PercMx[i,s-1] = np.array(Percentage[k][s]).mean()
        i += 1
    PercMx *= 100
        
    FreqMx = np.zeros((nmice,3))
    i = 0
    for k in Frequency.keys():
        for s in [1,2,3]:
            FreqMx[i,s-1] = np.array(Frequency[k][s]).mean()
        i += 1
    
    DurMx = np.zeros((nmice,3))
    i = 0
    for k in Duration.keys():
        for s in [1,2,3]:
            DurMx[i,s-1] = np.array(Duration[k][s]).mean()
        i += 1
        
    DurHist = {1:[], 2:[], 3:[]}
    for s in [1,2,3]:
        DurHist[s] = np.squeeze(np.array(reduce(lambda x,y: x+y, [Duration[k][s] for k in Duration.keys()])))
        

    if pplot == True:      
        # plot bars summarizing results - Figure 1
        set_fontsize(15)
        plt.figure(figsize=(10, 5))
        ax = plt.axes([0.1, 0.15, 0.2, 0.8])
        plt.bar([1,2,3], PercMx.mean(axis=0), align='center', facecolor='gray')
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'], rotation=60)
        for s in range(1,4):
            plt.plot(np.ones((nmice,))*s, PercMx[:,s-1], 'o', color='black')
        plt.ylabel('Percentage (%)')
        plt.xlim([0.2, 3.8])
        box_off(ax)
            
        ax = plt.axes([0.4, 0.15, 0.2, 0.8])
        plt.bar([1,2,3], DurMx.mean(axis=0), align='center', facecolor='gray')
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'], rotation=60)
        for s in range(1,4):
            plt.plot(np.ones((nmice,))*s, DurMx[:,s-1], 'o', color='black')
        plt.ylabel('Duration (s)')
        plt.xlim([0.2, 3.8])
        box_off(ax)
            
        ax = plt.axes([0.7, 0.15, 0.2, 0.8])
        plt.bar([1,2,3], FreqMx.mean(axis=0), align='center', facecolor='gray')
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'], rotation=60)
        for s in range(1,4):
            plt.plot(np.ones((nmice,))*s, FreqMx[:,s-1], 'o', color='black')
        plt.ylabel('Frequency (1/h)')
        plt.xlim([0.2, 3.8])
        box_off(ax)
        plt.show(block=False)    

        # plot histograms - Figure 2            
        plt.figure(figsize=(5, 10))
        ax = plt.axes([0.2,0.1, 0.7, 0.2])
        h, edges = np.histogram(DurHist[1], bins=40, range=(0, 300), normed=1)
        binWidth = edges[1] - edges[0]
        plt.bar(edges[0:-1], h*binWidth, width=5)
        plt.xlim((edges[0], edges[-1]))
        plt.xlabel('Duration (s)')
        plt.ylabel('Freq. REM')
        box_off(ax)
        
        ax = plt.axes([0.2,0.4, 0.7, 0.2])
        h, edges = np.histogram(DurHist[2], bins=40, range=(0, 1200), normed=1)
        binWidth = edges[1] - edges[0]
        plt.bar(edges[0:-1], h*binWidth, width=20)
        plt.xlim((edges[0], edges[-1]))
        plt.xlabel('Duration (s)')
        plt.ylabel('Freq. Wake')
        box_off(ax)
        
        ax = plt.axes([0.2,0.7, 0.7, 0.2])
        h, edges = np.histogram(DurHist[3], bins=40, range=(0, 1200), normed=1)
        binWidth = edges[1] - edges[0]
        plt.bar(edges[0:-1], h*binWidth, width=20)
        plt.xlim((edges[0], edges[-1]))
        plt.xlabel('Duration (s)')
        plt.ylabel('Freq. NREM')
        box_off(ax)
        plt.show(block=False)
    
    return PercMx, DurMx, FreqMx
  
    

def sleep_timecourse_list(ppath, recordings, tbin, n, tstart=0, tend=-1, ma_thr=-1, pplot=True):
    """
    simplified version of sleep_timecourse
    plot sleep timecourse for a list of recordings
    The function does not distinguish between control and experimental mice.
    It computes/plots the how the percentage, frequency (1/h) of brain states and duration 
    of brain state episodes evolves over time.
    
    See also sleep_timecourse
    
    @Parameters:
        ppath                       Base folder with recordings
        recordings                  list of recordings as e.g. generated by &load_recordings
        tbin                        duration of single time bin in seconds
        n                           number of time bins
    @Optional:
        tstart                      start time of first bin in seconds
        tend                        end time of last bin; end of recording if tend==-1
        ma_thr                      set microarousals (wake periods <= ma_thr seconds) to NREM
                                    if ma_thr==-1, don't do anything
        pplot                       plot figures summarizing results
    
    @Return:
        TimeMx, DurMx, FreqMx       Dict[state][time_bin x mouse_id]
    """
    
    if type(recordings) != list:
        recordings = [recordings]
    
    Mice = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        Mice[idf] = 1
    Mice = Mice.keys()
    
    TimeCourse = {}
    FreqCourse = {}
    DurCourse = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        # time bin in Fourier time
        dt = NBIN * 1/SR

        M = load_stateidx(ppath, rec)[0]        
        M[np.where(M)==5] = 2
        # polish out microarousals
        if ma_thr>0:
            seq = get_sequences(np.where(M==2)[0])
            for s in seq:
                if len(s)*dt <= ma_thr:
                    M[s] = 3
        
        
        if tend==-1:
            iend = len(M)-1
        else:
            iend = int(np.round((1.0*tend) / dt))
        M = M[0:iend+1]
        istart = int(np.round((1.0*tstart) / dt))
        ibin = int(np.round(tbin / dt))
        
        # how brain state percentage changes over time
        perc_time = []
        for i in range(n):
            M_cut = M[istart+i*ibin:istart+(i+1)*ibin]            
            perc = []
            for s in [1,2,3]:
                perc.append( len(np.where(M_cut==s)[0]) / (1.0*ibin) )
            perc_time.append(perc)
        
        perc_vec = np.zeros((n,3))
        
        for i in range(3):
            perc_vec[:,i] = np.array([v[i] for v in perc_time])
        TimeCourse[rec] = perc_vec


        # how frequency of sleep stage changes over time
        freq_time = []
        for i in range(n):
            M_cut = M[istart+i*ibin:istart+(i+1)*ibin]            
            freq = []
            for s in [1,2,3]:
                tmp = len(get_sequences(np.where(M_cut==s)[0])) * (3600. / (ibin*dt))                
                freq.append(tmp)
            freq_time.append(freq)
        
        freq_vec = np.zeros((n,3))
        
        for i in range(3):
            freq_vec[:,i] = np.array([v[i] for v in freq_time])
        FreqCourse[rec] = freq_vec
        
        # how duration of sleep stage changes over time
        dur_time = []
        for i in range(n):
            M_cut = M[istart+i*ibin:istart+(i+1)*ibin]            
            dur = []
            for s in [1,2,3]:
                tmp = get_sequences(np.where(M_cut==s)[0])
                tmp = np.array([len(j)*dt for j in tmp]).mean()                
                dur.append(tmp)
            dur_time.append(dur)
        
        dur_vec = np.zeros((n,3))
        
        for i in range(3):
            dur_vec[:,i] = np.array([v[i] for v in dur_time])
        DurCourse[rec] = dur_vec


    # collect all recordings belonging to a Control mouse        
    TimeCourseMouse = {}
    DurCourseMouse = {}
    FreqCourseMouse = {}
    # Dict[mouse_id][time_bin x br_state]
    for mouse in Mice:
        TimeCourseMouse[mouse] = []
        DurCourseMouse[mouse] = []
        FreqCourseMouse[mouse] = []

    for rec in recordings:
        idf = re.split('_', rec)[0]
        TimeCourseMouse[idf].append(TimeCourse[rec])
        DurCourseMouse[idf].append(DurCourse[rec])
        FreqCourseMouse[idf].append(FreqCourse[rec])
    
    mx = np.zeros((n, len(Mice)))
    TimeMx = {1:mx, 2:mx.copy(), 3:mx.copy()}
    mx = np.zeros((n, len(Mice)))
    DurMx = {1:mx, 2:mx.copy(), 3:mx.copy()}
    mx = np.zeros((n, len(Mice)))
    FreqMx = {1:mx, 2:mx.copy(), 3:mx.copy()}        
    # Dict[R|W|N][time_bin x mouse_id]
    i = 0
    for k in TimeCourseMouse.keys():
        for s in range(1,4):
            # [time_bin x br_state]
            tmp = np.array(TimeCourseMouse[k]).mean(axis=0)
            TimeMx[s][:,i] = tmp[:,s-1]
            
            tmp = np.array(DurCourseMouse[k]).mean(axis=0)
            DurMx[s][:,i] = tmp[:,s-1]
            
            tmp = np.array(FreqCourseMouse[k]).mean(axis=0)
            FreqMx[s][:,i] = tmp[:,s-1]                                    
        i += 1

    if pplot==True:
        label = {1:'REM', 2:'Wake', 3:'NREM'}

        # plot percentage of brain state as function of time
        plt.figure()        
        for s in range(1,4):
            ax = plt.axes([0.1, (s-1)*0.3+0.1, 0.8, 0.2])
            plt.errorbar(range(n), TimeMx[s].mean(axis=1), yerr = TimeMx[s].std(axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            box_off(ax)
            plt.xlim((-0.5, n-0.5))
            if s==1:
                plt.ylim([0, 0.2])
            else:
                plt.ylim([0, 1.0])
            plt.ylabel(label[s])    
            plt.draw()
        

        # plot duraion as function of time
        plt.figure()
        for s in range(1,4):
            ax = plt.axes([0.1, (s-1)*0.3+0.1, 0.8, 0.2])
            plt.errorbar(range(n), DurMx[s].mean(axis=1), yerr = DurMx[s].std(axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            box_off(ax)
            plt.xlim((-0.5, n-0.5))
            plt.ylabel(label[s])    
            plt.draw()
        

        # plot frequency as function of time
        plt.figure()
        for s in range(1,4):
            ax = plt.axes([0.1, (s-1)*0.3+0.1, 0.8, 0.2])
            plt.errorbar(range(n), FreqMx[s].mean(axis=1), yerr = FreqMx[s].std(axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            box_off(ax)
            plt.xlim((-0.5, n-0.5))
            plt.ylabel(label[s])    
            plt.draw()
        
        plt.show(block=False)
        
    return TimeMx, DurMx, FreqMx



def sleep_timecourse(ppath, trace_file, tbin, n, tstart=0, tend=-1, pplot=True):
    """
    plot how percentage of REM,Wake,NREM changes over time;
    compares control with experimental data; experimental recordings can have different "doses"
    a simpler version is sleep_timecourse_list
    
    @Parameters
    trace_file-  text file, specifies control and experimental recordings
    tbin    -    size of time bin in seconds
    n       -    number of time bins
    @Optional:
    tstart  -    beginning of recording (time <tstart is thrown away)
    tend    -    end of recording (time >tend is thrown away)
    pplot   -    plot figure if True
    
    @Return:
    TimeMxCtr - Dict[R|W|N][time_bin x mouse_id] 
    TimeMxExp - Dict[R|W|N][dose][time_bin x mouse_id]
    """
    (ctr_rec, exp_rec) = load_dose_recordings(ppath, trace_file)
    
    Recordings = []
    Recordings += ctr_rec
    for k in exp_rec.keys():
        Recordings += exp_rec[k]
        
    CMice = {}
    for mouse in ctr_rec:
        idf = re.split('_', mouse)[0]
        CMice[idf] = 1
    CMice = CMice.keys()
    
    EMice = {}
    for d in exp_rec.keys():
        mice = exp_rec[d]
        EMice[d] = {}
        for mouse in mice:
            idf = re.split('_', mouse)[0]
            EMice[d][idf] = 1
        EMice[d] = EMice[d].keys()
        
    TimeCourse = {}
    Mice = {}
    for rec in Recordings:
        idf = re.split('_', rec)[0]
        Mice[idf] = 1
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        # time bin in Fourier time
        dt = NBIN * 1/SR

        M = load_stateidx(ppath, rec)[0]        
        M[np.where(M)==5] = 2
        if tend==-1:
            iend = len(M)-1
        else:
            iend = int(np.round((1.0*tend) / dt))
        M = M[0:iend+1]
        istart = int(np.round((1.0*tstart) / dt))
        ibin = int(np.round(tbin / dt))
        
        perc_time = []
        for i in range(n):
            M_cut = M[istart+i*ibin:istart+(i+1)*ibin]            
            perc = []
            for s in [1,2,3]:
                perc.append( len(np.where(M_cut==s)[0]) / (1.0*ibin) )
            perc_time.append(perc)
        
        perc_vec = np.zeros((n,3))
        
        for i in range(3):
            perc_vec[:,i] = np.array([v[i] for v in perc_time])
        TimeCourse[rec] = perc_vec
        
    # collect all recordings belonging to a Control mouse        
    TimeCourseCtr = {}
    # Dict[mouse_id][time_bin x br_state]
    for mouse in CMice:
        TimeCourseCtr[mouse] = []

    for rec in Recordings:
        idf = re.split('_', rec)[0]
        if rec in ctr_rec:
            TimeCourseCtr[idf].append(TimeCourse[rec])
    
    mx = np.zeros((n, len(CMice)))
    TimeMxCtr = {1:mx, 2:mx.copy(), 3:mx.copy()}
    # Dict[R|W|N][time_bin x mouse_id]
    i = 0
    for k in TimeCourseCtr.keys():
        for s in range(1,4):
            # [time_bin x br_state]
            tmp = np.array(TimeCourseCtr[k]).mean(axis=0)
            TimeMxCtr[s][:,i] = tmp[:,s-1]
        i += 1
                
    # collect all recording belonging to one Exp mouse with a specific dose
    TimeCourseExp = {}
    # Dict[dose][mouse_id][time_bin x br_state]
    for d in EMice.keys():
        TimeCourseExp[d]={}
        for mouse in EMice[d]:
            TimeCourseExp[d][mouse] = []
    
    for rec in Recordings:
        idf = re.split('_', rec)[0]
        for d in exp_rec.keys():
            if rec in exp_rec[d]:
                TimeCourseExp[d][idf].append(TimeCourse[rec])
    
    # dummy dictionally to initialize TimeMxExp
    # Dict[R|W|N][dose][time_bin x mouse_id]
    TimeMxExp = {1:{}, 2:{}, 3:{}}
    for s in [1,2,3]:
        TimeMxExp[s] = {}
        for d in EMice.keys():
            TimeMxExp[s][d] = np.zeros((n, len(EMice[d])))
    
    for d in TimeCourseExp.keys():
        i = 0    
        for k in TimeCourseExp[d]:
            print k
            tmp = np.array(TimeCourseExp[d][k]).mean(axis=0)
            for s in [1,2,3]:
                # [time_bin x br_state] for mouse k
                #tmp = sum(TimeCourseExp[d][k]) / (1.0*len(TimeCourseExp[d][k]))                
                TimeMxExp[s][d][:,i] = tmp[:,s-1]
            i=i+1

    
    if pplot == True:
        plt.figure()
        
        ndose = len(EMice.keys())
        
        ax = plt.axes([0.1, 0.7, 0.8, 0.2])
        plt.errorbar(range(n), TimeMxCtr[1].mean(axis=1), yerr = TimeMxCtr[1].std(axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
        box_off(ax)
        plt.xlim((-0.5, n-0.5))
        #plt.ylim([0, 0.2])
        plt.yticks([0, 0.1, 0.2])
        plt.ylabel('% REM')    
        plt.ylim((0,0.2))    
        
        i = 1
        for d in TimeMxExp[1].keys():            
            c = 1 - 1.0/ndose*i
            plt.errorbar(range(n), TimeMxExp[1][d].mean(axis=1), yerr = TimeMxExp[1][d].std(axis=1),  color=[c, c, 1], fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            i += 1
    
        ax = plt.axes([0.1, 0.4, 0.8, 0.2])
        plt.errorbar(range(n), TimeMxCtr[2].mean(axis=1), yerr = TimeMxCtr[2].std(axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
        box_off(ax)
        plt.xlim((-0.5, n-0.5))
        #plt.ylim([0, 0.2])
        plt.yticks([0, 0.1, 0.2])
        plt.ylabel('% Wake')    
        plt.ylim((0,1))    
        plt.yticks(np.arange(0, 1.1, 0.25))
    
        i = 1
        for d in TimeMxExp[2].keys():            
            c = 1 - 1.0/ndose*i
            plt.errorbar(range(n), TimeMxExp[2][d].mean(axis=1), yerr = TimeMxExp[2][d].std(axis=1),  color=[c, c, 1], fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            i += 1
                
        ax = plt.axes([0.1, 0.1, 0.8, 0.2])
        plt.errorbar(range(n), TimeMxCtr[3].mean(axis=1), yerr = TimeMxCtr[3].std(axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
        box_off(ax)
        plt.xlim((-0.5, n-0.5))
        #plt.ylim([0, 0.2])
        plt.yticks([0, 0.1, 0.2])
        plt.ylabel('% NREM')    
        plt.ylim((0,1))    
        plt.yticks(np.arange(0, 1.1, 0.25))
        plt.show()
    
        i = 1
        for d in TimeMxExp[2].keys():            
            c = 1 - 1.0/ndose*i
            plt.errorbar(range(n), TimeMxExp[3][d].mean(axis=1), yerr = TimeMxExp[3][d].std(axis=1),  color=[c, c, 1], fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            i += 1
        
    return TimeMxCtr, TimeMxExp 

        

def sleep_spectrum(ppath, recordings, istate=1, pmode=1, twin=3, ma_thr=20.0, f_max=-1, pplot=True):
    """
    calculate power spectrum for brain state i state for the given recordings 
    @Param:
    ppath    -    folder containing all recordings
    recordings -  single recording (string) or list of recordings
    @Optional:
    istate   -    state for which to calculate power spectrum
    twin     -    time window (in seconds) for power spectrum calculation
                  the longer the higher frequency resolution, but the more noisy
    ma_thr   -    short wake periods <= $ma_thr are considered as sleep
    f_max    -    maximal frequency, if f_max==-1: f_max is maximally possible frequency
    pplot    -    pplot==1: plot figure showing result
    pmode    -    mode: 
                  pmode == 0, compare state during laser with baseline outside laser interval
                  pmode == 1, just plot power spectrum for state istate
    
    errorbars: If it's multiple mice make errorbars over mice; if it's multiple
    recordings of ONE mouse, show errorbars across recordings; 
    if just one recording show now errorbars
                  
    @Return:
    Pow     -    Dict[No loaser = 0|Laser = 1][array], where array: mice x frequencies, if more than one mouse;
                 otherwise, array: recordings x frequencies
    F       -    Frequencies
    """
    if type(recordings) != list:
        recordings = [recordings]

    Mice = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not(Mice.has_key(idf)):
            Mice[idf] = Mouse(idf, rec, 'E')
        else:
            Mice[idf].add(rec)
    
    # Spectra: Dict[mouse_id][laser_on|laser_off][list of powerspectrum_arrays]
    Spectra = {}
    Ids = Mice.keys()
    for i in Ids:
        Spectra[i] = {0:[], 1:[]}
        Spectra[i] = {0:[], 1:[]}

    for idf in Mice:
        for rec in Mice[idf].recordings:
            # load EEG
            EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EEG.mat'))['EEG']).astype('float')

            # load brain state
            M,S = load_stateidx(ppath, rec)
            sr = get_snr(ppath, rec)
            nbin = np.round(2.5*sr)
            dt = nbin * 1/sr
            nwin = np.round(twin*sr)

            M[np.where(M==5)]=2
            # flatten out microarousals
            seq = get_sequences(np.where(M==2)[0])
            for s in seq:
                if len(s)*dt <= ma_thr:
                    M[s] = 3            
            # get all sequences of state $istate
            seq = get_sequences(np.where(M==istate)[0])
    
            if pmode == 1:
                laser = load_laser(ppath, rec)
                (idxs, idxe) = laser_start_end(laser, SR=sr)
                # downsample EEG time to spectrogram time    
                idxs = [int(i/nbin) for i in idxs]
                idxe = [int(i/nbin) for i in idxe]
                
                laser_idx = []
                for (i,j) in zip(idxs, idxe):
                    laser_idx += range(i,j+1)
                laser_idx = np.array(laser_idx)
                
            if pmode == 1:
                # first analyze frequencies not overlapping with laser
                for s in seq:
                    s = np.setdiff1d(s, laser_idx)
                    if len(s)*nbin >= nwin:
                        sup = range(int(s[0]*nbin), int((s[-1]+1)*nbin))
                        if sup[-1]>len(EEG):
                            sup = range(int(s[0]*nbin), len(EEG))
                        Pow, F = power_spectrum(EEG[sup], nwin, 1/sr)
                        Spectra[idf][0].append(Pow)
                        
                # now analyze sequences overlapping with laser
                for s in seq:
                    s = np.intersect1d(s, laser_idx)
                    
                    if len(s)*nbin >= nwin:
                        # calculate power spectrum
                        # upsample indices
                        # brain state time 0     1         2
                        # EEG time         0-999 1000-1999 2000-2999
                        
                        sup = range(int(s[0]*nbin), int((s[-1]+1)*nbin))
                        if sup[-1]>len(EEG):
                            sup = range(int(s[0]*nbin), len(EEG))
                        Pow, F = power_spectrum(EEG[sup], nwin, 1/sr)
                        Spectra[idf][1].append(Pow)
                        
            # don't care about laser
            if pmode == 0:
                for s in seq:
                    if len(s)*nbin >= nwin:
                        sup = range(int(s[0]*nbin), int((s[-1]+1)*nbin))
                        if sup[-1]>len(EEG):
                            sup = range(int(s[0]*nbin), len(EEG))
                        Pow, F = power_spectrum(EEG[sup], nwin, 1/sr)
                        Spectra[idf][0].append(Pow)                        
                
            Pow = {0:[], 1:[]}
            if len(Ids)==1:
                #only one mouse
                Pow[0] = np.array(Spectra[Ids[0]][0])
                Pow[1] = np.array(Spectra[Ids[0]][1])
            else:
                Pow[0] = np.zeros((len(Ids),len(F)))
                Pow[1] = np.zeros((len(Ids),len(F)))
                i = 0
                for m in Ids:
                    Pow[0][i,:] = np.array(Spectra[m][0])
                    Pow[1][i,:] = np.array(Spectra[m][1])
                    i += 1

    if f_max > -1:
        ifreq = np.where(F<=f_max)[0]           
        for l in Pow.keys():
            Pow[0] = Pow[0][:,ifreq]
            if pmode==1: Pow[1] = Pow[1][:,ifreq]
            F = F[ifreq]
    else:
        f_max = F[-1]
    
    if pplot == True:
        
        plt.figure()
        set_fontsize(14)
        ax = plt.subplot(111)

        plt.plot(F, Pow[0].mean(axis=0), color='gray', lw=2)
        if len(Ids) == 1:
            a = Pow[0].mean(axis=0)-Pow[0].std(axis=0)
            b = Pow[0].mean(axis=0)+Pow[0].std(axis=0)            
            plt.fill_between(F, a, b, alpha=0.5, color='gray')
        if pmode==1: plt.plot(F, Pow[1].mean(axis=0), color='blue', lw=2)
        box_off(ax)
        plt.xlim([0, f_max])
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.show()
        
    return Pow, F


#TRANSITION ANALYSIS
    

def transition_analysis(ppath, rec_file, pre, laser_tend, tdown, large_bin):

    E = load_recordings(ppath, rec_file)[1]
    post = pre + laser_tend

    
    # Dict:  Mouse_id --> all trials of this mouse 
    MouseMx = {}
    for m in E:
        idf = re.split('_', m)[0]
        MouseMx[idf] = []
        
    for m in E:
        trials = _whole_mx(ppath, m, pre, post, tdown)
        idf = re.split('_', m)[0]
        MouseMx[idf] += trials
    
    
    # Markov Computation & Bootstrap 
    # time axis:
    t = np.linspace(-pre, post-large_bin, int((pre+post)/large_bin))
    t = t+large_bin/2.0

    ### indices during and outside laser stimulation
#    % indices of large bins during laser
#    ij = find(t>=0 & t<laser_tend); 
#    tmp = find(t>=0 & t<(laser_tend+after_laser));
#    % indices of large bins outsize laser (and $after_laser)
#    ii = setdiff(1:length(t), tmp); clear tmp;
    
    

def _whole_mx(ppath, name, pre, post, tdown, ptie_break=1):
    """
    @Return:
        List of trials
    """
    
    SR = get_snr(ppath, name);
    #NBINS = (round(SR)*5/2)*ds;   
    NBIN = np.round(2.5*SR)
    dt = NBIN * 1.0/SR
    ds = np.round(tdown/dt)

    ipre  = np.round(pre/tdown)
    ipost = np.round(post/tdown)
    
    
    # load brain state
    M = load_stateidx(ppath, name)[0]
    # downsample brain states
    M = _downsample_states(M, ds, ptie_break)

    
    (idxs, idxe) = laser_start_end(load_laser(ppath, name))

    idxs = [int(i/NBIN) for i in idxs]
    idxe = [int(i/NBIN) for i in idxe]

    trials = []
    for s in idxs:
        trials.append(M[s-ipre:s:ipost]) # i.e. element ii+1 is the first overlapping with laser
        
    #trials = np.array(trials)
    
    return trials

    
    
def _downsample_states(M, nbin, ptie_break=1):
    """
    ptie_break     -    tie break rule: 
                        if 1, wake wins over NREM which wins over REM (Wake>NREM>REM) in case of tie
    """
    
    n = np.floor(len(M))/(1.0*nbin)
    Mds = np.zeros((n,))

    for i in range(n):
        m = M[(i-1)*nbin:i*nbin]
            
        S = [len(np.where(m==s)[0]) for s in [1,2,3]]
        
        if ptie_break == 0:
            Mds[i] = np.argmax(S)+1  
        else:
            tmp = S[1,2,0]
            ii = np.argmax(tmp)
            ii = [1,2,0][ii]
            Mds[i] = ii+1
    
    return Mds
    


def infraslow_rhythm(ppath, recordings, ma_thr=20.0, min_dur = 160, band=[10,15], state=3, win=64, pplot=True, pflipx=True, pnorm=False):
    """
    calculate powerspectrum of EEG spectrogram to identify oscillations in sleep activity within different frequency bands;
    only contineous NREM periods are considered for
    @PARAMETERS:
    ppath        -       base folder of recordings
    recordings   -       single recording name or list of recordings
    
    @OPTIONAL:
    ma_thr       -       microarousal threshold; wake periods <= $min_dur are transferred to NREM
    min_dur      -       minimal duration [s] of a NREM period
    band         -       frequency band used for calculation
    win          -       window (number of indices) for FFT calculation
    pplot        -       if True, plot window showing result
    pflipx       -       if True, plot wavelength instead of frequency on x-axis
    pnorm        -       if True, normalize spectrum (for each mouse) by its total power
    
    @RETURN:
    SpecMx, f    -       ndarray [mice x frequencies], vector [frequencies]
    """
    import scipy.linalg as LA

    #min_dur = win*2.5
    min_dur = np.max([win*2.5, min_dur])
    
    if type(recordings) != list:
        recordings = [recordings]

    Spec = {}    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        Spec[idf] = []
    mice = Spec.keys()
    
    for rec in recordings:
        idf = re.split('_', rec)[0]

        # sampling rate and time bin for spectrogram
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        dt = NBIN * 1/SR
        
        # load sleep state
        M = load_stateidx(ppath, rec)[0]
        seq = get_sequences(np.where(M==state)[0], np.round(ma_thr/dt))
        seq = [range(s[0], s[-1]+1) for s in seq]
        
        # load frequency band
        P = so.loadmat(os.path.join(ppath, rec,  'sp_' + rec + '.mat'));
        SP = np.squeeze(P['SP'])
        freq = np.squeeze(P['freq'])
        ifreq = np.where((freq>=band[0]) & (freq<=band[1]))
        pow_band = SP[ifreq,:].mean(axis=0)
        
        seq = [s for s in seq if len(s)*dt >= min_dur]   
        for s in seq:
            y,f = power_spectrum(pow_band[:,s], win, dt)
            y = y.mean(axis=0)
            Spec[idf].append(y)
        
    # Transform %Spec to ndarray
    SpecMx = np.zeros((len(Spec.keys()), len(f)))
    i=0
    for idf in Spec.keys():
        SpecMx[i,:] = np.array(Spec[idf]).mean(axis=0)
        if pnorm==True:
            SpecMx[i,:] = SpecMx[i,:]/LA.norm(SpecMx[i,:])
        i += 1

    if pplot == True:
        plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])
        
        x = f[1:]
        if pflipx == True: x = 1.0/f[1:]
        y = SpecMx[:,1:]
        if len(mice) <= 1:
            ax.plot(x, y.mean(axis=0), color='gray', lw=2)
            
        else:
            ax.errorbar(x, y.mean(axis=0), yerr=y.std(axis=0), color='gray', fmt='-o')

        box_off(ax)
        if pflipx == True:
            plt.xlabel('Wavelength (s)')
        else:
            plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.show()

    return SpecMx, f





def ma_rhythm(ppath, recordings, ma_thr=20.0, min_dur = 160, band=[10,15], state=3, win=64, pplot=True, pflipx=True, pnorm=False):
    """
    calculate powerspectrum of EEG spectrogram to identify oscillations in sleep activity within different frequency bands;
    only contineous NREM periods are considered for
    @PARAMETERS:
    ppath        -       base folder of recordings
    recordings   -       single recording name or list of recordings
    
    @OPTIONAL:
    ma_thr       -       microarousal threshold; wake periods <= $min_dur are transferred to NREM
    min_dur      -       minimal duration [s] of a NREM period
    band         -       frequency band used for calculation
    win          -       window (number of indices) for FFT calculation
    pplot        -       if True, plot window showing result
    pflipx       -       if True, plot wavelength instead of frequency on x-axis
    pnorm        -       if True, normalize spectrum (for each mouse) by its total power
    
    @RETURN:
    SpecMx, f    -       ndarray [mice x frequencies], vector [frequencies]
    """
    import scipy.linalg as LA

    min_dur = np.max([win*2.5, min_dur])
    
    if type(recordings) != list:
        recordings = [recordings]

    Spec = {}    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        Spec[idf] = []
    mice = Spec.keys()
    
    for rec in recordings:
        idf = re.split('_', rec)[0]

        # sampling rate and time bin for spectrogram
        #SR = get_snr(ppath, rec)
        #NBIN = np.round(2.5*SR)
        #dt = NBIN * 1/SR
        dt = 2.5
        
        # load sleep state
        M = load_stateidx(ppath, "", ann_name=rec)[0]
        Mseq = M.copy()
        Mseq[np.where(M != 2)] = 0
        Mseq[np.where(M == 2)] = 1
        seq = get_sequences(np.where(M==state)[0], ibreak=int(np.round(ma_thr/dt))+1)
        seq = [range(s[0], s[-1]+1) for s in seq]
        
        #pdb.set_trace()
        # load frequency band
        #P = so.loadmat(os.path.join(ppath, rec,  'sp_' + rec + '.mat'));
        #SP = np.squeeze(P['SP'])
        #freq = np.squeeze(P['freq'])
        #ifreq = np.where((freq>=band[0]) & (freq<=band[1]))
        #pow_band = SP[ifreq,:].mean(axis=0)
        
        #pdb.set_trace()
        seq = [s for s in seq if len(s)*dt >= min_dur]   
        #pdb.set_trace()
        for s in seq:
            y,f = power_spectrum(Mseq[s], win, dt)
            #y = y.mean(axis=0)
            Spec[idf].append(y)
        
    # Transform %Spec to ndarray
    SpecMx = np.zeros((len(Spec.keys()), len(f)))
    i=0
    for idf in Spec.keys():
        SpecMx[i,:] = np.array(Spec[idf]).mean(axis=0)
        if pnorm==True:
            SpecMx[i,:] = SpecMx[i,:]/LA.norm(SpecMx[i,:])
        i += 1

    if pplot == True:
        plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])
        
        x = f[1:]
        if pflipx == True: x = 1.0/f[1:]
        y = SpecMx[:,1:]
        if len(mice) <= 1:
            ax.plot(x, y.mean(axis=0), color='gray', lw=2)
            
        else:
            ax.errorbar(x, y.mean(axis=0), yerr=y.std(axis=0), color='gray', fmt='-o')

        box_off(ax)
        if pflipx == True:
            plt.xlabel('Wavelength (s)')
        else:
            plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.show()

    return SpecMx, f




if __name__ == '__main__':
    #ppath = '/Users/tortugar/Documents/Berkeley/Data_old/RawData'
    #name = 'G168_090116n1'
    
    ppath = '/Users/tortugar/Documents/Penn/Data/RawData/'
    name = 'S4_110317n1'
    
    
    #ppath = '/Volumes/BB8/Data/RawData'
    #name  = 'G195_010617n1'
    
    
    #ppath = '/Volumes/BB8/Data/RawData'
    #ppath = '/Volumes/BB7/Data/RawData/'
    #name = 'G1_10232012n1'
    
    #name = 'G215_030217n1'
    #name = 'G215_030117n1'
#
#    EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'))['EEG'][:,0]
#    SR = get_snr(ppath, name)
#    P, f = power_spectrum(EEG, int(round(SR)), 1/SR)
#
#    swin = round(SR)*5;
#    fft_win = round(swin/5);
#
#    # 0.5 Hz frequency resolution
#    Pxx, f, t = spectral_density(EEG, int(swin), 1*int(fft_win), 1/SR)
#    
    #calculate_spectrum(ppath, name, fres=0.5)
   

    C,D,f,t = laser_triggered_eeg(ppath, name, 120, 240, 50, pnorm=2, pplot=1)
    #threshold_crossing(np.array([1,1,1,0,1,1,1]), 0.5, 3, 1, 1)
    #M,S = sleep_state(ppath, name, pwrite=0, pplot=1)
    
    #plt.ion()
    plt.figure()
   
    ax = plt.subplot(111)
    #plt.imshow(np.flipud(A))
    plt.pcolormesh(t, f, C) #, vmin = np.median(A)*0.1, vmax = np.median(A)*2)
    plt.axis('tight')
    #ax.set_yscale('log')
    #ax.caxis([0, A.median()])
    #plt.pause(0.001)
    #plt.show()
    
    ipre = np.where(t<0)[0]
    ilsr = np.where((t>=0) & (t<=120))[0]
    
    
    #plt.figure()
    #plt.ion()
    #plt.plot(f,C[:,ipre].mean(axis=1))
    #plt.plot(f,C[:,ilsr].mean(axis=1))
    
    #plt.pause(0.001)
    #plt.show()
    