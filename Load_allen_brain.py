import numpy as np
import matplotlib.pyplot as plt
import h5py
from numpy.core.defchararray import less_equal
#%%
fraw = h5py.File('/Users/skeeley/Desktop/AllenBrainData/103920_processed.h5', 'r')
f = h5py.File('/Users/skeeley/Desktop/AllenBrainData/103920.h5', 'r')





def bin_spks(dfftimes, sptimes):

    j = 0
    binnedspks = np.zeros(np.shape(dfftimes))
    currenttime= sptimes[j]
    for i in np.arange(np.size(dfftimes)):
        while currenttime>= dfftimes[i] and currenttime <dfftimes[i+1]:
            binnedspks[i]+=1
            j+=1
            if j >= np.size(sptimes):
                break
            currenttime = sptimes[j]
    return binnedspks


def create_Xstim_vec(binned_stims,sweep_order):

    xstimVec = np.zeros_like(binned_stims)
    j = 0
    stimulus = 0
    for i in np.arange(np.size(binned_stims)):
        xstimVec[i] = stimulus
        if binned_stims[i] == 1:
            stimulus = sweep_order[j]
            j += 1

    return xstimVec

def create_Xstim_full(binned_stims,sweep_order, sweep_table):

    xstimVec = np.zeros([np.shape(binned_stims)[0],4])
    j = 0
    stimulus = 0
    for i in np.arange(np.shape(binned_stims)[0]):
        xstimVec[i,:] = sweep_table[:,int(stimulus)]
        if binned_stims[i] == 1:
            stimulus = sweep_order[j]
            j += 1

    return xstimVec


#%%
list(fraw.keys())
list(f.keys())


trace = f['dff']
sptimes = np.array(f['sptimes'])
dfsamp = np.array(f['dto'])

dfftimes = dfsamp*np.arange(np.size(trace))

binned_spks = bin_spks(dfftimes, sptimes)
plt.plot(dfftimes, trace)
plt.plot(dfftimes, binned_spks*.1)
plt.legend(['df/f', 'spks'])
plt.xlabel('time (s)')
plt.show()



### stim params
iStimOn = np.array(fraw['iStimOn'])
iStimOff= np.array(fraw['iStimOff'])
sweep_table= np.array(fraw['sweep_table'])
sweep_order= np.array(fraw['sweep_order'])

dte = np.array(fraw['dte'])
iFrames = fraw['iFrames'][0]

zeroed_iFrames = iFrames - iFrames[0]
zeroed_istimOn = iStimOn - iFrames[0]
zeroed_istimOff = iStimOff - iFrames[0]
iStimOn_times= zeroed_istimOn*dte
iStimOff_times= zeroed_istimOff*dte
iFrames_times = zeroed_iFrames*dte


iStim_times = np.empty((iStimOn_times.size + iStimOff_times.size,), dtype=iStimOn_times.dtype)
iStim_times[0::2] = iStimOn_times
iStim_times[1::2] = iStimOff_times




binned_stims = bin_spks(dfftimes, iStim_times)


#xstimVec_raw = create_Xstim_vec(binned_stims,sweep_order)
xstimVec_full =  create_Xstim_full(binned_stims,sweep_order, sweep_table)
xstimVec_ori = xstimVec_full[:,0]  # order is orientation, phase, SF, contrast. First one shoudl be orientation but could be different per cell so worth checking



S = 10 # max spike count to consider
ygrid = np.arange(0, S+1)

# Set up GLM
D_in = 19 # dimension of stimulus
D = D_in + 1 # total dims with bias
T = np.size(trace)
dt = dfsamp
#bias = npr.randn(T)
#nlfun = lambda x : softplus_stable(x, bias=None, dt=dt)


#Xstim = npr.randn(T,1)
Xstim = np.expand_dims(xstimVec_ori, axis =1 )
from scipy.linalg import hankel
Xmat1 = hankel(Xstim[:,0], Xstim[:,0][-D_in:])
Xmat = np.hstack((np.ones((T,1)), Xmat1))



# %%
