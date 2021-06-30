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
xstimVec = create_Xstim_vec(binned_stims,sweep_order)

#dto = np.array(f['dto'])
# ephys_raw = np.array(f['ephys_raw'])
# sptimes = np.array(f['sptimes'])
# StimTrig = np.array(fraw['StimTrig'][0])

# spk = np.array(fraw['spk'])
# Vmfd = np.array(fraw['Vmfd'][0])

# etimes = np.arange(0,dte*np.size(Vmfd[528046:]),dte) ## from raw data
# plt.plot(otimes_raw, f_cell)
# plt.plot(etimes[1:], Vmfd[528046:]*100000)


#otimes = np.arange(0,dto*np.size(trace),dto)

### subtract iframes[0]*dte from stimulus times, this should align with dff starting point. 

### times in spktimes in the dff data is aligned with 0 probably...

## primary trick is ephys offset, which is exactly algined with stimuli --- 
# istimon[0] is going to be when the first stimulus goes on, istimoff[0] COULD be when the secondstimulus goes
# on if there is no blank stimulus in between. There are 700 presentations of 64 possible stimuli. Each stimulus is a 
# four vector (orientation, SF, etc,etc) with a -1 if there is no stimulus. There should be 
plt.show()


S = 10 # max spike count to consider
ygrid = np.arange(0, S+1)

# Set up GLM
D_in = 19 # dimension of stimulus
D = D_in + 1 # total dims with bias
T = np.size(trace)
dt = dfsamp
#bias = npr.randn(T)
nlfun = lambda x : softplus_stable(x, bias=None, dt=dt)


#Xstim = npr.randn(T,1)
Xstim = np.expand_dims(xstimVec, axis =1 )
from scipy.linalg import hankel
Xmat1 = hankel(Xstim[:,0], Xstim[:,0][-D_in:])
Xmat = np.hstack((np.ones((T,1)), Xmat1))



# %%
