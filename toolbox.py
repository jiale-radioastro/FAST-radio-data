import numpy as np

def noise_onoff(num,ttbin=1,file=0):  
    # default: noise diode signals are periodically injected. e.g. on-1s off-15s on-1s off-15s ......   
    # To determine the index of 'num'th cycle of noise 'on' and noise 'off'
    # noise 'on': noise_on[0] - noise_on[1]   noise 'off': noise_off[0] - noise_off[1]
    # num: the number of the cycle
    # ttbin: first round of time rebin. default 1
    # file: the number of the read file. default 0
    # tinterval: the length of the cycle period. default 80*1024
    # tnoise: the length of the noise injection time. default 5*1024
    # filelen: the length of a single file. default 512*1024
    
    #num starts from 0   default ttbin at 1  default start file at 0
    tinterval=80*1024//ttbin
    tnoise=5*1024//ttbin
    prelen=512*1024//ttbin*file
    prenum=int(np.ceil(prelen/tinterval))
    num=num+prenum
    noise_on=[num*tinterval-prelen,num*tinterval+tnoise-prelen]
    noise_off=[num*tinterval+tnoise-prelen,num*tinterval+2*tnoise-prelen]
    return noise_on,noise_off

def noise_count(ttbin=1,file=0,length=512*1024):
    tinterval=80*1024//ttbin
    tnoise=5*1024//ttbin
    prelen=512*1024//ttbin*file
    n_start=np.ceil(prelen/tinterval)
    n_end=np.ceil((prelen+length//ttbin)/tinterval)
    return int(n_end-n_start)
