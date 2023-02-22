import numpy as np

################################################subtract noise diode signal#####################################################
def noise_onoff(num,ttbin=1,file=0,filelen=512*1024,tperiod=80*1024,tnoise=5*1024):  
    # default: noise diode signals are periodically injected. e.g. on-1s off-15s on-1s off-15s ......   
    # To determine the index of 'num'th cycle of noise 'on' and noise 'off'
    # noise 'on': noise_on[0] - noise_on[1]   noise 'off': noise_off[0] - noise_off[1]
    # num: the sequence number of the cycle in the fits file. cycle starts with the noise injection
    # ttbin: first round of time rebin. default 1
    # file: the sequence number of the fits file. default 0
    # filelen: the length of a single file in time domain. default 512*1024
    # tperiod: the length of the cycle period. default 80*1024
    # tnoise: the length of the noise injection time. default 5*1024
    # the default value of tinterval and tnoise accords to the noise injection mode: 1s noise every 16s, time cadence 0.2ms
    tperiod=tperiod//ttbin
    tnoise=tnoise//ttbin
    prelen=filelen//ttbin*file
    prenum=int(np.ceil(prelen/tperiod))              # the number of previous cycles
    num=num+prenum                    # the real sequence number of the cycle
    noise_on=[num*tperiod-prelen,num*tperiod+tnoise-prelen]
    noise_off=[num*tperiod+tnoise-prelen,num*tperiod+2*tnoise-prelen]
    return noise_on,noise_off

def noise_count(file=0,currlen=512*1024,filelen=512*1024,tperiod=80*1024,tnoise=5*1024):
    # count the number of noise diode signals in a file
    prelen=filelen*file
    n_start=np.ceil(prelen/tperiod)
    n_end=np.ceil((prelen+currlen)/tperiod)
    return int(n_end-n_start)

################################################polarization and flux calibration#############################################
def obs2true(I,Q,U,V,f=1,xi=0,au2t=1):
    # based on  Xiao-Hui Sun et al 2021 Res. Astron. Astrophys. 21 282
    # calibrate the observed flux in four Stokes parameters
    I2=(I-Q*f)/(1-f**2)
    Q2=(Q-I*f)/(1-f**2)
    U2=U*np.cos(2*xi)+V*np.sin(2*xi)
    V2=-U*np.sin(2*xi)+V*np.cos(2*xi)
    return I2*au2t,Q2*au2t,U2*au2t,V2*au2t

def fxi_cal(I,Q,U,V,noise_T=12.5*np.ones(1024)):
    # based on  Xiao-Hui Sun et al 2021 Res. Astron. Astrophys. 21 282
    # determine the differential gain and phase of the orthogonal feeds using the noise diode signals
    f=Q/I
    xi=0.5*np.angle(U+complex(0,1)*V)
    I2=(I-Q*f)/(1-f**2)
    return f,xi,noise_T/I2

def read_dat(datfile):
    # read data array from .dat file
    datstr = np.genfromtxt(datfile,delimiter='\t',dtype=str)
    datdata=[]
    for linestr in datstr:
        tmp=linestr.split(' ')
        while '' in tmp:
            tmp.remove('')
        for num in tmp:
            datdata.append(float(num))
     
    return datdata

def give_noise_T(adat,bdat,freqdat,freq):
    # derive the noise temperature from the .dat file
    # .dat files can be downloaded from the noise diode calibration report
    # (https://fast.bao.ac.cn/cms/category/telescope_performance/noise_diode_calibration_report/)
    # T_noise=np.sqrt(T_A*T_B)
    datstr = np.genfromtxt(adat,delimiter='\t',dtype=str)
    adatdata=read_dat(adat)
    bdatdata=read_dat(bdat)
    freqdatdata=read_dat(freqdat)
    raw_T=np.sqrt(np.array(adatdata)*np.array(bdatdata))
    raw_freq=np.array(freqdatdata)
    return np.interp(freq,raw_freq,raw_T)

################################################data concat and reduction####################################################
def concatdata(data,pol):
    # concatenate data in the given polarization
    # input data dimension time time pol freq 0   512 1024 4 1024 1
    # output data dimension time freq  512*1024 1024
    # notice that the raw data is in the format of uint8, better change to int
    # I=XX+YY, Q=XX-YY, U=XY+YX, V=XY-YX
    if pol=="XX":
        return np.int16(np.concatenate(data['data'][:,:,0,:,0]))
    elif pol=="YY":
        return np.int16(np.concatenate(data['data'][:,:,1,:,0]))
    elif pol=="XY":
        return np.int16(np.concatenate(data['data'][:,:,2,:,0]))
    elif pol=="YX":
        return np.int16(np.concatenate(data['data'][:,:,3,:,0]))
    elif pol=="I":
        return np.int16(np.concatenate(data['data'][:,:,0,:,0]))+np.int16(np.concatenate(data['data'][:,:,1,:,0]))
    elif pol=="Q":
        return np.int16(np.concatenate(data['data'][:,:,0,:,0]))-np.int16(np.concatenate(data['data'][:,:,1,:,0]))
    elif pol=="U":
        return np.int16(np.concatenate(data['data'][:,:,2,:,0]))+np.int16(np.concatenate(data['data'][:,:,3,:,0]))
    elif pol=="V":
        return np.int16(np.concatenate(data['data'][:,:,2,:,0]))-np.int16(np.concatenate(data['data'][:,:,3,:,0]))
    
def give_sp(filedata,ttbin):
    # obtain four Stokes parameters from filedata
    I_data=trebin(concatdata(filedata,"I"),ttbin)
    Q_data=trebin(concatdata(filedata,"Q"),ttbin)
    U_data=trebin(concatdata(filedata,"U"),ttbin)
    V_data=trebin(concatdata(filedata,"V"),ttbin)
    return [I_data, Q_data, U_data, V_data]
    
def trebin(data,ttbin):
    # rebin data in time dimension
    # data dimension time freq
    if ttbin==1:
        return data
    else:
        sz=np.shape(data)
        data2=np.zeros([sz[0]//ttbin,sz[1]])
        for i in range(sz[0]//ttbin):
            data2[i,:]=np.mean(data[ttbin*i:ttbin*(i+1),:],axis=0)
        return data2

def frebin(data,ffbin):
    # rebin data in freq dimension
    # data dimension time freq
    if ffbin==1:
        return data
    else:
        sz=np.shape(data)
        data2=np.zeros([sz[0],sz[1]//ffbin])
        for i in range(sz[1]//ffbin):
            data2[:,i]=np.mean(data[:,ffbin*i:ffbin*(i+1)],axis=1)
        return data2

################################################give timelist and freqlist######################################################
def give_time(tbin=0.000196608,ttbin=1,ttbin2=1,length=512*1024,prelength=512*1024,file=0):
    # give time list in s
    # tbin: time interval in s
    # ttbin: first round of time rebin
    # ttbin2: second round of time rebin
    # length: the length of the current file in time domain
    # prelength: the length of the previous files in time domain
    num=length//ttbin//ttbin2
    pretime=prelength*file*tbin
    return pretime+tbin*ttbin*ttbin2*np.linspace(0,num-1,num)

def give_freq(ffbin,freqlist=np.linspace(1000,1000+1023*0.48828125,1024),length=1024):
    # give frequency list in MHz
    # ffbin: frequency rebin
    # length: the length of the current file in frequency domain
    # freqlist: the raw frequency list. can be derived from 'filedata[0]['DAT_FREQ']'
    num=length//ffbin
    return np.array([np.mean(freqlist[ffbin*i:ffbin*(i+1)]) for i in range(num)])

################################################remove background##############################################################
def trend_remove(dspec,method='fit',index=1,average=1):
    # remove the long-term background flux variation
    # two methods available
    # 'fit' method (recommended): use polynomial fitting to determine the background. 
    # linear fitting only works for short period (~2 min)
    # index: the index of the polynomial fitting
    # 'startend' method: draw a line between the start and end points to estimate the background
    # average: the averaging length
    sz=np.shape(dspec)
    dspec_new=np.zeros([sz[0],sz[1]])
    for freqi in range(sz[1]):
        lc=dspec[:,freqi]
        if method == 'startend':
            y0=np.mean(lc[range(average)])
            x0=0.5*(0+average-1)
            y1=np.mean(lc[range(sz[0]-average,sz[0])])
            x1=0.5*(sz[0]-average+sz[0]-1)
            k=(y1-y0)/(x1-x0)
            dspec_new[:,freqi]=lc-k*(np.arange(sz[0])-x0)-y0
        elif method == 'fit':
            kb=np.polyfit(np.arange(sz[0]),lc,index)
            f1=np.poly1d(kb)
            dspec_new[:,freqi]=lc-f1(np.arange(sz[0]))
    return dspec_new

def fluct_remove(dspec,method='freqwin',freqwin=[],average=1):
    # remove the random broadband flux fluctuation of Stokes I
    # three methods available
    # 'win' method: choose a certain quiet spectral window as background
    # 'min' method: choose the frequency channels with the smallest flux and do averaging
    # 'winmin' method: combine the two methods above
    sz=np.shape(dspec)
    dspec_new=np.zeros([sz[0],sz[1]])
    for timeri in range(sz[0]):
        spec=np.array(dspec[timeri,:])
        if method=='win':
            dspec_new[timeri,:]=np.array(dspec[timeri,:])-np.nanmean(spec[freqwin[0]:freqwin[1]])
        elif method=='min':
            spec=np.sort(spec)
            spec = spec[np.isfinite(spec)]
            dspec_new[timeri,:]=np.array(dspec[timeri,:])-np.mean(spec[0:average-1])
        elif method=='winmin':
            spec=spec[freqwin[0]:freqwin[1]]
            spec=np.sort(spec)
            spec = spec[np.isfinite(spec)]
            dspec_new[timeri,:]=np.array(dspec[timeri,:])-np.mean(spec[0:average-1])
    return dspec_new
