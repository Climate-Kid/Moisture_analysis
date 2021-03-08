import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import xarray as xr
import scipy
from scipy.stats import binned_statistic

# read land-sea fraction 
## next set land fraction larger than 0.01 as nan
## set land fraction smaller than 0.01 as zero so that we could later add precip or precipitable water directly
## with lsm_nan
path='.../lsm_10.nc'
ps = xr.open_dataset(path)
lsm=ps['lsm'].values[0]
lsm_nan=lsm.copy()
lsm_nan[lsm<0.01]=0
lsm_nan[lsm>=0.01]=np.nan
lsm_nan_20S20N=lsm_nan[70:111]

# read precipitable water from ERA5 (already regrided to 1d x 1d) 
pw_obs=np.zeros((20 , 366, 41, 360))
path='/reanalyasis/'
ps = xr.open_dataset(path+'cwv_daily_1d.nc')
mid_pw=ps['tcw'].values
a=0
pw_obs=np.zeros((20, 123, 41, 360))
for iyear in np.arange(2000,2020):
    if np.mod(iyear,4)==0:
        pw_obs[iyear-2000]=mid_pw[a+182:a+305] # JASO
        a=a+366
    else:
        pw_obs[iyear-2000]=mid_pw[a+181:a+304]
        a=a+365
lon_tpobs=ps['lon'].values
lat_tpobs=ps['lat'].values
# TP GPCP
path='.../GPCP1_3/'
ps = xr.open_dataset(path+'gpcp_2000_2019.nc')
tp_cheyenne=ps['tp'].values
tp_obs=np.zeros((20, 123, 41, 360))
a=0
for iyear in np.arange(2000,2020):
    if np.mod(iyear,4)==0:
        tp_obs[iyear-2000]=tp_cheyenne[iyear-2000,182:305] #JASO
        a=a+366
    else:
        tp_obs[iyear-2000]=tp_cheyenne[iyear-2000,181:304]
        a=a+365

# GEFS data
## data structure:  20, 123, 16, 41, 360   <=> iyear, iday, forecast lead time, ilat, ilon
# lat: 0->41 = 20N-20S;    lon: 0->360 = 0 degree east - 359 degree east
tp_GEFS=np.load('.../tp_25Feb_forhist_gefs.npy')
pw_GEFS=np.load('.../pw_25Feb_forhist_gefs.npy')

# PW probability distribution
## GEFS
pw_hist_gefs=np.zeros((20,16,129))
ilon1=120; ilon2=180
mid_pw_gefs=pw_GEFS[:,:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,None,:,ilon1:ilon2]
for iyear in np.arange(2000,2020):
    for i in np.arange(16):
        midi=~np.isnan(mid_pw_gefs[iyear-2000,:,i])
        hist, edges = np.histogram(mid_pw_gefs[iyear-2000,:,i][midi].flatten(), bins=np.arange(10,75,.5))
        total_num1=np.nansum(~np.isnan(mid_pw_gefs[iyear-2000,:,i])) 
        pw_hist_gefs[iyear-2000,i]=hist/total_num1
# Obs sh
pw_hist_obs=np.zeros((20,129))

mid_pw_obs=pw_obs[:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,:,ilon1:ilon2]
for iyear in np.arange(2000,2020):
    hist, edges = np.histogram(mid_pw_obs[iyear-2000,:].flatten(), bins=np.arange(10,75,.5))
    total_num2=np.nansum(~np.isnan(mid_pw_obs[iyear-2000])) 
    pw_hist_obs[iyear-2000]=hist/total_num2
    

####### plot
binsu=np.arange(10,75,.5)
fig, axes = plt.subplots(figsize=(6,4))
colors = plt.cm.jet(np.linspace(0,1,16))

for i in np.arange(0,16):
    midbins=(binsu[1:]+binsu[:-1])/2
    mid=np.nanmean(pw_hist_gefs[:,i,:],axis=0)[:110]
    mid=mid*100
    axes.plot(midbins[:110], mid, color=colors[i],label='Day '+str(i+1))
mid=np.nanmean(pw_hist_obs[:,:],axis=0)[:110]
mid=mid*100
axes.plot(midbins[:110], mid,linewidth=3, color='k',label='ERA5')
axes.set_xlabel('PW [mm]',fontsize=16)
axes.set_ylabel('Percentage [%]',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
axes.set_title('(a) W.Pac: PW hist',fontsize=18,loc='left')
plt.xticks(np.array([10,20,30,40,50,60,65]), ['10','20','30','40','50','60','65'],fontsize=16)  ; 
plt.legend(numpoints=1,loc='upper left', ncol=2)
fig.tight_layout()
plt.savefig('.../PW_hist.pdf',format='pdf',dpi=300, bbox_inches='tight')

# TP probability distribution
## GEFS 
tp_hist_gefs=np.zeros((20,16,24))
mid_tp_gefs=tp_GEFS[:,:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,None,:,ilon1:ilon2]
for iyear in np.arange(2000,2020):
    for i in np.arange(16):
        hist, edges = np.histogram(mid_tp_gefs[iyear-2000,:,i].flatten(), bins=np.arange(0,50,2))
        total_num1=np.nansum(~np.isnan(mid_tp_gefs[iyear-2000,:,i])) 
        tp_hist_gefs[iyear-2000,i]=hist/total_num1
    
## Obs
tp_hist_obs=np.zeros((20,24))

mid_tp_obs=tp_obs[:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,:,ilon1:ilon2]
for iyear in np.arange(2000,2020):
    hist, edges = np.histogram(mid_tp_obs[iyear-2000].flatten(), bins=np.arange(0,50,2))
    total_num2=np.nansum(~np.isnan(mid_tp_obs[iyear-2000])) 
    tp_hist_obs[iyear-2000]=hist/total_num2

####### plot
binsu=np.arange(0,50,2)#

fig, axes = plt.subplots(figsize=(8,4))
colors = plt.cm.jet(np.linspace(0,1,16))
midbins=(binsu[1:]+binsu[:-1])/2
mid=np.nanmean(tp_hist_obs[:,:],axis=0)
mid=mid*100
axes.bar(midbins, mid,width=1.8, label='GPCP')

for i in np.array([0,1]):
    midbins=(binsu[1:]+binsu[:-1])/2
    mid_tp=np.nanmean(tp_hist_gefs[:,:,:],axis=0)[:,0:-1]
    mid=np.nanmean(tp_hist_gefs[:,i,:],axis=0)
    mid=mid*100
    axes.plot(midbins, mid,'o-', color=colors[i-1],label='Day '+str(i+1))

axes.set_xlabel('TP [mm]',fontsize=16)
axes.set_ylabel('Percentage [%]',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
axes.set_title('(a) W. Pac: TP hist',fontsize=18,loc='left',y=1.025)
plt.xticks(np.arange(0,50,5),fontsize=15)  ; 
axes.set_yscale('log')
plt.legend(numpoints=1,loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
fig.tight_layout()
plt.savefig('.../tp_hist.pdf',format='pdf',dpi=300, bbox_inches='tight')


# PW versus TP
## GEFS 
pw_tp_gefs=np.zeros((20,16,129))
mid_pw_gefs=pw_GEFS[:,:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,None,:,ilon1:ilon2]
mid_tp_gefs=tp_GEFS[:,:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,None,:,ilon1:ilon2]
for iyear in np.arange(2000,2020):
    for i in np.arange(16):
        midpw=mid_pw_gefs[iyear-2000,:,i]
        midtp=mid_tp_gefs[iyear-2000,:,i]
        mid=binned_statistic(midpw[~np.isnan(midpw)], midtp[~np.isnan(midtp)],bins=np.arange(10,75,.5))
        pw_tp_gefs[iyear-2000,i]=mid[0]

## Obs 
pw_tp_obs=np.zeros((20,129))
mid_pw_obs=pw_obs[:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,:,ilon1:ilon2]
mid_tp_obs=tp_obs[:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,:,ilon1:ilon2]
for iyear in np.arange(2000,2020):
    midpw=mid_pw_obs[iyear-2000,:]
    midtp=mid_tp_obs[iyear-2000,:]
    mid=binned_statistic(midpw[~np.isnan(midtp)], midtp[~np.isnan(midtp)],bins=np.arange(10,75,.5))
    pw_tp_obs[iyear-2000]=mid[0]

####### plot
binsu=np.arange(10,75,.5)#
fig, axes = plt.subplots(figsize=(8,4))
colors = plt.cm.jet(np.linspace(0,1,16))

for i in np.arange(0,16):
    midbins=(binsu[1:]+binsu[:-1])/2
    mid_pw_tp=np.nanmean(pw_tp_gefs[:,i,:],axis=0)[:110]
    axes.plot(midbins[:110], mid_pw_tp, color=colors[i],label='Day '+str(i+1))
    
mid_pw_tp=np.nanmean(pw_tp_obs[:,:],axis=0)[:110]
axes.plot(midbins[:110], mid_pw_tp, color='k',label='GPCP',linewidth=2)
axes.set_xlabel('PW [mm]',fontsize=16)
axes.set_ylabel('Precip [mm/day]',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
axes.set_title('(a) W. Pac: Precip vs. PW ',fontsize=18,loc='left',y=1.025)
plt.xticks(np.array([10,20,30,40,50,60,65]), ['10','20','30','40','50','60','65'],fontsize=13)  ; 
plt.legend(numpoints=1,loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
fig.tight_layout()
plt.savefig('.../pw_tp.pdf',format='pdf',dpi=300, bbox_inches='tight')



Written by Jiacheng Ye; supervised by Professor Zhuo Wang




