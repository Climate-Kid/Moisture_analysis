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
## lsm_nan structure: 41 x 360
path='YOUR_DIRECTORY/lsm_10.nc'
ps = xr.open_dataset(path)
lsm=ps['lsm'].values[0]
lsm_nan=lsm.copy()
lsm_nan[lsm<0.01]=0
lsm_nan[lsm>=0.01]=np.nan
lsm_nan_20S20N=lsm_nan[70:111]

# read data (already regrided to 1d x 1d) 
# load GEFS data and ERA5 data (for obs PW analysis) and GPCP data (for obs precip analysis) from 2000 to 2001 but only in the first five days for July in each year
# due to github maximum file size limit
### data structure:  2, 5, 16, 41, 360   <=> iyear, iday, forecast lead time, ilat, ilon
### lat: 0->41 = 20N->20S;    lon: 0->360 = 0 degree east - 359 degree east
tp_GEFS=np.load('YOUR_DIRECTORY/tp_GEFS.npy')
pw_GEFS=np.load('YOUR_DIRECTORY/pw_GEFS.npy')
tp_obs=np.load('YOUR_DIRECTORY/tp_obs.npy')
pw_obs=np.load('YOUR_DIRECTORY/pw_obs.npy')


# PW probability distribution
## GEFS
pw_hist_gefs=np.zeros((16,129))
ilon1=120; ilon2=180
mid_pw_gefs=pw_GEFS[:,:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,None,:,ilon1:ilon2]
for i in np.arange(16):
    midi=~np.isnan(mid_pw_gefs[:,:,i])
    hist, edges = np.histogram(mid_pw_gefs[:,:,i][midi].flatten(), bins=np.arange(10,75,.5))
    total_num1=np.nansum(~np.isnan(mid_pw_gefs[:,:,i])) 
    pw_hist_gefs[i]=hist/total_num1
# Obs pw
##pw_hist_obs=np.zeros(129)

mid_pw_obs=pw_obs[:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,:,ilon1:ilon2]
hist, edges = np.histogram(mid_pw_obs.flatten(), bins=np.arange(10,75,.5))
total_num2=np.nansum(~np.isnan(mid_pw_obs)) 
pw_hist_obs=hist/total_num2



####### plot
binsu=np.arange(10,75,.5)
fig, axes = plt.subplots(figsize=(6,4))
colors = plt.cm.jet(np.linspace(0,1,16))

for i in np.arange(0,16):
    midbins=(binsu[1:]+binsu[:-1])/2
    mid=pw_hist_gefs[i][:110]
    mid=mid*100
    axes.plot(midbins[:110], mid, color=colors[i],label='Day '+str(i+1))
mid=pw_hist_obs[:110]
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
fig.savefig('YOUR_DIRECTORY/PW_hist_all_year_mixed.pdf',format='pdf',dpi=300, bbox_inches='tight')

# TP probability distribution
## GEFS 
tp_hist_gefs=np.zeros((16,24))
mid_tp_gefs=tp_GEFS[:,:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,None,:,ilon1:ilon2]
for i in np.arange(16):
    hist, edges = np.histogram(mid_tp_gefs[:,:,i].flatten(), bins=np.arange(0,50,2))
    total_num1=np.nansum(~np.isnan(mid_tp_gefs[:,:,i])) 
    tp_hist_gefs[i]=hist/total_num1
    
## Obs
#tp_hist_obs=np.zeros((20,24))

mid_tp_obs=tp_obs[:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,:,ilon1:ilon2]
hist, edges = np.histogram(mid_tp_obs.flatten(), bins=np.arange(0,50,2))
total_num2=np.nansum(~np.isnan(mid_tp_obs)) 
tp_hist_obs=hist/total_num2

####### plot
binsu=np.arange(0,50,2)#

fig, axes = plt.subplots(figsize=(8,4))
colors = plt.cm.jet(np.linspace(0,1,16))
midbins=(binsu[1:]+binsu[:-1])/2
mid=tp_hist_obs
mid=mid*100
axes.bar(midbins, mid,width=1.8, label='GPCP')

for i in np.array([0,1]):
    midbins=(binsu[1:]+binsu[:-1])/2
    mid=tp_hist_gefs[i]
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
fig.savefig('YOUR_DIRECTORY/TP_hist_all_year_mixed.pdf',format='pdf',dpi=300, bbox_inches='tight')

# PW versus TP
## GEFS 
pw_tp_gefs=np.zeros((16,129))
mid_pw_gefs=pw_GEFS[:,:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,None,:,ilon1:ilon2]
mid_tp_gefs=tp_GEFS[:,:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,None,:,ilon1:ilon2]
for i in np.arange(16):
    midpw=mid_pw_gefs[:,:,i]
    midtp=mid_tp_gefs[:,:,i]
    mid=binned_statistic(midpw[~np.isnan(midpw)], midtp[~np.isnan(midtp)],bins=np.arange(10,75,.5))
    pw_tp_gefs[i]=mid[0]

## Obs 
#pw_tp_obs=np.zeros(129)
mid_pw_obs=pw_obs[:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,:,ilon1:ilon2]
mid_tp_obs=tp_obs[:,:,:,ilon1:ilon2]+lsm_nan_20S20N[None,None,:,ilon1:ilon2]
midpw=mid_pw_obs
midtp=mid_tp_obs
mid=binned_statistic(midpw[~np.isnan(midtp)], midtp[~np.isnan(midtp)],bins=np.arange(10,75,.5))
pw_tp_obs=mid[0]

####### plot
binsu=np.arange(10,75,.5)#
fig, axes = plt.subplots(figsize=(8,4))
colors = plt.cm.jet(np.linspace(0,1,16))

for i in np.arange(0,16):
    midbins=(binsu[1:]+binsu[:-1])/2
    mid_pw_tp=pw_tp_gefs[i][:110]
    axes.plot(midbins[:110], mid_pw_tp, color=colors[i],label='Day '+str(i+1))
    
mid_pw_tp=pw_tp_obs[:110]
axes.plot(midbins[:110], mid_pw_tp, color='k',label='GPCP',linewidth=2)
axes.set_xlabel('PW [mm]',fontsize=16)
axes.set_ylabel('Precip [mm/day]',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
axes.set_title('(a) W. Pac: Precip vs. PW ',fontsize=18,loc='left',y=1.025)
plt.xticks(np.array([10,20,30,40,50,60,65]), ['10','20','30','40','50','60','65'],fontsize=13)  ; 
plt.legend(numpoints=1,loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
fig.tight_layout()
fig.savefig('YOUR_DIRECTORY/pw_tp_all_year_mixed.pdf',format='pdf',dpi=300, bbox_inches='tight')



Written by Jiacheng Ye; supervised by Professor Zhuo Wang




