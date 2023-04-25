"""
Determine unknown noon parameters (offset of tmin and tmax from sunrise and solar noon, respectively);
required to estimate noontime values of FWI inputs.
Determine values from CanRCM4 hourly temperature data, later regridded to CanLEAD grid using regrid_diurnal_estimates.py.
"""

import xarray as xr
import numpy as np
import gc
import glob
from scipy import stats

import sys
import os
import subprocess
sys.path.append(os.path.expanduser('~/fwi_updates/CanLEAD-FWI-v1/'))
from filepaths import fwipaths 
from config import canada_bounds_rotated_index
tracking_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip() 

#%% Equations 

# Create custom functions to find the hour of tasmax and tasmin
def hourmax(da): 
    '''
    Determine the hour of daily maximum temperature (tasmax). Input temperature data must be hourly, and grouped daily using groupby.
    The function returns the index from 0 to 23 of taxmax within each 24 hour period, which indicates the time, in hours, from midnight. 
    
    Parameters
    ----------
    da : xr dataarray of tas (temperature) with daily grouping

    Returns
    ----------
    out : Time (index) of maximum temperature, in hours from midnight 
    '''  
    out = da.argmax(dim='time') # return index (time) of the maximum temperature value
    return out

def hourmin(da):
    '''
    Determine the hour of daily minimum temperature (tasmin). Input temperature data must be hourly, and grouped daily using groupby.
    The function returns the index from 0 to 23 of tasmin within each 24 hour period, which indicates the time, in hours, from midnight. 
    
    Parameters
    ----------
    da : xr dataarray of tas (temperature) with daily grouping

    Returns
    ----------
    out : Time (index) of tasmin temperature, in hours from midnight 
    '''  
    out = da.argmin(dim='time') # return index (time) of the minimum temperature value
    return out

def get_circmean(da, min_max, dawn_noon): 
    '''
    Calculate the 'day of year' circular mean of the timing of maximum or minimum temperature. Add attributes.
    
    Parameters
    ----------
    da : xr dataarray of time of tmax or tmin
    name : name to add to output
    min_max : 'minimum' or 'maximum', indicating if input is time of min or max temperature

    Returns
    ----------
    mn : xr dataset containing the circular mean of the timing of tmax or tmin, by day of year 
    '''  
    mn = xr.apply_ufunc(stats.circmean, da.groupby('time.month'), # take circular mean across 30 years by day of year, max value = period = 24, min value = 0
                        kwargs=dict(high=24, low=0),
                        input_core_dims=[['time']], # broadcast across all times except time, that is, take mean over time dim (grouped by day of year)
                        output_dtypes=[da.dtype],
                        vectorize=True, # loop over non-core dims
                        dask='parallelized') # apply_ufunc to speed up via dask
                        
    mn.attrs['description'] = f'Day of year climatological ({start_yr}, {end_yr}) circular mean of the offset of daily {min_max} temperature from {dawn_noon} in decimal hours'
    mn.attrs['name'] = f'Offset of {min_max} temperature from {dawn_noon}'
    mn.attrs['units'] = 'Decimal hours'
    return mn.rename(f'h{min_max[:3]}_offset').to_dataset()

def mask_add_ens(ds): # clip to Canada domain, add realization as dimension
    ds = ds.isel(**canada_bounds_rotated_index) # clip to Canada domain
    ds = ds.assign_coords(nens = 'r' + ds.attrs['experiment_id'][-1] + 'r' + ds.attrs['realization'])
    ds = ds.expand_dims('nens') # add realization as dimension, to allow concat into one file later
    return ds

#%% Determine daily timing (index) of max and min temperature for each day of year using CanRCM4 hourly data
# For use in noontime temperature estimates based on Beck and Trevitt (1989). For additional details see calculate_noon_rh_t.py.

ens = sys.argv[1] # get ensemble group number from run file: [1,2,3,4,5] 

for real in [8,9,10]: # get number, only 3 have hourly data: [8,9,10] 
       
    start_yr, end_yr = 1971, 2000
    import_years = range(start_yr, end_yr+1) # want to import all files that contain these years
      
    # import hourly temperature data for specified realization
    # get filenames that contain desired timeframe (years in import_years) for realization
    flnms = []
    for yr in import_years: 
        flnms = flnms + glob.glob(f'{fwipaths.input_data}CanRCM4/NAM-44_CCCma-CanESM2_historical-r{ens}/1hr/atmos/tas/r{real}i2p1/*{yr}*nc') 
    tas = xr.open_mfdataset(np.unique(flnms), preprocess=mask_add_ens, # geog_mask_ens_dims used to assign ens+real as a dimension and clip to Canada domain
                            chunks={'time':-1, 'rlat':10, 'rlon': 10}).sel(time=slice(str(start_yr), str(end_yr))) 
    tas = tas.resample(time='1H').nearest() # fix times off by a couple seconds
    
    # time of sunrise and noon in UTC, generated in utc_sunrise_noon.py
    noon_sunrise = xr.open_dataset(f'{fwipaths.working_data}CanRCM4_utc_sunrise_solar_noon.nc',
                                   chunks={'time':-1, 'rlat':10, 'rlon': 10}).sel(time=slice(str(start_yr), str(end_yr))) 
    noon_sunrise = noon_sunrise.resample(time='1D').nearest() # fix times off by a couple seconds
    
    # find timing of maximum temperature, typically after noontime 
    hmax = tas['tas'].resample(time='24H').map(hourmax) # find timing (index) of max temp in every 24 hr period, indexed from 0 to 23 in UTC (timezone of tas)
    hmax_offset = hmax - noon_sunrise['solar_noon_utc'] # get offset of hmax from solar noon  
    
    hmax_offset_out = get_circmean(hmax_offset.chunk({'time':-1, 'rlat':10, 'rlon': 10}),
                                   'maximum',
                                   'solar noon') # take sum stats and add attrs (see eqn above)
    ''' Negative values of hmax_offset could occur either when:
             (1) tmax occurs in a different (next) day than noon due to UTC timezone usage (e.g., 3:00 - 23:00 = -20, but it's a +4 hour offset)
             (2) tmax occurred before noon (due to front passing through, etc.) (e.g., 22:00 - 23:00 = -1, a true -1 hour offset)
    These will be handled when taking the circular mean during get_circmean, which will 'convert' negative values to corresponding 
    values ranging from 0 to 24. For example, -20 is equivalent to 4 when taking a circular mean; -1 is equivalent to 23 when taking a circular mean.
    
    Typically, tmin occurs near sunrise, while tmax occurs a few hours after noon. Temperature relationships at high latitudes (with solar day/night) 
    or over ocean areas cannot be assumed to always hold this relationship. For this reason we set tnoon to tmax above the Arctic circle, 
    and do not calculate tnoon (or fire weather) over ocean areas. 
    
    The same relationship stands for hmin_offset and sunrise, below. Since tmin generally occurs near sunrise, we use sunrise as the time of tmin
    in noontime temperature estimates. The following determination of tmin offsets is used for testing only.
    ''' 
   
    del([hmax, hmax_offset])
    gc.collect()
    
    # find offset of minimum temperature from sunrise, typically after sunrise
    hmin =  tas.tas.resample(time='24H').map(hourmin) # find timing (index) of min temp in every 24 hr period, indexed from 0 to 23 in UTC midnight (timezone of tas)
    hmin_offset = hmin - noon_sunrise['sunrise_utc'] # get offset of hmax from solar noon 
    hmin_offset_out = get_circmean(hmin_offset.chunk({'time':-1, 'rlat':10, 'rlon': 10}),
                                   'minimum',
                                   'sunrise')
    
    del([tas, hmin, hmin_offset])
    gc.collect()
    
    # add script attrs, save
    t_out = xr.merge([hmax_offset_out, hmin_offset_out])
    t_out.attrs['git_id'] = tracking_id
    t_out.attrs['git_repo'] = 'https://github.com/ECCC-CCCS/CanLEAD-FWI-v1/'
    t_out.attrs['history'] = f"Generated by {os.path.basename(sys.argv[0])}"  # get script name from run file
    encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 4} for var in t_out.data_vars} # save as float32 and compress to save space
    if not os.path.exists(f'{fwipaths.working_data}offsets_tmin_tmax/'):
        os.makedirs(f'{fwipaths.working_data}offsets_tmin_tmax/')
    t_out.to_netcdf(f'{fwipaths.working_data}offsets_tmin_tmax/offsets_tmin_tmax_month_{start_yr}_{end_yr}_r{ens}r{real}i2p1.nc')
    
    del([hmax_offset_out, hmin_offset_out, t_out])
    gc.collect() # free up space
                     
