"""
Find timing (time offset in decimal hours) of sunrise and solar noon in UTC for each grid cell,
for both CanLEAD and CanRCM4 grids from 1950-2100.

"""
#%%

import xarray as xr
from pvlib.solarposition import sun_rise_set_transit_spa
import pvlib
import datetime 
import gc
import subprocess
import sys
import os
sys.path.append(os.path.expanduser('~/fwi_updates/CanLEAD-FWI-v1/'))
from filepaths import fwipaths 
from config import canada_bounds, canada_bounds_rotated_index
import glob
tracking_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip() 

def geog_mask(ds):
    ds = ds.isel(**canada_bounds_rotated_index) # clip to Canada domain
    return ds

#%% Define function to get sunrise and noon offset from midnight using PVlib package

def get_sunrise_noon(lat, lon, time_array=None): 
    '''
    Using pvlib package, get date and time of sunrise and solar noon, localized to UTC.
    Subtract daily UTC midnight from time of sunrise and noon to get the "offset" from UTC midnight
    (That is, time in decimal hours of sunrise and noon from UTC midnight.)

    Designed to be used to xarray.apply_ufunc, and vectorized over lat and lon.
        
    Parameters
    ----------
    lat : dataarray or dataset index of latitudes
    lon : dataarray or dataset index of longitudes
    time_array : CF timeindex of dates for which you want sunrise and noon.

    Returns
    ----------
    sunrise, noon:   xarray dataarrays of time (or 'offset') in decimal hours of sunrise and noon from UTC midnight
    '''      
    ## get time of input data, convert to midnight timestamp  
    times_utc = time_array.to_datetimeindex().tz_localize('UTC') # get time index of input dataset and assign appropriate timezone (UTC)
    if (times_utc.hour == 12).all(): # if the timestamp is noon, create a 'midnight_utc' time index but subtracting 12 hours from index
        utc_midnight = times_utc - datetime.timedelta(hours = 12)
 
    # get df of sunrise, 'transit' (equivalent to solar noon), and sunset in UTC using PVlib function
    df = sun_rise_set_transit_spa(utc_midnight, # specify UTC to return in same timezone as CanRCM4
                                  latitude=lat, longitude=lon)
    df = df.drop('sunset', axis=1)
    
    timedeltas = df.subtract(df.index, axis=0) # subtract UTC midnight (the index) to get timedelta object (hour of sunrise and noon) instead of date objects
    
    # convert timedelta objects to decimal hours after (UTC) midnight
    sunrise = timedeltas.sunrise.dt.total_seconds() / (60 * 60)
    noon = timedeltas.transit.dt.total_seconds() / (60 * 60) 
    
    return sunrise, noon  # return hour of sunrise and noon from UTC midnight, will return NaN for polar night and polar day

#%% Load data, apply sunrise and noon function, add_attrs, save

for key in ['CanLEAD', 'CanRCM4']: # for both CanLEAD and CanRCM4 grid, do:

    # load data    
    if key == 'CanRCM4':
        inpath = fwipaths.input_data + 'CanRCM4/NAM-44_CCCma-CanESM2_historical-r1/day/atmos/tas/r1i1p1/'
        fls = glob.glob(inpath + '*nc') # get all r1i1p1 files, 1950-2100
        ds = xr.open_mfdataset(fls, preprocess=geog_mask).chunk({'time':-1, 'rlat':10, 'rlon':10})
        
    elif key == 'CanLEAD':
        ds = xr.open_dataset(glob.glob(f"{fwipaths.input_data}CanLEAD/CanRCM4-S14FD-MBCn/r1_r1i1p1/prAdjust*nc")[0]).sel(**canada_bounds).chunk({'time':-1, 'lat':10, 'lon':10})
    
    input_times = ds.indexes['time'] # get time dimension, for use in function and to re-add coords post-processing
    sunrise_utc, solar_noon_utc = xr.apply_ufunc(get_sunrise_noon, ds.lat, ds.lon, # input lat and lon (on rlat-rlon dimensions or lat-lon dimensions)
                                                 input_core_dims=[[],[]], # broadcast over all rlon, rlat (apply func to each rlat-rlon or lat-lon pair)
                                                 output_core_dims=[['time'], ['time']], # add new dim 'time' to output (inputs have only rlat-rlon or lat-lon dims)
                                                 output_sizes={'time': input_times.size},
                                                 kwargs = {'time_array': input_times}, # ds.time index array input as kwarg, not broadcast
                                                 vectorize=True,
                                                 dask='parallelized')
    
    # Re-add time coords, which were output as a numbered index
    sunrise_utc['time'] = input_times
    solar_noon_utc['time'] = input_times
    
    # Merge data arrays, add attrs, save
    time_sunrise_noon = xr.merge([solar_noon_utc.rename('solar_noon_utc'), 
                                  sunrise_utc.rename('sunrise_utc')])
    
    ## Add attrs to variables
    for factor in ['sunrise', 'solar_noon']:
        time_sunrise_noon[f'{factor}_utc'].attrs['description'] = f'Time, in decimal hours, of local {factor} from UTC midnight (00:00)'
        time_sunrise_noon[f'{factor}_utc'].attrs['full_name'] = f'Time of local {factor} in UTC'
        time_sunrise_noon[f'{factor}_utc'].attrs['units'] = 'Decimal hours from UTC midnight (00:00)'
    
    for factor in ['time', 'lat', 'lon']: # add attrs from input dataset
        time_sunrise_noon[factor].attrs = ds[factor].attrs
       
    encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 4} for var in time_sunrise_noon.data_vars} # save sunrise_offset and noon_offset as float32 and compress to save space. Set encoding before adding rotated_pole as variable
    
    if key == 'CanRCM4': # if CanRCM4, add rotated_pole attrs
        for factor in ['rlat', 'rlon']:
            time_sunrise_noon[factor].attrs = ds[factor].attrs
        time_sunrise_noon['rotated_pole'] = ds['rotated_pole'] # add rotated_pole info as var
         
    ## Add file attrs
    time_sunrise_noon.attrs['git_id'] = tracking_id
    time_sunrise_noon.attrs['pvlib_info'] = f'pvlib {pvlib.__version__} solarposition.sun_rise_set_transit_spa'
    time_sunrise_noon.attrs['description'] = 'Time, in decimal hours, of noon and sunrise from UTC midnight (00:00)'
    time_sunrise_noon.attrs['history'] = f"Generated by {os.path.basename(sys.argv[0])}"
    
    ## Save
    time_sunrise_noon.to_netcdf(f'{fwipaths.working_data}{key}_utc_sunrise_solar_noon.nc', encoding=encoding)
    
    del([ds, sunrise_utc, solar_noon_utc, time_sunrise_noon, input_times])
    gc.collect() # free up space
