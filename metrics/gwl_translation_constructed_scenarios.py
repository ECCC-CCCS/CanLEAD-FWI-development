import pandas as pd
import numpy as np
import glob
import xarray as xr
import os
import sys
from filepaths import fwipaths
import gc
import subprocess
from tqdm import tqdm
import datetime
from config_stats import take_climatological_mean_pseudo_rcps 
tracking_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

#%% Get target GWL for RCPs 2.6 and 4.5. Convert global GSAT by ensemble member to RCP-averaged GWL and GSAT, calc climo means. 
# GSAT calculated from area-weighted monthly average CanESM2 data. 

GWL = pd.DataFrame() # initialize dataframe to hold global warming levels

for rcp in ['rcp26','rcp45','rcp85']: # for each RCP available from CanESM2
    fl = f'{fwipaths.working_data}/GWL/{rcp}_annual_mean_GSAT_area_weighted_mon.csv' 
    gsat_fut = pd.read_csv(fl, index_col=0) # open area-weighted monthly average CanESM2 temperatures calculated from netCDF output (5 realizations)
    gsat_hist = pd.read_csv(f'{fwipaths.working_data}/GWL/historical_annual_mean_GSAT_area_weighted_mon.csv', index_col=0) # open historical period, which is the same for all RCPs
    gsat_df = pd.concat([gsat_hist, gsat_fut]) # merge historical and future periods
    baseline_temp = gsat_df.loc[1850:1900].mean() # calculate average temperature in the preindustrial (PI; 1950-1900) baseline period
    gwl = gsat_df - baseline_temp # calculate the global warming levels compared to the PI average
    GWL[rcp] = gwl.mean(axis=1) # take average across realizations (5 for each RCPs), then add to GWL dataframe
   
# take rolling mean of GWL and GSAT to get the climatological averages
target_GWL = GWL.rolling(30, center=True).mean().shift(-1)  # shift(-1) to duplicate xscen and IPCC
# from xscen: 'rolling defines the window as [n-10,n+9], but the the IPCC defines it as [n-9,n+10], where n is the center year'
# https://xscen.readthedocs.io/en/latest/_modules/xscen/extract.html#get_warming_level'      

#%% Define function to find nearest matching temperature

def find_nearest(base_rcp_timeseries, dT, window_length, return_option='window'):
    '''
    Return a window or central index, in years, representing the time at which a specific 
    GWL ('of the GWL, centered around the nearest value to the base rcp sample array
    
    Parameters
    ----------
    base_rcp_timeseries : pandas series
        Pandas series of annual global warming level with respect to your specified baseline, with a row for each year, where the index is the year.
    dT : float
       Warming level, e.g., 3 for a global warming level of +3 degree Celsius with respect to your specified baseline.
    window_length : int
       Size of the rolling window to compute the warming level, in years.
    return_option : string, optional
        Whether to return the central index of the sample ('central'), or the range (in years) for the climatoligical window ('window').
        The default is 'window'.

    Returns
    -------
    Object.
        Returns the label (years, if input series is correctly formatted) of the central index of 
        the window ('central') or a tuple of the range of the climatological window ('window').

    '''
    
    # shift(-1) to duplicate xscen and IPCC, from xscen: 'rolling defines the window
    # as [n-10,n+9], but the the IPCC defines it as [n-9,n+10], where n is the center year'
    base_rcp_climatologies = base_rcp_timeseries.rolling(window_length, center=True, min_periods=1).mean().shift(-1)        
    delta = base_rcp_climatologies - dT # get the difference between the GWL requested and the base_RCP warming levels
    central_index = np.argmin(abs(delta)) # find the minimum value; the first (earliest) occurrence is returned if multiple are the same
    
    window_half_length = np.floor(window_length/2) #e.g. a 30 or 31 year window -> 15; 10 year windw -> 5
    # If an even climo period requested, pad minimum index by 1 (e.g. 1971-2000, for a 30 year period). Else, include minimum value.
    if window_length % 2 == 1:
        minval=int(max(0,
                       central_index-window_half_length)) 
    else:
        minval=int(max(0,
                       central_index-window_half_length+1))  
    maxval=int(min(len(delta)-1,
                   central_index+window_half_length)) #-1:because of Python zero-based indexing
    
    assert maxval - minval == window_length-1, f'Window length = {maxval-minval+1}, not {window_length}. Window: {delta.index[minval]}, {delta.index[maxval]}'
    # Weird stuff because of python zero based indexing, but slice() methodology in xr.sel uses inclusive boundaries for time

    if return_option=='central':
        return delta.index[central_index] # return central sample
    elif return_option=='window':
        return (str(delta.index[minval]), str(delta.index[maxval])) # return index range for whole window

def add_attrs(ds):
    ds['period'].attrs['short_name'] = 'climatological_period'
    ds['period'].attrs['long_name'] = f'Outputs at the level of global warming reached in the specified climatological period in {rcp.upper()}.'
    ds['period'].attrs['description'] = f'Climate change impacts represent the level of global warming reached in the specified climatological period in {rcp.upper()}, '\
                                        +'translated from the CanLEAD-FWI RCP8.5 ensemble.'  
    ds.attrs['history'] = f'Generated by {sys.argv[0]}'
    ds.attrs['git_id'] = tracking_id
    ds.attrs['git_repo'] = 'https://github.com/ECCC-CCCS/CanLEAD-FWI-v1/'
    ds.attrs['creation_date'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    if rcp == 'rcp26':
        ds.attrs['rcp'] = 'Constructed RCP2.6'
    elif rcp == 'rcp45':
        ds.attrs['rcp'] = 'Constructed RCP4.5'
    return ds 

#%%
version = f'CanLEAD-FWI-{sys.argv[1]}-v1'

# Get CanESM2-LE global warming levels (temp anomaly from PI) for RCP85, available with CanLEAD ensemble
tasAmon = pd.read_csv(f'{fwipaths.input_data}/CanLEAD/tasAnom_PI_Ayr_CanESM2_historical-rcp85_185001-210012.csv', index_col=1) # open CSV, 55 realizations of temp, each as a separate column
# Drop 5 CMIP5 runs, separate from the CanESM2 LE
tasAmon = tasAmon.drop(columns=['PI_1850-1900=13.66_degC', 'CanESM2_CMIP5_r1i1p1', 'CanESM2_CMIP5_r2i1p1', 'CanESM2_CMIP5_r3i1p1',
                                'CanESM2_CMIP5_r4i1p1','CanESM2_CMIP5_r5i1p1'])
# rename columns to expected format below
tasAmon = tasAmon.rename(columns={col_old: col_old[11:17] for col_old in tasAmon.columns})

tas_baseline_period='1850-1900'
tas_window=30

# get target RCP GWL, averaged over 5 available CanESM2 RCPs in previous worflow
dec_gwl = target_GWL.loc[np.arange(1965, 2090, 10), ['rcp26', 'rcp45']] # 30-year mean rolling values are saved by the center year, so select mid-decadal years every decade
dec_gwl['period'] = [f'{i-14}-{i+15}' for i in dec_gwl.index.values] # then add column with more meaningful window labels, rather than single years 

fut_gwl = dec_gwl.loc[np.arange(1995, 2090, 10)] # only want period from 2001-2030 ... 2071-2100
hist_gwl = dec_gwl.loc[np.arange(1965, 1990, 10)] # only want period from 2001-2030 ... 2071-2100

# Canada mask, excluding northern Arctic
final_mask = xr.open_dataset(f'{fwipaths.input_data}/CanLEAD_FWI_final_mask.nc')['CanLEAD_FWI_mask'] 

### pull GWL translation time slices and take climo mean ###

test_stat = sys.argv[2]

for rcp in tqdm(['rcp26', 'rcp45']): # for each target RCP
        
    if test_stat == 'QS-DEC_mean':
        freq = 'seasonal'
    elif test_stat == 'MS_mean':
        freq = 'monthly'
    else:
        freq = 'annual'
    
    # create output directory if it doesn't exist
    outpath = f'{fwipaths.output_data}{version}/summary_stats/constructed_{rcp.upper()}/{test_stat}/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    # get filenames for annual data of all 50 realizations under RCP8.5
    afls = glob.glob(f'{fwipaths.output_data}{version}/summary_stats/RCP85/{test_stat}/*{test_stat}.nc')
    assert len(afls) == 50, f'Number of files does not equal 50: {len(afls)}'
    
    for fl in tqdm(afls): # loop over CANLEAD-FWI RCP8.5 output data
    
        annual_alldat = xr.open_dataset(fl).chunk({'time':-1, 'lat':10, 'lon':10})
        real = annual_alldat.realization.values[0][:6] # get realization label, shortened to have common length e.g. r1_r4i or r2_r10
        
        if 'quantile' in annual_alldat.coords:
            annual_alldat = annual_alldat.rename({'quantile': 'annual_quantiles'})
        
        data_dict = {}
        
        # add historical period data to dictionary. This is unchanged in time from RCP85
        # info on GWL and base RCP period is added so that it can later be merged with future data
        for period, midyear in zip(hist_gwl['period'], hist_gwl.index): # loop over decadal periods
            wl = tasAmon[real].rolling(30, center=True).mean().shift(-1).loc[midyear] # find the GWL at the 30 year window x realization of interest (period is represented by midyear) 
            base_rcp_data = annual_alldat.sel(time=slice(period.split('-')[0], period.split('-')[1])) # select the data for the source period 
            
            single_warming_level = take_climatological_mean_pseudo_rcps(base_rcp_data, freq) # take climo mean. this function will automate creations of seasonal dim, if needed as indicated in "freq"
            single_warming_level = single_warming_level.assign_coords(period=period).expand_dims('period') # assign period coordinate
            # assign warming level coordinates associated with the target period (e.g., GWL and source RCP period)
            # (unchanged in time from RCP85 for historical period, added so hist and future datasets can later be merged)
            single_warming_level = single_warming_level.assign_coords({'warming_level': ("period", [f'GWL:{wl:.2f}Cvs{tas_baseline_period}']),
                                                                       'source_rcp_period': ("period", [f'{period.split("-")[0]}-{period.split("-")[1]}']), # add source RCP period (string)
                                                                       'source_rcp_period_first_year': ("period", [period.split('-')[0]]) # add source RCP period (float), this will allow easier stats (avg,etc) later across reals
                                                                       })
            data_dict[period] = single_warming_level # save output to dictionary
            
        # get data for constructed RCPs for non-historical forcings (1981-2010 to 2071-2100)
        for wl, period in zip(fut_gwl[rcp], fut_gwl['period']): # loop over future decadal periods
            base_rcp_period = find_nearest(tasAmon[real], wl, tas_window, return_option='window') # use find nearest function to get RCP8.5 period that is closeset to target GWL
            base_rcp_data = annual_alldat.sel(time=slice(base_rcp_period[0], base_rcp_period[1])) # select the data from the source period
            
            single_warming_level = take_climatological_mean_pseudo_rcps(base_rcp_data, freq) # take climo mean. this function will automate creations of seasonal dim, if needed as indicated in "freq"
            single_warming_level = single_warming_level.assign_coords(period=period).expand_dims('period') # assign period coordinate
            # assign warming level coordinates associated with the target period (e.g., GWL and source RCP period)
            single_warming_level = single_warming_level.assign_coords({'warming_level': ("period", [f'GWL:{wl:.2f}Cvs{tas_baseline_period}']),
                                                                       'source_rcp_period': ("period", [f'{base_rcp_period[0]}-{base_rcp_period[1]}']), # add source RCP period (string)
                                                                       'source_rcp_period_first_year': ("period", [base_rcp_period[0]]) # add source RCP period (float), this will allow easier stats (avg,etc) later across reals
                                                                       })
            data_dict[period] = single_warming_level # save output to dictionary
                  
        out = xr.merge(data_dict.values()) # xarray merge will automatically merge along time ("period" dim)
        out = add_attrs(out) # add attrs defined in above func
        out = out.where(final_mask==100) # mask with Canadian boundaries and ecozone mask
        out.to_netcdf(f'{outpath}/{annual_alldat.realization.values[0]}_{rcp}_{version}_{test_stat}_30yr_mean.nc') # save
    
        del([annual_alldat, base_rcp_data, single_warming_level, out])
        gc.collect()