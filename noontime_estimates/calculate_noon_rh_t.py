'''
Calculate noontime estimated values of temperature and RH from daily maximum and minimum temperature and daily average RH. 
'''

#%% Set up
import numpy as np
import xarray as xr
import gc
import sys
import os
import math
import subprocess
sys.path.append(os.path.expanduser('~/fwi_updates/CanLEAD-FWI-v1/'))
import xclim as xc # xclim functions for unit conversions
from xclim.indices import saturation_vapor_pressure
from filepaths import fwipaths
from config import canada_bounds
import datetime
tracking_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip() 

#%% Noontime temperature and RH equations

def tas_noon(tmin, tmax, h_sunrise, h_noon, hmax_offset, hmin_offset=0): 
    '''
    Calculate temperature at time t, by fitting a sine curve between tn and tx. Based on Beck and Trevitt (1989). 
    
    "tn" and "tx" expect the time of minimum and maximum temperature, respectively, in UTC. Therefore, they 
    will not align with local timezone values (e.g., solar noon in UTC will not be at 12h00 UTC).
        
    tx > tn to ensure expected behaviour (that is, noon must be after sunrise, see Beck and Trevitt 1989). 
    Test added to ensure UTC sunrise is not before UTC solar noon, due to time zone shifts with circular
    data.
    
    "t" represents the time, in UTC hours, at which temperature output is desired. This is UTC solar noon 
    for our purpose.
    
    Beck, J. A., & Trevitt, A. C. F. 1989. Forecasting diurnal variations in meteorological parameters for predicting
    fire behaviour. Canadian Journal of Forest Research, 19(6), 791-797. https://cdnsciencepub.com/doi/abs/10.1139/x89-120
    
    Parameters
    ----------
    tmin : minimum temperature, in degrees Celsius or Kelvin
    tmax : maximum temperature, in same units as tmin
    h_sunrise : time of sunrise, in hours after UTC midnight
    h_noon : time of solar noon, in hours after UTC midnight
    hmax_offset : offset of maximum temperature from solar noon, in decimal hours (referred to as 'beta' in Beck and Trevitt 1989)
    hmin_offset : offset of minimum temperature from sunrise, in decimal hours (referred to as 'beta' in Beck and Trevitt 1989), here assumed to be zero
        
    Returns
    -------
    tnoon : temperature at time t (local solar noon), in the same units as tmin and tmax
    '''
    # make some adjustments to reflect circular nature of time data 
    if h_sunrise > h_noon: # check if time sunrise > solar noon due to UTC adjustment
        h_noon = h_noon + 24 ## if yes, shift time of max temperature to the following day by adding 24 hours (e.g., h_sunrise = 23 UTC, h_noon = 5 UTC, shift "noon" to next day by hnoon = 5 + 24 = 29)
    # create sine function of temperature between hmin and hmax, solve for temperature at time t
    tn = h_sunrise + hmin_offset
    tx = h_noon + hmax_offset
    # t : time, in hours, at which temperature output is desired, in UTC (UTC solar noon in our case)
    t = h_noon
    ft = (t - tn) / (tx - tn)
    tnoon = tmin + (tmax - tmin)*np.sin(ft*math.pi/2)
    return tnoon
tas_noon = np.vectorize(tas_noon)

def RH_noon_wrapper(tnoon, tasmin, tasmax, hurs):
    '''
    Calculate approximate noontime relative humidity using daily minimum, maximum and noontime temperature, as well as 
    daily average relative humidity. Based on approximation of Allen et al (1998, Ch 3), and assumption that vapour 
    pressure remains approximately constant in any 24H period (Beck and Trevitt 1989). Saturation vapour pressure is determined
    using World Meteorological Organization (2008) method, valid from -45 degC to 60 degC. 

    References: 
        
    Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. 1998. Crop evapotranspiration-Guidelines for
    computing crop water requirements-FAO Irrigation and drainage paper 56. FAO, Rome, 300(9), 
    D05109. https://www.fao.org/3/x0490e/x0490e07.htm#calculation%20procedures 
    
    Taylor, S., St-Amant, R., Regniere, J., and Spears, J. 2010. Stochastic Simulation of Daily Forest Fire Weather
    for Monthly Climate Normals. Internal report. Natural Resources Canada.
    
    [WMO] World Meteorological Organization. Guide to meteorological instruments and methods of observation. 
    World Meteorological Organization, Geneva, Switzerland, 2008. ISBN 978-92-63-10008-5. OCLC: 288915903.

    Parameters
    ----------
    tnoon : xarray dataarray
        Daily noontime temperature in degrees Celsius.
    tasmin : xarray dataarray
        Daily minimum temperature in degrees Celsius.
    tasmax : xarray dataarray
        Daily maximum temperature in degrees Celsius.
    hurs : xarray dataarray
        Daily mean relative humidity, as a percentage

    Returns
    -------
    RHn : xarray dataarray
        Approximate relative humidity at noon, as a percentage.

    '''
    # Determine RHnoon based on approximation of Allen et al (1998, Ch 3) 
    def RH_noon(svpnoon, svptmn, svptmx, RH):
        vp_avg = RH/100 * np.mean((svptmn, svptmx)) # Determine mean vapour pressure (vp) from mean RH and mean of svp@tmax and svp@tmin
        RHnoon = vp_avg/svpnoon * 100 # Assume vapour pressure remains approx contstant within 24 hr period, vp_noon = vp_avg. Use to find RHnoon. 
        return RHnoon
    
    assert tasmin.attrs['units'] == '°C', f'tasminAdjust units must be in degC, not {tasmin.attrs["units"]}'
    assert tasmax.attrs['units'] == '°C', f'tasmaxAdjust units must be in degC, not {tasmax.attrs["units"]}'
    assert tnoon.attrs['units'] == '°C', f'tnoon must be in degC, not {tnoon.attrs["units"]}'
    assert hurs.attrs['units'] in ['%', 'pct', 'percent'], f'hursAdjust units are {hurs.attrs["units"]}, not %' # check units are in percent
    hurs = xr.where(hurs > 100, 100, hurs) # fix CanLEAD hursAdjust > 100
    
    # Find saturation vapour pressure at tmin, tmax, and tnoon using xclim. xr.map_blocks to speed up computation
    svp_tmx = xr.map_blocks(saturation_vapor_pressure, tasmax, kwargs={"method": "wmo08"}) # Output in Pa. Valid from -45C to 60C
    svp_tmn = xr.map_blocks(saturation_vapor_pressure, tasmin, kwargs={"method": "wmo08"})
    svp_noon = xr.map_blocks(saturation_vapor_pressure, tnoon, kwargs={"method": "wmo08"})
    assert svp_tmx.attrs['units'] == svp_tmn.attrs['units'] == svp_noon.attrs['units'], 'Saturation vapour pressure units do not match'
    
    RHn = xr.apply_ufunc(RH_noon, svp_noon, svp_tmn, svp_tmx, hurs, # broadcast apply RH_noon func over all lat-lon-time dimensions
                         dask="parallelized", vectorize=True).rename("RH_noon")
    RHn = xr.where(RHn > 100, 100, RHn) # Note: separate test script used to check occurence of this (which is mostly over water)
    return RHn

#%% Set up job

version = sys.argv[2] # EWEMBI or S14FD
InputDataDir = f'{fwipaths.input_data}CanLEAD/CanRCM4-{version}-MBCn/'
OutputDataDir = f'{fwipaths.working_data}noontime/' 
if not os.path.exists(OutputDataDir): 
    os.makedirs(OutputDataDir)
    
## Add CanLEAD filename components here to make read-in more streamlined
fname1 = "_NAM-44i_CCCma-CanESM2_rcp85_"
fname2 = f"_CCCma-CanRCM4_r2_ECCC-MBCn-{version}-1981-2010_day_19500101-21001231.nc" 
   
## Import estimated offsets of hmax, and time of local solar noon in UTC. Predetermined from CanRCM4 in utc_sunrise_noon.py and diurnal_estimates.py

chunks = {"time": 2555, 'lat': 10, 'lon': 10} # define chunks to use on data import, to speed up calculations with dask
sunrise_noon = xr.open_dataset(f'{fwipaths.working_data}CanLEAD_utc_sunrise_solar_noon.nc', chunks=chunks).sel(**canada_bounds) # get sunrise and solar noon
temp_offsets = xr.open_dataset(f'{fwipaths.working_data}CanLEAD_offsets_tmin_tmax_1971_2000_all_realization_circmean_1950_2100_daily.nc').sel(**canada_bounds) # time of tmin and tmax, from 0 to 23

# Get ensemble group from job file. For each realization in group, calculate noontime estimates 
j = sys.argv[1]
for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]: 
    # define ensemble number
    if m in [8,9,10]: nens = f'r{j}_r{m}i2p1' 
    else: nens = f'r{j}_r{m}i1p1'
     
    ## import CanLEAD-CanRCM4 relative humidity, maximum temperature, and minimum temperature
    
    flnm = InputDataDir + nens + "/tasmaxAdjust" + fname1 + nens + fname2 
    tasmaxAdjust = xr.open_dataset(flnm, chunks=chunks).sel(**canada_bounds)['tasmaxAdjust'] # load data using subset and chunking specified above
    tasmaxAdjust = xc.core.units.convert_units_to(tasmaxAdjust, 'degC') # convert units to degrees Celsius
        
    flnm = InputDataDir + nens + "/tasminAdjust" + fname1 + nens + fname2 
    tasminAdjust = xr.open_dataset(flnm, chunks=chunks).sel(**canada_bounds)['tasminAdjust'] 
    tasminAdjust = xc.core.units.convert_units_to(tasminAdjust, 'degC') # convert units to degrees Celsius
            
    flnm = InputDataDir + nens + "/hursAdjust" + fname1 + nens + fname2 
    hursAdjust = xr.open_dataset(flnm, chunks=chunks).sel(**canada_bounds)
    
    ## create func to add default script attributes and then save after calculation
    
    # Generic administrative attributes to add to output datasets
    attrs_to_add = dict(## Admin attrs ##
                        Conventions = "CF-1.8",
                        institution = "Canadian Centre for Climate Services (CCCS)",
                        institute_id = "CCCS",
                        contact = "ccsc-cccs@ec.gc.ca", 
                        domain = 'Canada',
                        creation_date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                        product = "fire-weather-projection-inputs", # note difference from CFFWIS output product name
                        project_id = "CanLEAD-FWI", # CanLEAD
                        product_version = f'CanLEAD-FWI-{version}-v1',
                        title = f'Noontime inputs for Canadian Forest Fire Weather Index System (CFFWIS) projections based on CanLEAD-CanRCM4-{version}', # note difference from CFFWIS output title 
                        history = f"Generated by {os.path.basename(sys.argv[0])}. xclim version: {xc.__version__}",  ## LV: Need to add new git repo info here (link to repo, hash tag)
                        git_id = tracking_id,
                        git_repo = 'https://github.com/ECCC-CCCS/CanLEAD-FWI-v1/',
                        CORDEX_grid = 'NAM-44i'
                        )
    # add some CanLEAD attrs, renaming by prepending 'CanLEAD-CanRCM'
    attrs_to_rename = ['driving_model_id', 'driving_experiment_name',  'driving_model_ensemble_member', 'realization',
                       'initialization_method', 'physics_version', 'forcing', 'model_id','rcm_version_id', 'CCCma_runid', 'references', 'institution', 'institute_id',
                       'experiment_id', 'experiment', 'bc_method', 'bc_method_id', 'bc_observation',  'bc_info', 'bc_observation_id', 'bc_period'] 
     
    # Create func to add default script attributes, and save
    def add_attrs(ds, var, attrs_to_add=attrs_to_add, attrs_to_rename=attrs_to_rename):
        # Copy attrs to time, lat and lon
        ds['time_bnds'] = hursAdjust['time_bnds'] 
        for dim in ['time', 'lat', 'lon', 'time_bnds']:
            ds[dim].attrs = hursAdjust[dim].attrs
        # Add admin attrs defined above    
        for attr_name, attr_val in attrs_to_add.items():
            ds.attrs[attr_name] = attr_val        
        for attr_name in attrs_to_rename:  
            ds.attrs['CanLEAD_CanRCM4_' + attr_name] = hursAdjust.attrs[attr_name]                        
        # Set encoding and save
        encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 2}, # set compression specifications. LV: Can modify to enhance compression, or add lossy compression
                    'time_bnds': {'_FillValue': None, 'dtype': 'float64'} } 
        ds = ds.transpose("time", "lat", "lon", "bnds") # reorder dims to match CF-preferred ordering to T-Y-X
        ds.to_netcdf(f"{OutputDataDir}{nens}_{var}_1950_2100_{version}.nc", encoding=encoding) 
        
    ## calculate approx temperature at noon using function 'tas_noon' detailed above
    
    # apply tas_noon using groupby('time.month') for all inputs other than temp_offsets, which has month instead time dim
    tnoon = xr.apply_ufunc(tas_noon, 
                           tasminAdjust, tasmaxAdjust,  # tmin, tmax
                           sunrise_noon.sunrise_utc, # h_sunrise
                           sunrise_noon.solar_noon_utc, #  h_noon
                           temp_offsets.hmax_offset, # hmax_offset, hmin_offset=0 as per equation default
                           dask='parallelized').rename("tnoon").to_dataset()
    tnoon = xr.where(tnoon.lat >= 65, tasmaxAdjust, tnoon) # above Arctic circle, set tnoon to tasmax. As "sunrise" does not exist during some times of year above 66 deg N, we cannot use noontime adjustment eqns 
    tnoon.tnoon.attrs['units'] = '°C'
    tnoon.tnoon.attrs['standard_name'] = 'air_temperature'
    tnoon.tnoon.attrs['long_name'] = 'Approximate air temperature at solar noon' 
        
    ## Add attrs and save
    tnoon.attrs['description'] = 'Estimated temperature at solar noon based on methods of Beck and Trevitt (1989).' # Where "temperature at solar noon" does not occur at the time index noon UTC time. Outputs instead represent the value at the solar noon that would occur between the time_bnds for that indexed time, for that lat-lon location
    tnoon.attrs['references'] = 'Beck, J. A., & Trevitt, A. C. F. (1989). Forecasting diurnal variations in meteorological parameters for predicting fire behaviour. Canadian Journal of Forest Research, 19(6), 791-797.'
    tnoon.attrs['methods'] = f'tasmaxAdjust and tasminAdjust from CanLEAD-CanRCM4-{version}-v1. '\
                            + 'Offset between solar noon and maximum temperature estimated from CanRCM4 hourly near-surface temperature output. '\
                            + 'Minimum temperature assumed to occur as sunrise. '\
                            + f'Solar noon and sunrise determined for each location (grid point lat-lon) using {sunrise_noon.attrs["pvlib_info"]}.'        
    
    add_attrs(tnoon, 'tnoon') # add additional attrs, and save
    
    ## calculate approx relative humidity at noon
    
    RH_noon = RH_noon_wrapper(tnoon=tnoon.tnoon, tasmin=tasminAdjust, 
                              tasmax=tasmaxAdjust, hurs=hursAdjust['hursAdjust']).to_dataset()
    RH_noon.RH_noon.attrs['units'] = '%'
    RH_noon.RH_noon.attrs['standard_name'] = 'relative_humidity'
    RH_noon.RH_noon.attrs['long_name'] = 'Approximate relative humidity at solar noon'  
    
    RH_noon.attrs['description'] = 'Estimated relative humidity at solar noon based on methods of Allen et al. (1998) and Beck and Trevitt (1989).'
    RH_noon.attrs['references'] = 'Beck, J.A., & Trevitt, A.C.F. (1989). Forecasting diurnal variations in meteorological parameters for predicting fire behaviour. Canadian Journal of Forest Research, 19(6), 791-797. \n'\
                                  + 'Allen, R.G., Pereira, L.S., Raes, D., & Smith, M. (1998). Crop evapotranspiration-Guidelines for computing crop water requirements-FAO Irrigation and drainage paper 56. FAO, Rome, 300(9), '\
                                  + 'D05109. www.fao.org/3/X0490E/x0490e0n.htm'
    RH_noon.attrs['methods'] = f'tasmaxAdjust, tasminAdjust, and hursAdjust from CanLEAD-CanRCM4-{version}-v1. '\
                               + 'Offset between solar noon and maximum temperature estimated from CanRCM4 hourly near-surface temperature output. '\
                               + 'Minimum temperature assumed to occur as sunrise. '\
                               + f'Solar noon and sunrise determined for each location (grid point lat-lon) using {sunrise_noon.attrs["pvlib_info"]}.'       
    
    add_attrs(RH_noon, 'RH_noon') # add additional attrs, and save
  
    del([tasmaxAdjust, tasminAdjust, RH_noon, tnoon])
    gc.collect()

