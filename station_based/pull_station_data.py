''' 
Script to generate individual station netcdf datasets of weather station CFFWIS observations as well as projected CFFWIS outputs from 
CanLEAD-FWI. Run for each station which meets the quality checks specified in the function read_station_data (see read_obs_data.py)
'''

import os
import sys
import glob
import xarray as xr
sys.path.append(os.path.expanduser('~/fwi_updates/CanLEAD-FWI-v1/')) 
from read_obs_data import read_station_data 
from filepaths import fwipaths

provider_name = sys.argv[1] # code indicating provincial, territorial or national data source

## Load and save station observational data. 

station_dataset, quality_checks, kurtosis_checks = read_station_data(provider_name, coverage=0.70, min_length=20, 
                                                                     month_start=5, month_end=9, run_script=sys.argv[0])

# save obs

obs_out_path = f'{fwipaths.input_data}/station_observations'
if not os.path.exists(obs_out_path):
    os.makedirs(obs_out_path)

kn = list(station_dataset.keys())[0] # get single station to set encoding
encoding_obs = {var:  {'dtype': 'float32',
                      'zlib': True, # compress outputs
                      'complevel': 3, # 1 to 9, where 1 is fastest, and 9 is maximum compression
                      '_FillValue': 1e+20 # missing value depreciated, not added
                      } for var in station_dataset[kn]['obs_data'].data_vars}  
                                        
for s in station_dataset.keys(): 
    station_dataset[s]['obs_data'].to_netcdf(f'{obs_out_path}/{s}_{provider_name}_obs_data.nc', encoding=encoding_obs) # save 

quality_checks['P_T'] = provider_name
quality_checks.to_csv(f'{obs_out_path}/quality_report_{provider_name}.csv')
kurtosis_checks.to_csv(f'{obs_out_path}/kurtosis_report_{provider_name}.csv')

## Load and save CanLEAD pointwise data

version = f'CanLEAD-FWI-{sys.argv[2]}-v1' # set version
CanLEAD_FWI_input_data = glob.glob(f'{fwipaths.output_data}/{version}/*.nc') # get all realizations

# Loop over CanLEAD realizations and accumulate pointwise data
for r in CanLEAD_FWI_input_data:
    r_name=os.path.basename(r)[:10] # get realization ID from file name
    
    mod_ds = xr.open_dataset(r)
    mod_ds = mod_ds.assign_coords(realization=r_name).expand_dims('realization') # add realization as dimension
    
    for s in station_dataset.keys(): # loop over observation-based locations, and load nearest model data
        station_dataset[s]['model_data'][r_name] = mod_ds.sel(lat=station_dataset[s]['obs_data']['lat'],
                                                              lon=station_dataset[s]['obs_data']['lon'],
                                                              method='nearest').drop_vars('time_bnds') # drop time bands, we don't want to index by realization in merge
    
    time_bounds = mod_ds.squeeze().time_bnds # pull time_bnds before close
    time_units, time_cal = mod_ds.time.encoding['units'], mod_ds.time.encoding['calendar'] # pull some encoding
    mod_ds.close()

## Concatenate collected data for each station location, along new 'realization' dimension

# to save space, set encoding as float32, zlib compress
encoding_mod = {var: {'dtype': 'float32',
                      'zlib': True, # compress outputs
                      'complevel': 3, # 1 to 9, where 1 is fastest, and 9 is maximum compression
                      '_FillValue': 1e+20 # missing value depreciated, not added
                      } for var in station_dataset[kn]['model_data'][r_name].data_vars} 
del(encoding_mod['fire_season_mask']) # drop encoding for fire_season_mask, want to keep as original dtype (bool) to save space
# add encoding for lat, lon, time
for var in ['lat','lon','time','time_bnds']:
    encoding_mod[var] = {'dtype': 'float64',
                         '_FillValue': None}  
    if var in ['time', 'time_bnds']: 
        encoding_mod[var]['units'] = time_units
        encoding_mod[var]['calendar'] = time_cal
encoding_mod['realization'] = {'dtype': 'S1'} # add string encoding for realization 
encoding_mod['fire_season_mask'] = {'dtype': 'bool', '_FillValue': None} 

model_out_path = f'{fwipaths.output_data}/{version}/station_outputs/model_data/'
if not os.path.exists(model_out_path):
    os.makedirs(model_out_path)

# set some realization attrs to be written to outfile
realization_attrs = dict(standard_name = 'realization',
                         long_name = 'Ensemble member',
                         units = '')  

# for each station: merge, add time_bnds, add realization attrs, and save
for s in station_dataset.keys():    
    # merge all realizations. keep all attrs, except drop conflicts (e.g., realization name)
    station_dataset[s]['model_data'] = xr.merge(station_dataset[s]['model_data'].values(), 
                                                combine_attrs='drop_conflicts') 
    # re-add time_bnds
    station_dataset[s]['model_data']['time_bnds'] = time_bounds
    # set realization coordinate attrs
    station_dataset[s]['model_data']['realization'].attrs = realization_attrs
    # save
    station_dataset[s]['model_data'].to_netcdf(f'{model_out_path}/{s}_{provider_name}_{version}_model_data.nc', encoding=encoding_mod)

