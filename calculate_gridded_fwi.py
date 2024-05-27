"""
Calculate Canadian Forest Fire Weather Index System commponents using xclim 
"""

import xarray as xr
from xclim.indices.fire import fire_weather_ufunc, fire_season
import xclim as xc
import sys
import os
sys.path.append(os.path.expanduser('~/fwi_updates/CanLEAD-FWI-v1/'))
from filepaths import fwipaths
from config import canada_bounds
import gc
import datetime
import subprocess 
import numpy as np
tracking_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip() 

## SET ENSEMBLE TO CALCULATE FWI

EnsembleNumber = []
j = sys.argv[1] #for j in np.arange(1,6): #1,6 if you want full ensemble
tmp = ['r' + str(j) + '_r' + str(m) + 'i1p1' for m in range(1,8) ]
tmp2 = ['r' + str(j) + '_r' + str(m) + 'i2p1' for m in range(8,11) ]
EnsembleNumber = EnsembleNumber + tmp + tmp2

target_dataset = sys.argv[2] # 'EWEMBI' or 'S14FD'
InputDataDir = f'{fwipaths.input_data}CanLEAD/CanRCM4-{target_dataset}-MBCn/' # for unadjusted wind and precip
InputDataDir2 = f'{fwipaths.working_data}noontime/' # for noontime adjusted RH and Tnoon
OutputDataDir = f'{fwipaths.output_data}CanLEAD-FWI-{target_dataset}-v1/'
if not os.path.exists(OutputDataDir):
    os.makedirs(OutputDataDir)

flnm_a = '_NAM-44i_CCCma-CanESM2_rcp85_'
flnm_b = f'_CCCma-CanRCM4_r2_ECCC-MBCn-{target_dataset}-1981-2010_day_19500101-21001231.nc'

# Canada mask, excluding high Arctic
final_mask = xr.open_dataset(f'{fwipaths.input_data}/CanLEAD_FWI_final_mask.nc')['CanLEAD_FWI_mask']

long_names_cffwis = {'FFMC': 'Fine Fuel Moisture Code',
                     'DMC': 'Duff Moisture Code',
                     'DC': 'Drought Code',
                     'BUI': 'Buildup Index',
                     'ISI': 'Initial Spread Index',
                     'FWI': 'Fire Weather Index',
                     'DSR': 'Daily Severity Rating',
                     'fire_season_mask': 'Fire season mask'
                      }

# LV: Taken almost verbatim from CFS website. How to properly cite, do I need to paraphrase?
# Descriptions of CFFWIS components are taken almost verbatim from: Canadian Forest Service. (no date). Background Information: Canadian
# Forest Fire Weather Index (FWI) System. Available at: https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi
description_cffwis = {'FFMC': 'Numeric rating of the moisture content of litter and other cured fine fuels. '\
                              +'This code is an indicator of the relative ease of ignition and the flammability of fine fuel (NRCan n.d.).',
                      'DMC': 'Numeric rating of the average moisture content of loosely compacted organic layers of moderate depth. '\
                             +'This code gives an indication of fuel consumption in moderate duff layers and medium-size woody material (NRCan n.d.).',
                      'DC': 'Numeric rating of the average moisture content of deep, compact organic layers. This code is a useful indicator of '\
                            +'seasonal drought effects on forest fuels and the amount of smoldering in deep duff layers and large logs (NRCan n.d.).',
                      'BUI': 'Numeric rating of the total amount of fuel available for combustion. It is based on the DMC and the DC (NRCan n.d.).', 
                      'ISI': 'Numeric rating of the expected rate of fire spread. It is based on wind speed and FFMC. '\
                             +'Actual spread rates vary between fuel types at the same ISI (NRCan n.d.).',
                      'FWI': 'Numeric rating of fire intensity. It is based on the ISI and the BUI, and is used as a general index of fire danger '\
                             +'throughout the forested areas of Canada (NRCan n.d.).',
                      'DSR': 'Numeric rating of the difficulty of controlling fires. It is based on the Fire Weather Index (NRCan n.d.).',
                      'fire_season_mask': 'Boolean mask of overwintering period (fire weather calculations turned off) or active fire season '\
                                            +'(when fire weather calculations are turned on), based on temperature thresholds.' 
                      }

for e in EnsembleNumber:
           
    flnm_hurs = f'{fwipaths.working_data}noontime/{e}_RH_noon_1950_2100_{target_dataset}.nc'
    hurs = xr.open_dataset(flnm_hurs)['RH_noon']
    assert hurs.units in ['pct', 'percent', '%'], f'RH_noon in {hurs.units}' 
      
    # tnoon, used for CFFWIS calculations    
    flnm_tnoon = f'{fwipaths.working_data}noontime/{e}_tnoon_1950_2100_{target_dataset}.nc'
    tnoon = xr.open_dataset(flnm_tnoon)['tnoon']
    assert tnoon.units in ['degC', 'degreesC', '°C'], f'tnoon in {tnoon.units}' 
    
    # tmax, used only for determination of 'active' fire season 
    flnm_tmax = f'{InputDataDir}/{e}/tasmaxAdjust{flnm_a}{e}{flnm_b}'
    tasmaxAdjust = xr.open_dataset(flnm_tmax).sel(canada_bounds)['tasmaxAdjust']
    tasmaxAdjust = xc.core.units.convert_units_to(tasmaxAdjust, 'degC')   
  
    flnm_wind = f'{InputDataDir}/{e}/sfcWindAdjust{flnm_a}{e}{flnm_b}'
    sfcWind = xr.open_dataset(flnm_wind).sel(canada_bounds)
    sfcWindAdjust = xc.core.units.convert_units_to(sfcWind['sfcWindAdjust'], 'km/h')    
    
    flnm_pr = f'{InputDataDir}/{e}/prAdjust{flnm_a}{e}{flnm_b}'
    prAdjust = xr.open_dataset(flnm_pr).sel(canada_bounds)['prAdjust']
    prAdjust = xc.core.units.convert_units_to(prAdjust, 'mm/day') 
    
    # In absence of snow depth data, determine 'active' fire season following methods of CFFDRS and Wotton and Flannigan (1993),
    # where fire season starts after 3 days of tmax > 12 degC, and ends after 3 days of tmax < 5 degC
    fire_season_mask = fire_season(tasmaxAdjust, method='WF93', freq=None,  
                                   temp_start_thresh='12 degC', temp_end_thresh='5 degC', 
                                   temp_condition_days=3)
    
     # Run FWI System function from xclim, output as dictionary
    allout = fire_weather_ufunc(tas = tnoon,
                                pr = prAdjust,
                                sfcWind = sfcWindAdjust,
                                hurs = hurs,
                                lat = prAdjust.lat,
                                # overwintering procedures
                                season_mask = fire_season_mask, # calculated above using tmax based on Wotton and Flannigan (1993)
                                overwintering = True, # activate overwintering of DC 
                                # Overwintering of drought code following CFFDRS methods, described in Lawson and Armitage (2008).
                                # Must specify the fall soil moisture carryover fraction, and effectiveness of precipitation
                                # in recharging soil moisture. Values of 0.75 are used for both following the methods of the Canadian Forest
                                # Service observation-based FWI (NFC 2012).
                                carry_over_fraction=1, # McElhinny et al (2020), Hanes and Wotton (2024) https://www.canadawildfire.org/_files/ugd/90df79_deb361a23d534441851a03055b6b67d2.pdf
                                wetting_efficiency_fraction=0.50, # Hanes and Wotton (2024) https://www.canadawildfire.org/_files/ugd/90df79_deb361a23d534441851a03055b6b67d2.pdf
                                # Default spring start-up values for DMC and FFMC:
                                dmc_start=6,
                                ffmc_start=85
                                )
       
    del(allout['winter_pr']) 
    
    for key in allout.keys(): 
        allout[key] = allout[key].rename(key) # rename all dictionary items to allow merge into one dataset
    allout = xr.merge(allout.values())
    allout['fire_season_mask'] = fire_season_mask.rename('fire_season_mask')    
    
    # mask out areas not in Canada land area, excluding Northern Arctic
    allout = allout.where(final_mask==100)
    
    ### Add attributes, add fire_season_length variable, set encoding, and save ### 
   
    for var in allout.data_vars: # set attrs for all FWI inputs
        if var == 'fire_season_mask':
            allout[var].attrs['standard_name'] = 'status_flag' 
            allout[var].attrs['long_name'] = long_names_cffwis[var]
            allout[var].attrs['description'] = description_cffwis[var]
            allout[var].attrs['flag_values'] = np.array((True, False)).astype('int8') # True (active_fire_season), False (overwintering_period)
            allout[var].attrs['flag_meanings'] = 'active_fire_season overwintering_period'
        else:
            allout[var].attrs['units'] = ''
            allout[var].attrs['short_name'] = var # CF does not have a 'standard name' for FWI indices, so use "short_name"
            allout[var].attrs['long_name'] = long_names_cffwis[var]
            allout[var].attrs['description'] = description_cffwis[var]
            allout[var].attrs['ancillary_variables'] = 'fire_season_mask'
                 
    # add time bounds back on
    allout['time_bnds'] = sfcWind['time_bnds'] 
    
    # Add attributes to variables
    for var in ['time', 'lat', 'lon', 'time_bnds']:
        allout[var].attrs = sfcWind[var].attrs
    
    # Add administrative attributes to dataset
    attrs_to_add = dict(## Admin attrs ##
                        Conventions = "CF-1.8",
                        institution = "Canadian Centre for Climate Services (CCCS)",
                        institute_id = "CCCS",
                        contact = "ccsc-cccs@ec.gc.ca", 
                        domain = 'Canada',
                        creation_date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                        product = "fire-weather-projections",
                        project_id = "CanLEAD-FWI", # CanLEAD
                        product_version = f'CanLEAD-FWI-{target_dataset}-v1',
                        title = f'Canadian Forest Fire Weather Index System (CFFWIS) projections based on CanLEAD-CanRCM4-{target_dataset}', 
                        history = f"Generated by {os.path.basename(sys.argv[0])}", 
                        git_id = tracking_id, 
                        git_repo = 'https://github.com/ECCC-CCCS/CanLEAD-FWI-v1/',
                        CORDEX_grid = 'NAM-44i',
                        input_variables = f'Noon temperature: {os.path.basename(flnm_tnoon)}. '\
                                         + f'Maximum daily temperature (for determination of active fire season): {os.path.basename(flnm_tmax)}. '\
                                         + f'Noon relative humidity: {os.path.basename(flnm_hurs)}. '\
                                         + f'Daily mean wind speed: {os.path.basename(flnm_wind)}. '\
                                         + f'Precipitation: {os.path.basename(flnm_pr)}.',
                        ## CFFWIS method specific attrs ## 
                        fire_season = 'Active fire season and overwintering periods are determined using daily maximum '\
                                      +'temperature (tmax) following the methods of Wotton and Flannigan (1993). '\
                                      +'Spring start-up (start of calculations, and active fire season) occurs on the fourth day '\
                                      +'following three consecutive days of tmax >12 °C, autumn shut-down (end of active fire season '\
                                      +'and beginning of overwintering) occurs on the fourth day after three consecutive days of tmax <5 °C.',
                        overwintering = 'The Drought Code (DC) is overwintered following the CFFDRS methods described in '\
                                        +'Lawson and Armitage (2008). A value of one is used for the carry-over fraction '\
                                        +'and a value of 0.5 is used for wetting efficiency fraction (Hanes and Wotton, 2024).',
                        references = 'Van Vliet, L. D. et al. In review. Developing user-informed fire weather projections for Canada. Climate Services.'\
                                     +'Natural Resources Canada (NRCan). [no date]. Background Information: Canadian Forest Fire Weather Index (FWI) System. '\
                                     +'Accessed on: 2023-04-27. Available at: https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi.' , # LV: to be updated with final accepted publication
                        index_package_information = f'CFFWIS outputs calculated using xclim  {xc.__version__} indices.fwi.fire_weather_ufunc and indices.fwi.fire_season. '\
                                                    +'Reference: Logan, Travis, et al. Ouranosinc/xclim: V0.39.0. v0.39.0, Zenodo, 2 Nov. 2022, p., doi:10.5281/zenodo.7274811.', 
                        land_fraction = 'Analysis only performed on "land" grid cells, determined using nearest-neighbour analysis from original NAM-44 grid.' 
                        )
                   
    # add CanLEAD attrs, renaming by prepending 'CanLEAD-CanRCM'
    attrs_to_rename = ['driving_model_id', 'driving_experiment_name', 
                       'driving_model_ensemble_member', 'realization',
                       'initialization_method', 'physics_version', 'forcing',
                       'model_id','rcm_version_id',
                       'CCCma_runid',
                       'experiment_id', 'experiment',
                       'bc_method', 'bc_method_id', 'bc_observation', 
                       'bc_info', 'bc_observation_id', 'bc_period', 
                       'references', 'institution', 'institute_id']
  
    attrs_to_copy = ['frequency', 'data_licence']
    
    # Add all attrs defined above    
    for attr_name, attr_val in attrs_to_add.items():
        allout.attrs[attr_name] = attr_val
                            
    for attr_name in attrs_to_copy:  
        allout.attrs[attr_name] = sfcWind.attrs[attr_name]
        
    for attr_name in attrs_to_rename:  
        allout.attrs['CanLEAD_CanRCM4_' + attr_name] = sfcWind.attrs[attr_name]
              
    # LV: if desired we can save as int16 using offset and scale factor, for additional, lossy compression, but without performance hit                     
    # LV: can add some chunking below, if desired
    # write encoding and compression encoding for out file
    encoding = {var: {'dtype': 'float32',
                      'zlib': True, # compress outputs
                      'complevel': 4, # 1 to 9, where 1 is fastest, and 9 is maximum compression
                      '_FillValue': 1e+20 # missing value depreciated, not added
                      } for var in allout.data_vars} 
    # drop encoding for fire_season_mask, want to keep as original dtype (bool) to save space
    del(encoding['fire_season_mask'], encoding['time_bnds']) 
    # add encoding for lat, lon, time
    for var in ['lat','lon','time','time_bnds']:
        encoding[var] = {'dtype': 'float64',
                         '_FillValue': None}  
        if var in ['time', 'time_bnds']: 
            encoding[var]['units'] = sfcWind.time.encoding['units']  
            encoding[var]['calendar'] = sfcWind.time.encoding['calendar']
        
    allout = allout.transpose("time", "lat", "lon", "bnds") # reorder dims to match CF-preferred ordering to T-Y-X

    # Save                                  
    allout.to_netcdf(f'{OutputDataDir}{e}_CanLEAD-FWI-{target_dataset}-v1.nc', encoding=encoding) 
       
    del([tnoon, tasmaxAdjust, fire_season_mask, hurs, sfcWindAdjust, sfcWind, prAdjust, allout])
    gc.collect()
    
