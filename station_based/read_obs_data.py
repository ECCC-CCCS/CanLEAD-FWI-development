'''
Custom function to read observational CFFWIS station data in from csv, pre-process, and write to netcdf (with attributes) 
on a per-station basis if quality checks are passed. Run within 'pull_station_data.py'.                                                                                   
'''

def read_station_data(provider_name, coverage=0.80, min_length=20, month_start=5, month_end=9, run_script=None):
    '''
    Function to read in station data, clean it up (check for duplicates, minimum data length and quality).
    Outputs xarray dataset with station attributes retained. 
    
    Parameters
    ----------
    provider_name : String. 
        Province or 'NAT' for national, capitalized code.
    coverage : Float, optional
        Proportion of data coverage required within the fire season to keep year of data. The default is 0.80.
    min_length : Integer, optional
        Number of years of suitable data need to retain station and output observations. The default is 20.
    month_start : Integer, optional
        Start month of 'fire season', the period in which 'coverage' and 'min_length' conditions will be verified.
        Default is 5, for May. (Starts on the first day of the month)
    month_end : Integer, optional
        Last month of 'fire season', the period in which 'coverage' and 'min_length' conditions will be verified.
        Default is 9, for September. (Ends on the last date of the month)
    run_script : name of script used to run function, to add to attrs
    
    Returns
    -------
    station_dataset : 
        xarray dataset with fwi codes, subcodes, and input meteorological variables as well as station attributes.
    
    '''
    
    import pandas as pd
    import xarray as xr
    import os
    import bz2
    import glob
    import nested_dict as nd
    import numpy as np
    from scipy import stats
    from datetime import date
    import subprocess 
    tracking_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    import sys
    sys.path.append(os.path.expanduser('~/fwi_updates/CanLEAD-FWI-v1/')) 
    from filepaths import stnpaths, fwipaths
            
    station_dataset = nd.nested_dict() # initialize a dictionary to store all data
    
    if provider_name == 'NAT':
        
        ''' NAT (national) station data obtained from: 
            Natural Resources Canada, Canadian Forest Service, Wildland Fire Information Systems. Canadian Forest Fire Weather Index System data 
            from the Canadian Wildland Fire Information System, Version 3.0. Unpublished Material. Metadata Date: 2012-02-15. 
            Available at: https://cwfis.cfs.nrcan.gc.ca/downloads/fwi_obs/ 
        '''
        
        station_input_data_prefix = stnpaths.nat
        
        # Load station metadata files
        stn_list = pd.concat([
                              pd.read_csv(os.path.join(station_input_data_prefix,'cwfis_allstn2017.csv')),
                              pd.read_csv(os.path.join(station_input_data_prefix,'cwfis_allstn2019.csv'))
                              ]).drop_duplicates(subset='aes', keep='first')
        stn_data = pd.concat([
                              pd.read_csv(bz2.BZ2File(os.path.join(station_input_data_prefix,'cwfis_fwi1950sv3.0.csv.bz2'), 'r'),          dtype={'aes': str}),
                              pd.read_csv(bz2.BZ2File(os.path.join(station_input_data_prefix,'cwfis_fwi1960sv3.0.csv.bz2'), 'r'),          dtype={'aes': str}),
                              pd.read_csv(bz2.BZ2File(os.path.join(station_input_data_prefix,'cwfis_fwi1970sv3.0.csv.bz2'), 'r'),          dtype={'aes': str}),
                              pd.read_csv(bz2.BZ2File(os.path.join(station_input_data_prefix,'cwfis_fwi1980sv3.0.csv.bz2'), 'r'),          dtype={'aes': str}),
                              pd.read_csv(bz2.BZ2File(os.path.join(station_input_data_prefix,'cwfis_fwi1990sv3.0.csv.bz2'), 'r'),          dtype={'aes': str}),
                              pd.read_csv(bz2.BZ2File(os.path.join(station_input_data_prefix,'cwfis_fwi2000sv3.0opEC_2015.csv.bz2'), 'r'), dtype={'aes': str}),
                              pd.read_csv(bz2.BZ2File(os.path.join(station_input_data_prefix,'cwfis_fwi2010sopEC.csv.bz2'), 'r'),          dtype={'aes': str})
                              ]).reset_index(drop=True)
        stn_data=stn_data.dropna(axis=0,how='any',subset=['ffmc', 'dmc', 'dc', 'bui', 'isi', 'fwi']) #drop any records that don't have FWI info
        stn_data=stn_data.reset_index(drop=True)
        
        # Convert rep_date to datetime object, get year and month
        stn_data['date'] = pd.to_datetime(stn_data['rep_date'], format='%Y-%m-%d %H:%M:%S') 
        stn_data['year'] = pd.DatetimeIndex(stn_data['date']).year
        stn_data['month'] = pd.DatetimeIndex(stn_data['date']).month
        stn_data = stn_data.reset_index()  
        stn_list = stn_list.reset_index()
        total_stations_with_data=stn_data['aes'].unique() # recorded as a diagnostic to see how many stations are lost
        
        stn_list=stn_list.set_index('aes')
        stn_list.rename(columns={"lat":"latitude",
                                 "lon":"longitude",
                                 "prov": 'P_T',
                                 "elev": 'elevation',
                                 },
                        inplace=True)
        stn_list['station_name'] = stn_list['name'] 
                
        stn_data.rename(columns={'ffmc': 'FFMC',
                                 'dmc': 'DMC',
                                 'dc': 'DC', 
                                 'isi': 'ISI',
                                 'bui': 'BUI', 
                                 'fwi': 'FWI',
                                 'dsr': 'DSR'},
                        inplace=True)
        stn_data=stn_data.set_index('aes') 
                
    else:
        
        if provider_name == 'YK': 
            ''' Data provided by the Yukon goverment'''
            station_input_data_prefix  = stnpaths.yk 
        
            stn_list = pd.read_csv(station_input_data_prefix + 'yukon-stations.csv')
            stn_data = pd.read_csv(station_input_data_prefix + 'yukon-daily-indices.csv', na_values="-1.#I").reset_index(drop=True) # interpret error code '-1.#I' as NaNs
                   
            # rename for consistency with provincial obs names
            stn_data.rename(columns={'Name': 'station_name', 
                                     'DateObs': 'weather_date',
                                     'Temperature': 'temp', 
                                     'RH': 'rh',
                                     'Wspd': 'ws', 
                                     'WD': 'wind_direction',
                                     'Rain24': 'precip' 
                                     }, 
                            inplace=True)
            
            stn_list.rename(columns={'Name': 'station_name', 
                                     'Lat': 'latitude',
                                     'Lon': 'longitude', 
                                     'Elevation': 'elevation',
                                     'OWNER': 'owner',
                                     'Comments': 'comments'}, 
                            inplace=True)
            stn_list['P_T'] = 'YK'
           
            stn_data['date'] = pd.to_datetime(stn_data['weather_date'], format='%Y%m%d %H:%M:%S') # Convert weather_date to datetime object
               
        else: 
            ''' For all regions besides Yukon, and excluding nationally-sourced (NAT) data from the CWFIC, station data obtained from 
                the Pacific Forestry Centre, CFS, NRCan
            '''
            station_input_data_prefix = stnpaths.prov
            f = glob.glob(os.path.join(station_input_data_prefix,provider_name+'_station_list*.csv'))
            
            if provider_name=='QC': # for QC, change province name from uppercase and use special encoding to read special characters
                stn_list = pd.read_csv(f[0], encoding='iso-8859-1', encoding_errors='replace')
                provider_name='Qc'
                stn_data = pd.concat([pd.read_csv(f, encoding='iso-8859-1', encoding_errors='replace') for f in glob.glob(os.path.join(station_input_data_prefix,provider_name+'*Daily*.csv'))]).reset_index(drop=True)
            else:
                stn_list = pd.read_csv(f[0])
                stn_data = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(station_input_data_prefix,provider_name+'*Daily*.csv'))]).reset_index(drop=True)
                
            stn_data['date'] = pd.to_datetime(stn_data['weather_date'], format='%Y%m%d') # Convert weather_date to datetime object 
            stn_data.rename(columns={'relative_humidity': 'rh',
                                     'wind_speed': 'ws', 
                                     'temperature': 'temp',
                                     'precipitation': 'precip', 
                                     'ffmc': 'FFMC',
                                     'dmc': 'DMC',
                                     'dc': 'DC', 
                                     'isi': 'ISI',
                                     'bui': 'BUI', 
                                     'fwi': 'FWI',
                                     'dsr': 'DSR'}, 
                            inplace=True)
        
        stn_data=stn_data.dropna(axis=0,how='any',subset=['FFMC', 'DMC', 'DC', 'BUI', 'ISI', 'FWI']) # drop any records that don't have FWI info
        stn_data=stn_data.reset_index(drop=True)
        
        # get year and month
        stn_data['year'] = pd.DatetimeIndex(stn_data['date']).year
        stn_data['month'] = pd.DatetimeIndex(stn_data['date']).month
        stn_data = stn_data.reset_index()
        stn_list = stn_list.reset_index()
        
        if provider_name in ['NS', 'NWT', 'YK']:
            total_stations_with_data=stn_data['station_name'].unique()
            stn_list=stn_list.set_index('station_name')
            stn_data=stn_data.set_index('station_name')
        else:
            total_stations_with_data=stn_data['station_code'].unique()
            stn_list=stn_list.set_index('station_code')
            stn_data=stn_data.set_index('station_code')    
                       
    quality_checks = pd.DataFrame(columns=['number_dropped_duplicates', 'years_dropped_insufficient_fire_season_data',
                                           'pass_quality_criteria', 'years_good_data', 'land_area_fraction']) # initiate quality tracking dataframe for records 
    kurtosis_checks = pd.DataFrame(columns=['FFMC', 'DMC', 'DC', 'BUI', 'ISI', 'FWI', 'DSR']) # initiate kurtosis tracking dataframe to flag for outliers 
            
    for s in stn_list.index: # this only evaluates stations that exist in station metadata list
        if s in stn_data.index: # that also have data
            
            obs_data=stn_data.loc[s] # get data for this station
            
            if len(obs_data.shape) > 1: # if data has more than one record
            
                ########## Check for and remove duplicates ##########
                
                # first, if duplicates are identical, keep only one
                obs_data = obs_data.drop('index', axis=1).drop_duplicates(keep='first') # keeps first duplicate. this line also drops the column index
                osize = obs_data.shape # get size of dataframe, to record if non-identifical duplicates dropped in next step
                
                # then, find and remove duplicate dates with values that differ, and note these in error report
                obs_data = obs_data.drop_duplicates(subset='date', keep=False) # consider only the 'date' column in dropping duplicates, keep=False to keep no duplicates
                if osize[0] != obs_data.shape[0]: # if size of dataframe not the same before/after drop performed (aka, any data was dropped), note this
                    quality_checks.loc[s, 'number_dropped_duplicates'] = osize[0] - obs_data.shape[0] # count records dropped and record in error report 
                
                ########## Remove years with insufficient fire season data ########## 
                
                # where fire season month_start and month_end are defined in function
                data_summer_count = obs_data[obs_data.month.isin(range(month_start, month_end + 1))].groupby('year').count() # subset data to fire season, groupby year, and count days in fire season with data (will not count NaNs) 
                data_year_count=obs_data.groupby('year').count() # count days in whole year - not just fire season - with data (will not count NaNs)
                
                # drop years with less than defined fire season coverage
                fire_season_days = (date(2023, month_end + 1, 1) - date(2023, month_start, 1)).days # find days in fire season period using 2023 as representative year (NB: 153 in MJJAS)
                yrs_insufficient_fs = data_summer_count[data_summer_count['date']<int(fire_season_days*coverage)].index # list of years that have less than required fire season coverage [coverage requirement is added as an attribute to output file]
                yrs_no_fs = data_year_count[~data_year_count.index.isin(data_summer_count.index)].index # years with no fire season data (aka year in data_year_count, but not in data_summer_count). Otherwise we could accidentally keep years with no summer data but with winter data 
                yrs_to_drop = np.append(yrs_insufficient_fs, yrs_no_fs)  
                
                obs_data = obs_data[~obs_data.year.isin(yrs_to_drop)] # drop all data from these years 
                quality_checks.loc[s, 'years_dropped_insufficient_fire_season_data'] = yrs_to_drop # record years dropped for records
                kurtosis_checks.loc[s] = stats.kurtosis(obs_data[['FFMC', 'DMC', 'DC', 'BUI', 'ISI', 'FWI', 'DSR']], # kurtosis for testing for outliers
                                                        axis=0, nan_policy='omit')
                
                ########## Count number of years, if at least min_length of years, then perform conversion to xarray dataset ##########
                    
                dyr = obs_data.groupby('year')['FWI'].count() # count values in years, see how many years of data we have
                if len(dyr) >= min_length: #If there is at least a certain span of data, convert to xr dataset and add to station_dataset dictionary
                    quality_checks.loc[s, 'pass_quality_criteria'] = 'yes'
                    quality_checks.loc[s, 'years_good_data'] = len(dyr)
                    
                    ########## Re-index dataframe with full year calendars ##########
                    
                    out = obs_data.set_index('date').sort_index() # set index from station_name to date, and sort by date
                    obs_yrs, obs_yre = out.index[0].year, out.index[-1].year
                    idx = pd.date_range(f'01-01-{obs_yrs}', f'12-31-{obs_yre}') # range potential range for obs data 
                    if provider_name in ['NAT', 'YK']: # NAT, YK have different date format, must specify it here so joining by date won't break dataframe
                        idx = pd.date_range(f'01-01-{obs_yrs} 12:00', f'12-31-{obs_yre} 12:00')
                    out = out.reindex(idx, fill_value=np.NaN) # re-index dataframe with idx. This will fill missing values (aka no data for that date) with NaNs
                    out = out[~((idx.month == 2) & (idx.day == 29))] # remove leap days from data. when doing bias correction, we need equal calendars, and model data is 365 day year format
                    out.index = xr.cftime_range(f'{obs_yrs}-01-01 12:00', f'{obs_yre}-12-31 12:00', freq='D', calendar='noleap', name='time')
                    
                    ########## convert dataframe to xarray dataset ##########
                    
                    if provider_name == 'NAT': # for NAT, retain following columns and convert to xarray
                        outxr = out[['temp', 'rh', 'ws', 'precip', 'calcstatus',
                                     'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'DSR']].to_xarray()
                    else: # for all others, keep following columns and convert to xarray. differences due to lack of 'calcstatus' for prov data
                        outxr = out[['temp', 'rh', 'ws', 'precip', 
                                     'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'DSR']].to_xarray()
                        
                    ########## add attributes to dataset ##########
                    
                    if provider_name == 'NAT':  # for NAT, add following from stn_list to ds attributes 
                        for atr in ['wmo', 'id', 'station_name', 'instr', 'longitude', 
                                    'latitude', 'elevation', 'P_T', 'tz_correct', 'agency']:
                            try: outxr.attrs[atr] = stn_list.loc[s, atr].strip() # formatting func strip() will fail for non-strings, so 'try-except' is needed
                            except AttributeError: outxr.attrs[atr] = stn_list.loc[s, atr] 
                        outxr.attrs['aes'] = s # 's' originally called 'aes', so re-set to this in attrs
                        outxr.attrs['description'] = 'Observational CFFWIS system inputs and outputs for national CWFIS stations.'
                        outxr.attrs['data_source'] = 'https://cwfis.cfs.nrcan.gc.ca/downloads/fwi_obs/' # origin of data
                    else:  # for PROV/TER, need following attrs
                        for atr in ['station_name', 'P_T', 'owner', 'longitude', 'latitude', 'elevation', 'comments']:
                            try: outxr.attrs[atr] = stn_list.loc[s, atr] # 'Comments' only present for YK
                            except KeyError: pass 
                        outxr.attrs['description'] = f'Observational CFFWIS system inputs and outputs for {provider_name} stations.' 
                        outxr.attrs['data_source'] = 'Pacific Forestry Centre, NRCan for all regions except Yukon, where station data was provided by the Government of Yukon, Wildland Fire Management branch.' # origin of data 
                    
                    outxr.attrs['station_code'] = s # for NWT, YK and SK: no station code, so station_code = station_name. Duplicate of AES for NAT, but added for consistency w provincial data    
                    if 'station_name' not in outxr.attrs:
                        outxr.attrs['station_name'] = s
                                                                          
                    outxr.attrs['history'] = f'Generated by {run_script}'
                    outxr.attrs['years_with_obs_data'] = dyr.index.values # years wtih obs data will be useful to know later, when doing bias adjustment
                    outxr.attrs['completeness_checks'] = f'Data retained for years with >= {coverage*100}% data coverage during the fire season, here defined as the start of month {month_start} to the end of month {month_end}.'\
                                                        +f' Duplicate dates and invalid values removed if present. Only stations with >= {min_length} years of acceptable data retained.'
                    outxr.attrs['git_id'] = tracking_id 
                    outxr.attrs['git_repo'] = 'https://github.com/ECCC-CCCS/CanLEAD-FWI-v1/commit/'
                    
                    # add lat and lon as dataset dims (in addition to being file attrs) 
                    outxr = outxr.assign_coords(lon=outxr.attrs['longitude']).expand_dims('lon')
                    outxr = outxr.assign_coords(lat=outxr.attrs['latitude']).expand_dims('lat')
                                  
                    ## some final quality checks ## 
                    for dv in ['ws', 'precip', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'DSR']:
                        outxr[dv] = xr.where((outxr[dv] >= 0), outxr[dv], np.nan)
                    outxr['rh'] = xr.where((outxr['rh'] >= 0) & (outxr['rh'] <= 100), outxr['rh'], np.nan)
                    outxr['FFMC'] = xr.where((outxr['FFMC'] >= 0) & (outxr['FFMC'] <= 101), outxr['FFMC'], np.nan) # only ffmc is not open-ended
                    # no checks for temp
                    
                    # add dataset to output dictionary
                    station_dataset[s]['obs_data'] = outxr
                else:
                    quality_checks.loc[s, 'pass_quality_criteria'] = 'no'
                    quality_checks.loc[s, 'years_good_data'] = len(dyr)
                    
    print(f'\n Useable stations (land and water) in {provider_name}: {len(list(station_dataset.keys()))} of {len(total_stations_with_data)}')
    
    land_mask = xr.open_dataset(fwipaths.working_data + 'CanLEAD_sftlf_nearest.nc')['sftlf']
    glac_mask = xr.open_dataset(fwipaths.working_data + 'CanLEAD_sftgif_nearest.nc')['sftgif']
    mask = xr.where((land_mask == 100) & (glac_mask == 0), 100, 0) # 100 where land AND not glacier, 0 elsewhere
    for s in station_dataset.keys():
        mod_cell = mask.sel(lat=station_dataset[s]['obs_data']['lat'].values[0],
                            lon=station_dataset[s]['obs_data']['lon'].values[0],
                            method='nearest')
        station_dataset[s]['obs_data'].attrs['land_area_fraction'] = mod_cell.values # 100 for land, 0 for glacier/water
        quality_checks.loc[s, 'land_area_fraction'] = mod_cell.values
    
    return station_dataset, quality_checks, kurtosis_checks
