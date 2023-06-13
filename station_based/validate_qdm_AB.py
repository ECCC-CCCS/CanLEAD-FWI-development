''' Run quantile delta mapping to bias adjust single grid-cell outputs from CanLEAD-FWI to station-based observational data. Validation routine; same as original but
    performs ABABAB cross-validation in reference period.
'''

# Import modules

import os
import sys
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from xclim import sdba
from xclim.sdba import construct_moving_yearly_window, unpack_moving_yearly_window
from xclim.sdba.processing import to_additive_space, from_additive_space, jitter
sys.path.append(os.path.expanduser('~/fwi_updates/CanLEAD-FWI-v1/')) 
from filepaths import fwipaths
import subprocess
tracking_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

#%% Match realizations, so we can train and adjust with different realizations

def assign_matches(EnsembleNumber):
    EnsembleNumber = np.append(EnsembleNumber, EnsembleNumber[0]) # add first real onto end, so last real has a match
    out = {EnsembleNumber[i]: EnsembleNumber[i+1] for i in range(len(EnsembleNumber)-1) } # len-1 so we don't assign to last (duplicate) realizations        
    return out

#%% QDM for bias adjustment and downscaling

def QM_routine(obs, hist, sim, qnts=None, realization=None, training_realization=None, 
               fwi_component=None, step=10, count_non_nans=None): 
    '''
    Apply quantile delta mapping on one realization and one FWI component. 
    Returns the QDM-adjusted data, a dataarray recording the number of records used for adjustment, 
    the adjustment factors used, and the automatic 'history' to output as an attribute.

    Parameters
    ----------
    obs : xarray dataarray
        Target data to be used for training.
    hist : xarray dataarray
        Model data to be used for training. Datarray should be of same dimensions as obs (years of data).
    sim : xarray dataarray
        Data to be adjusted, containing 1950-2100.
    qnts : array
        Quantiles to used in QDM (for which adjustment factors will be generated).
    realization : String
        Realization on which QDM is applied.
    training_realization : String
        Realization which is used for QDM training. Must be different from 'realization'.
    fwi_component : String
        FWI component on which QDM is being applied.
    step : Integer, optional
        Step of moving window in years, the slice of central data which will be retained. The default is 10.
    count_non_nans : xarray dataarray
        Datarray to record count of all non-nan data, i.e., data used for adjustment, for future records/testing.

    Returns
    -------
    out : xarray datarray
        Bias-adjusted model output for the input 'realization' and 'fwi_component'.
    count_non_nans: xarray dataarray
        count_non_nans dataarray with newly recorded count of all non-data, i.e. how much data used for adjustment
    adf : xarray datarray
        Adjustment factors used for QDM, to save for testing.
    out.attrs['history'] : String
        History of QDM, to be saved as attr in final output
        
    '''
    assert realization != training_realization, 'Error: Trying to train with same realization'
    
    # select realization, drop dim to avoid merge conflicts
    sim = sim.sel(realization=realization).squeeze().drop('realization') # select realization data on which to apply QDM
    hist = hist.sel(realization=training_realization).squeeze().drop('realization') # select different ensemble member to train with 
    
    # save data sizes, for validation purposes 
    count_non_nans.loc[{'realization': training_realization}] = hist.groupby('time.dayofyear').count() # count all non-nan data over training period, save in xarray dataarray
    count_non_nans.loc[{'realization': 'obs'}] = obs.groupby('time.dayofyear').count() 
    
    # Mask out NaNs to prevent discrepancies due to systemic differences in season start and end dates 
    hist = xr.where(obs.squeeze().notnull(), hist, np.NaN) # mask out OBS NaNs
    obs = xr.where(hist.squeeze().notnull(), obs, np.NaN) # mask out CanLEAD NaNs

    hist.attrs['units'] = '' # required by xclim, is lost during xr.where
    obs.attrs['units'] = ''
    
    if fwi_component == 'FFMC': # for FFMC, must convert to additive space using logit transform, as it has an upper bound (bounds = (0, 101))
        # first, jitter under and over threshold (i.e., replace values by a uniform random noise). Otherwise, values which equal zero or 101 will be converted to -inf and +inf
        # lower and upper bounds specify values under and over which jittering is performed, respectively. min and max match data possible min and max
        sim = jitter(sim, lower='0.25', upper='100.75', minimum='0', maximum='101') 
        hist = jitter(hist, lower='0.25', upper='100.75', minimum='0', maximum='101')
        obs = jitter(obs, lower='0.25', upper='100.75', minimum='0', maximum='101')
        # then, convert to additive stpace
        sim = to_additive_space(sim, lower_bound='0', upper_bound='101', trans='logit')
        hist = to_additive_space(hist, lower_bound='0', upper_bound='101', trans='logit')
        obs = to_additive_space(obs, lower_bound='0', upper_bound='101', trans='logit')
        kind = '+'  #  set adjustment type to additive (+)  
    else:
        kind = '*' # set adjustment type to multiplicative (*) for all other CFFWIS components, which are bounded by (0, infinity) 
             
    # Generate adjustment factors for each quantile defined in 'qnts' using cross-masked obs and hist as training data
    QM = sdba.QuantileDeltaMapping.train(obs, # observation data, target data
                                         hist,  # training data, modelled historical
                                         nquantiles=qnts, # quantiles for generating adjustment factors
                                         group='time', # time grouping == 'time', which means no grouping is applied beyond 30 year window
                                         kind=kind) # either '+' for additive adjustment or '*' for multiplicative adjustment, as defined above   
    
    # Adjust model data using adjustment factors defined above and stored in "QM"
    out = QM.adjust(sim, extrapolation="constant", interp="linear") # constant extrapolation (of adj factors), with linear interpolation between adj factors
    
    # Re-assign realization dimension dropped above, and name output to appropriate CFFWIS component name
    out = out.assign_coords({'realization':realization}).expand_dims(['realization']).rename(fwi_component)
        
    if fwi_component == 'FFMC': # for FFMC, convert back to from logit space, this is automatically logged in 'history'
        out = from_additive_space(out, lower_bound='0', upper_bound='101', trans='logit', units='dimensionless')
          
    adf = QM.ds.assign_coords({'realization':realization}).expand_dims(['realization']) # pull adjustment factors used for QDM from "QM" object, to save for records
    
    return out, count_non_nans, adf, out.attrs['history']

def apply_qm(s, provider_name, data_obs=None, data_model=None, quantiles=None, high_adj=None, fwi_components=None):
    '''
    Preprocess observational data and model output, then run quantile delta mapping (QDM) to bias adjust
    single grid-cell outputs from CanLEAD-FWI to station-based observational data. QDM-adjusted output
    is saved within the functions. Only quality check parameters are output.

    Parameters
    ----------
    s : String
        Station ID.
    provider_name : String
        Data provider name in form of province or territory initials or 'NAT' or national.
    data_obs : xarray dataset
        Station-based observation CFFWIS data. Target data to be used for training for QDM. 
    data_model : xarray dataset
        Single grid cell model output of CFFWIS components. 
        Must have a 'realization' dimension and contain all CFFWIS components and realizations in one dataset.
    quantiles : array
        Quantiles to used in QDM (for which adjustment factors will be generated).
    high_adj : dataframe
        Dataframe to record the highest adjustment factor for each station and CFFWIS component. Used for validation only.
    fwi_components : list
        List of strings identifying which CFFWIS components to bias adjust.
        
    Returns
    -------
    Dataframe
        Records the highest adjustment factor for each station and CFFWIS component.

    '''
    
    nan_components = {} # initialize a dictionary to record dataset sizes for target and training data
    
    for fwi_component in fwi_components: 
        
        ### Load and preprocess station and model data ### 
        
        # subset station data to CFFWIS component, remove years with no data
        years = data_obs.attrs['years_with_obs_data'] # use all years available for bias correction 
        yidx = len(years) # length of training data
                
        years_1 = years[0::2] # Select every other number, starting at index 0. This half of years for traing, other half for adjustment (validation). Then switch. 
        years_2 = years[1::2] # Select every other number, starting at index 1                
              
        dct_adjusted = {}
        dct_count_nans = {}
        dct_factors = {}
                
        for years_cal, years_val, marker in zip ([years_1, years_2],
                                                 [years_2, years_1],
                                                 ['years_1_cal', 'years_2_cal']):
                
            obs_data = data_obs[fwi_component]
                    
            o_lat, o_lon = obs_data.lat.values[0], obs_data.lon.values[0] # record station lat-lon
            obs_yrs, obs_yre = int(years.min()), int(years.max())
            
            obs_data = obs_data.sel(time=slice(f'{obs_yrs}-01-01', f'{obs_yre}-12-31')).squeeze() # keep lat-lon for obs, so will be on output data 
            obs_data = obs_data.where(obs_data.time.dt.year.isin(years_cal), drop=True) # drop all years with no data from index
                   
            # load model data
            mod_data = data_model[fwi_component]
            m_lat, m_lon = mod_data.lat.values[0], mod_data.lon.values[0]
            mod_data = mod_data.squeeze().drop(['lat', 'lon']) # drop lat-lon so there's not conflicts with obs vs model
                   
            # get ref period, remove years with no OBS data
            mod_ref = mod_data.sel(time=slice(f'{obs_yrs}-01-01', f'{obs_yre}-12-31')) # get training period data for model
            mod_ref = mod_ref.where(mod_ref.time.dt.year.isin(years_cal), drop=True) # drop all years with no OBS data from index, so we have same length of data as OBS
                                   
            mod_sim = mod_data.sel(time=slice(f'{obs_yrs}-01-01', f'{obs_yre}-12-31')) # get reference period data for model
            mod_sim = mod_sim.where(mod_sim.time.dt.year.isin(years_val), drop=True) 
                                               
            # match each realization to another realization, so can use separates real. for training and adjustment
            match_reals = assign_matches(mod_ref.coords['realization'].values) 
                                           
            ### Apply Quantile Delta Mapping (QDM) using xclim, per: 
            #    Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by
            #    quantile mapping: how well do methods preserve changes in quantiles and extremes?. 
            #    Journal of Climate, 28(17), 6938-6959.
            
            # initialize dataarray to record count of all non-nan data, i.e., data used for adjustment, for future records/testing.
            count_non_nans = xr.DataArray(data = -9999, 
                                          dims=["realization", "dayofyear"],
                                          coords=dict(realization = np.append(mod_data.realization.values, 'obs') ,
                                                      dayofyear=range(1,366)
                                                      ))
            
            # initialize dictionaries to record adjusted data and adjustment factors (for QC)
            adj_fact = {} 
            QDMall = {}
            # run QDM on each realization, save output in dictionaries
            for r in mod_ref.coords['realization'].values: 
                QDMall[r], count_non_nans, adj_fact[r], qdm_hist = QM_routine(obs_data, mod_ref, mod_sim,                                                                                                                 
                                                                              qnts=quantiles,
                                                                              realization=r, 
                                                                              training_realization=match_reals[r], 
                                                                              fwi_component=fwi_component,
                                                                              count_non_nans=count_non_nans
                                                                              ) # qdm_hist provides a record of QDM processing applied, for out attrs
            adjusted = xr.merge(QDMall.values()) # merge all realizations into one dataset
            adjusted['time_bnds'] = data_model['time_bnds'] 
            factors_all = xr.merge(adj_fact.values())    
            
            dct_adjusted[marker] = adjusted.assign_coords(calibration_data=("time", np.repeat(marker, len(adjusted.time))))
            dct_count_nans[marker] = count_non_nans.assign_coords(calibration_data=marker).expand_dims('calibration_data').rename(fwi_component)
            dct_factors[marker] = factors_all.assign_coords(calibration_data=marker).expand_dims('calibration_data')
            
        nan_components[fwi_component] = xr.merge(dct_count_nans.values()) # add count_non_nans to dictionary
        adjusted = xr.merge(dct_adjusted.values())
        factors_all =  xr.merge(dct_factors.values()) 
        
        # save adjustment factors
        factors_all.to_netcdf(f'{out_path}adj_factors_nans/{s}_{provider_name}_{fwi_component}_adjustment_factors.nc', encoding={'af': {'dtype': 'float32', 'zlib': True, 'complevel': 5},
                                                                                                                                 'hist_q': {'dtype': 'float32',  'zlib': True, 'complevel': 5}})  
        # record high adjustment factor values
        high_adj.loc[s, fwi_component] = factors_all.af.max().max().values
            
        ### Add attrs, and save to netCDF ###
        
        # define and add new attrs
        attrs_to_add = dict(product = "station-based-fire-weather-projections",
                            title = f'Canadian Forest Fire Weather Index System (CFFWIS) projections based on CanLEAD-CanRCM4-{sys.argv[3]}, '\
                                    +'bias-adjusted to station observations', 
                            creation_date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                            obs_lat_lon = (o_lat, o_lon), 
                            model_lat_lon = (m_lat, m_lon),
                            years_training_data = yidx,
                            calibration_year_group_1 = years_1,
                            calibration_year_group_2 = years_2,
                            history = f'Generated by {sys.argv[0]}. All NaNs in observation and model datasets cross-masked '\
                                      +'to avoid fire season start or end discrepancies. Quantile delta mapping (QDM) training '\
                                      +'is completed with different ensemble member than adjustment. QDM history: '\
                                      +f'{qdm_hist}. Quantiles: {quantiles}.', # xclim version is recorded already in 'qdm_hist'
                            git_id = tracking_id,
                            git_repo = 'https://github.com/ECCC-CCCS/CanLEAD-FWI-v1/',
                            )
        adjusted.attrs = attrs_to_add # add new attrs
                
        # copy only desired attrs from input model data
        mod_attrs_keep = ['Conventions', 'institute_id', 'institution', 'contact', 'frequency', 
                          'data_licence', 'index_package_information', 'references', 'overwintering',
                          'fire_season', 'project_id'] 
        for attr_name in mod_attrs_keep:
            adjusted.attrs[attr_name] = data_model.attrs[attr_name] 
            
        # add desired attrs from input obs data, pre-pend 'obs'
        for attr_name in ['description', 'data_source', 'station_name', 'station_code']:
            adjusted.attrs['obs_' + attr_name] = data_obs.attrs[attr_name] 
                    
        # replace all variable-specifc attrs lost during QDM (including for lat, lon, time, CFFWIS component, time_bnds)
        for var in adjusted.variables: 
            try: adjusted[var].attrs = data_model[var].attrs
            except KeyError: pass # for newly added dim 'calibration_data' 
        adjusted[fwi_component].attrs['long_name'] = "Bias-Adjusted " + adjusted[fwi_component].attrs['long_name']  
        
        # set encoding
        encoding_mod = {fwi_component: {'dtype': 'float32',
                                        'zlib': True, # compress outputs
                                        'complevel': 3, # 1 to 9, where 1 is fastest, and 9 is maximum compression
                                        '_FillValue': 1e+20 # missing value depreciated, not added
                                        } }
        # add encoding for lat, lon, time
        for var in ['lat','lon','time','time_bnds']: 
            encoding_mod[var] = {'dtype': 'float64',
                                 '_FillValue': None}  
            if var in ['time', 'time_bnds']: 
                encoding_mod[var]['units'] = data_model.time.encoding['units']
                encoding_mod[var]['calendar'] = data_model.time.encoding['calendar']
        encoding_mod['realization'] = {'dtype': 'S1'} # add string encoding for realization coord
                
        # save QDM-adjusted model
        print(adjusted)
        print(adjusted.attrs)
        adjusted.to_netcdf(f'{out_path}{s}_{provider_name}_{fwi_component}_{version}_QDM.nc', encoding=encoding_mod)
               
    # save count_non_nans after all fwi components are run
    encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 5} for var in fwi_components} # to save space, encode as float32, zlib compress
    xr.merge(nan_components.values()).to_netcdf(f'{out_path}adj_factors_nans/{s}_{provider_name}_count_nans.nc', encoding=encoding) 
    
    return high_adj # return high_adj df, will get fed back in for next station to record adj_factors

### Set params and run the functions above to apply ### 

# Import list of all stations that pass completeness checks
all_stations = pd.concat([pd.read_csv(fl, index_col=0).reset_index() for fl in glob.glob(f'{fwipaths.input_data}/station_observations/quality_report_*.csv')])
all_stations.rename(columns={'index': 'stn_id'}, inplace=True)
all_stations = all_stations[all_stations.pass_quality_criteria == 'yes']
all_stations = all_stations.reset_index(drop=True)

# Subset to run group
i1 = int(sys.argv[1]) 
step = int(sys.argv[2])
stations = all_stations[i1:i1+step] 

# Get the version from run file
version = f'CanLEAD-FWI-{sys.argv[3]}-v1' 

# Set and make output directory
out_path = f'{fwipaths.output_data}/{version}/station_outputs/validate_qdm_AB/' 
if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(f'{out_path}adj_factors_nans'):
    os.makedirs(f'{out_path}adj_factors_nans')
        
adjustment_factors = pd.DataFrame(columns=['FFMC','DMC','DC','ISI','BUI','FWI','DSR']) # initialize dataset to track high adjustment factors

# Run QDM for all stations 
for ii in tqdm(stations.index): 
    s, provider_name = stations.loc[ii,'stn_id'], stations.loc[ii,'P_T']
    
    # Set to run for select CFFWIS components
    if provider_name in ['NS', 'QC', 'SK']:
      fwi_components = ['FFMC','DC','DMC','ISI','BUI','FWI'] # DSR not available for these P/Ts
    else: 
      fwi_components = ['FFMC','DC','DMC','ISI','BUI','FWI','DSR'] # rest of P_Ts have DSR (for at least some stations)

    # obs data
    fl = f'{fwipaths.input_data}station_observations/{s}_{provider_name}_obs_data.nc' 
    obs = xr.open_dataset(fl)
    # model_data
    fl2 = f'{fwipaths.output_data}/{version}/station_outputs/model_data/{s}_{provider_name}_{version}_model_data.nc' 
    model = xr.open_dataset(fl2)
    
    # run QDM, which saves adjusted_model automatically, and returns only dateframe of adjustment factors
    adjustment_factors = apply_qm(s, provider_name, data_obs=obs, data_model=model,
                                  quantiles=np.linspace(0.01, 0.99, 99),
                                  high_adj=adjustment_factors,
                                  fwi_components=fwi_components)
               
    # save fire season mask into adjusted folder                   
    model.drop_vars(['FFMC','DC','DMC','ISI','BUI','FWI','DSR']).to_netcdf(f'{out_path}{s}_{provider_name}_fire_season_mask_{version}.nc', encoding={'fire_season_mask': {'dtype': 'bool', '_FillValue': None}})
    
# After running for all stations, save adjustment factor outputs
adjustment_factors.to_csv(f'{out_path}adj_factors_nans/adjustment_factors_{i1}-{i1+step}.csv') 
