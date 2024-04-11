'''
Calculate climatological (30-year) means for RCP85.
For constructed RCPs 2.6 and 4.5, this is completed during the GWL translation process
'''

# Set up code
import xarray as xr
import numpy as np
import glob
import sys
from filepaths import fwipaths
import gc
import subprocess
tracking_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

test_statistics = ['fire_season_length',
                   'MJJAS_quantile_fillna',
                   'exceedances_high',
                   'exceedances_extreme',
                   'exceedances_moderate',
                   'MJJAS_mean_fillna',
                   'annual_exceedances_1971_2000_MJJAS_95th_quantile_fillna'
                   ]
 
version = f'CanLEAD-FWI-{sys.argv[1]}-v1' # S14FD or EWEMBI

def take_climatological_mean(ds, frequency):
    '''
    Takes climatological (30 year) mean of input dataset, and appends that to attributes
    in 'cell_methods'.

    Parameters
    ----------
    ds : xarray dataset
        Dataset containly values in MONTHLY or ANNUAL frequency.
    frequency : String
        Valids values: "monthly" or "annual". Describes data frequency.

    Returns
    -------
    ds : xarray dataset
        Dataset, with annual or monthly values now as climatological means.

    '''
    # If frequency is seasonal instead of annual, unstack the time dimension into year and season dims
    # then take climo mean across all years, by season
    # For annual frequency data, simply rename time dim to year, take climo mean
    if frequency == 'seasonal': # if the data is by season, not year 
        # assign year and season as coords, instead of time dim
        year = ds.time.dt.year
        season = ds.time.dt.season
        # assign new coords
        ds = ds.assign_coords(year=("time", year.data), season=("time", season.data))
        # replace time with 'year' and 'month' as dims
        ds = ds.set_index(time=("year", "season")).unstack("time")  
    elif frequency == 'annual':
        ds['time'] = ds.time.dt.year.values
        ds = ds.rename({'time': 'year'})
    # take 30 year rolling means, labelled to the RIGHT EDGE of the window
    ds = ds.rolling(year=30, center=False).mean(keep_attrs=True)
    # define windows to keep, using the end year of the 30 year window ending each decade
    windows_keep = np.arange(1980,2101,10)
    ds = ds.sel(year=ds.year.isin(windows_keep)) # select only years we're interested in (by right-side labels), 1950 to 1970 labels are incomplete (NaN) groups
    ds = ds.rename({'year': 'period'}) # rename to reflect climatology
    ds['period'] = [f'{ii-29}-{ii}' for ii in ds.period.values] # relabel appropriate to windows
    for var in ds.data_vars: # append climatological mean as method in attrs
        ds[var].attrs['cell_methods'] = ds[var].attrs['cell_methods'] + ' time: mean over years' # in format: time: method1 within years time: method2 over years
    del(ds.attrs['frequency'])
    return ds

for test_stat in test_statistics: 
    
    outpath = f'{fwipaths.output_data}{version}/summary_stats/RCP85/{test_stat}/' # set input and output file directory
    fls = glob.glob(f'{outpath}/*_{version}_{test_stat}.nc') # get list of all annual frequency metrics files
    
    for fl in fls: 
     
        ds = xr.open_dataset(fl)
         
        # set encoding
        encoding = {var: {'dtype': 'float32', '_FillValue': 1e+20} for var in ds.data_vars}
        for var in ['lat','lon']: 
            encoding[var] = {'dtype': 'float64', '_FillValue': None}  # for lat, lon and time
                                                    
        # take 30 year averages, add tracking attrs, and save
        out30 = take_climatological_mean(ds, 'annual') # take climatological mean and update attrs via config function. All final stats are annual in frequency
        out30.attrs['history'] = f'Generated by {sys.argv[0]}'
        out30.attrs['git_id'] = tracking_id
        out30.attrs['git_repo'] = 'https://github.com/ECCC-CCCS/CanLEAD-FWI/' 
        
        out30.to_netcdf(f'{outpath}/{ds.realization.values[0]}_rcp85_{version}_{test_stat}_30yr_mean.nc', encoding=encoding) # save
        
        del([out30,ds])
        gc.collect()