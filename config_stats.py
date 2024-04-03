import numpy as np
import xarray as xr

stats_chunks = {'lat': 10, 'lon': 32, 'time': 9186}

def add_realization_dim(out_dataset):
    # get realization label from file attrs, which are standardized from CanLEAD
    realization_label = out_dataset.attrs['CanLEAD_CanRCM4_experiment_id'][-2:] + '_' + out_dataset.attrs['CanLEAD_CanRCM4_driving_model_ensemble_member']
    # add realization as a dimension on the dataset
    out_dataset = out_dataset.assign_coords(realization=realization_label).expand_dims('realization')
    # add associated attrs to new dimension
    realization_attrs = dict(standard_name = 'realization',
                             long_name = 'Ensemble member',
                             units = '')  
    out_dataset['realization'].attrs = realization_attrs
    return out_dataset, realization_label

def get_MJJAS_data(ds):
    fire_seas = [5,6,7,8,9] # define central fire season
    # create an dataset with an indentical time dimension to ds,
    # but where the time dimension contains only the month instead of the full date
    seasons = xr.full_like(ds.time.dt.month, fill_value='', dtype='U7')
    seasons.name = 'season'
    # if month is in MJJAS as defined in fire_seas, replace month with MJJAS flag
    seasons[ds.time.dt.month.isin(fire_seas)] = 'MJJAS'
    seasons[~ds.time.dt.month.isin(fire_seas)] = 'ONDJFMA' # else, replace with winter flag
    out_MJJAS = ds.groupby(seasons)['MJJAS'] # grouby seasons, then select only the MJJAS group
    out_MJJAS = out_MJJAS.fillna(0) # fill nans with zeros, so that season length will remain the same over the 21st century
    # if NaNs are not filled with zeros, then season length will grow over time (as no values are returned in the FWI System
    # in the winter period). If not fixed, changing season length will affect stats (means and percentiles), skewing comparisons between future and historical
    return out_MJJAS

def take_climatological_mean(ds, frequency):
    '''
    Takes climatological (30 year) mean of input dataset, and appends that to attributes
    in 'cell_methods'. Input datasets contain the entire 150 years of data.

    Parameters
    ----------
    ds : xarray dataset
        Dataset containly values in MONTHLY or ANNUAL frequency.
        This contains 150 years of data.
    frequency : String
        Valids values: "monthly" or "annual". Describes data frequency.

    Returns
    -------
    ds : xarray dataset
        Dataset, with annual or monthly values now as climatological means.

    '''
    # If frequency is seasonal instead of annual, unstack the time dimension into
    # year and season dims, then take climo mean of season only
    if frequency == 'seasonal': # if the data is by month, not year
        # assign year and month as coords, instead of time dim
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
    
def take_climatological_mean_pseudo_rcps(ds, frequency):
    '''
    Takes climatological (30 year) mean of input dataset, and appends that to attributes
    in 'cell_methods'. Input datasets should contain only one climatological period.

    Parameters
    ----------
    ds : xarray dataset
        Dataset containly n-year time slice of values in MONTHLY, SEASONAL or ANNUAL frequency.
        This contains only ONE climatological period.
    frequency : String
        Valids values: "monthly" or "annual". Describes data frequency.

    Returns
    -------
    ds : xarray dataset
        Dataset, with annual or monthly values now as climatological means for ONE climatological period.

    '''
    # If frequency is seasonal instead of annual, unstack the time dimension into
    # year and season dims, then take climo mean of season only
    if frequency == 'seasonal': # if the data is by season, not year
        # assign year and month as coords, instead of time dim
        year = ds.time.dt.year
        season = ds.time.dt.season
        # assign new coords
        ds = ds.assign_coords(year=("time", year.data), season=("time", season.data))
        # replace time with 'year' and 'month' as dims
        ds = ds.set_index(time=("year", "season")).unstack("time")  
    elif frequency == 'monthly': # if the data is by month, not year
        # assign year and month as coords, instead of time dim
        year = ds.time.dt.year
        month = ds.time.dt.month
        # assign new coords
        ds = ds.assign_coords(year=("time", year.data), month=("time", month.data))
        # replace time with 'year' and 'month' as dims
        ds = ds.set_index(time=("year", "month")).unstack("time")  
    elif frequency == 'annual':
        ds['time'] = ds.time.dt.year.values
        ds = ds.rename({'time': 'year'})
    ds = ds.mean(dim='year', keep_attrs=True)
    for var in ds.data_vars: # append climatological mean as method in attrs
        ds[var].attrs['cell_methods'] = ds[var].attrs['cell_methods'] + ' time: mean over years' # in format: time: method1 within years time: method2 over years
    del(ds.attrs['frequency'])
    return ds