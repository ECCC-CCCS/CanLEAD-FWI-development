"""
Regrid temperature parameters (offset of tmin and tmax from sunrise and solar noon, respectively) 
determined from hourly CanRCM4 outputs. 
Regrid from CanRCM4 grid to CanLEAD grid (NAM-44 to NAM-44i) using nearest neighbour. 
"""

import xarray as xr
import xesmf as xe
import sys
import os
import subprocess
sys.path.append(os.path.expanduser('~/fwi_updates/CanLEAD-FWI-v1/'))
from filepaths import fwipaths
from config import canada_bounds, canada_bounds_rotated_index, canada_bounds_wide
from scipy import stats
import numpy as np
import glob
import gc 
tracking_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

#%% Take ensemble statistics of hmin and hmax offsets determined in diurnal_estimates.py

fls = glob.glob(f'{fwipaths.working_data}offsets_tmin_tmax/offsets_tmin_tmax_month_1971_2000_*.nc')
alldata = xr.open_mfdataset(fls).chunk({'nens': -1, 'rlat': 10, 'rlon':10}) # will concat by ensemble member dimension added in previous script

out = alldata.reduce(stats.circmean, high=24, low=0, dim='nens', keep_attrs=True) # take the circular mean across all 15 realizations of timing of offsets from max and min temp (hmax and hmin), keep variable attrs

# Add a couple of additional attrs, most are copied from input files 
out.attrs = alldata.attrs
out.attrs['input_files'] = fls
out.attrs['details'] = 'All-realization (50 member ensemble) circular mean of offsets between solar noon and maximum temperature, and sunrise and minimum temperature, by month'
encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 4} for var in out.data_vars} # save as float32 and compress to save space
out.to_netcdf(f'{fwipaths.working_data}CanRCM4_offsets_tmin_tmax_month_1971_2000_all_realization_circmean.nc')
del([out, alldata, fls, encoding])
gc.collect() # free up space

#%% Regridding 

# Land and glacier masks, represent percentage area fraction (coverage) but have only two possible values: 0 or 100 
land_mask = xr.open_dataset(fwipaths.input_data + "CanRCM4/NAM-44_CCCma-CanESM2_historical-r1/fx/atmos/sftlf/"\
                            +"r1i1p1/sftlf_NAM-44_CCCma-CanESM2_historical-r1_r1i1p1_CCCma-CanRCM4_r2_fx.nc").isel(**canada_bounds_rotated_index).squeeze()
glac_mask = xr.open_dataset(fwipaths.input_data  + "CanRCM4/NAM-44_CCCma-CanESM2_historical-r1/fx/atmos/sftgif/"\
                            + "r1i1p1/sftgif_NAM-44_CCCma-CanESM2_historical-r1_r1i1p1_CCCma-CanRCM4_r2_fx.nc").isel(**canada_bounds_rotated_index).squeeze()

# Import sample CanLEAD file to obtain target NAM-44i grid, clip to boundary larger than desired final output
flnm_a = '_NAM-44i_CCCma-CanESM2_rcp85_'
flnm_b = '_CCCma-CanRCM4_r2_ECCC-MBCn-S14FD-1981-2010_day_19500101-21001231.nc'
CanLEAD_NAM44i = xr.open_dataset(f"{fwipaths.input_data}CanLEAD/CanRCM4-S14FD-MBCn/r1_r1i1p1/prAdjust{flnm_a}r1_r1i1p1{flnm_b}").sel(**canada_bounds_wide).isel(time=0).squeeze() 
    
# Create regridder to convert land and glacier masks, at this point we are only establishing the grid specs
regridder = xe.Regridder(ds_in=land_mask, # land_mask defines the input dataset GRID only
                         ds_out=CanLEAD_NAM44i, # CanLEAD_NAM44i defines the output GRID only
                         method='nearest_s2d') # nearest neighbour interpolation

CanLEAD_land = regridder(land_mask) # regrid land
CanLEAD_glac = regridder(glac_mask) # regrid glacier

# add binary mask to both input and output grids, and recreate regridder. This will be considered automatically considered by xesmf regridder when regridding
#  0 = do not consider, 1 = consider in regridding
land_mask['mask'] = xr.where((land_mask.sftlf == 100) & (glac_mask.sftgif == 0), 1, 0) # values of 1 represent areas of land only (not water or glacier) for original grid
CanLEAD_NAM44i['mask'] = xr.where((CanLEAD_land.sftlf >= 60) & (CanLEAD_glac.sftgif <= 40), 1, 0) # values of 1 represent areas of land only (> 70% land and <30% glacier) for CanLEAD grid

# create new regridder with mask
regridder_mask = xe.Regridder(ds_in=land_mask, # land_mask defines the input dataset GRID only and MASK
                              ds_out=CanLEAD_NAM44i, # CanLEAD_NAM44i defines the output GRID only and MASK
                              method='nearest_s2d') # use nearest neighbour interpolation, required as we cannot do a 'circular mean interpolation' for this data
                           
# Import min and max temperature timing params
hmax_hmin = xr.open_dataset(f'{fwipaths.working_data}CanRCM4_offsets_tmin_tmax_month_1971_2000_all_realization_circmean.nc')

# Regrid timing of tmax and tmin 
CanLEAD_hmax_hmin_masked = regridder_mask(hmax_hmin).sel(canada_bounds)

# Add attrs, and replace those lost through regridding
for var in CanLEAD_hmax_hmin_masked.data_vars:
    CanLEAD_hmax_hmin_masked[var].attrs = hmax_hmin[var].attrs
CanLEAD_hmax_hmin_masked.attrs = hmax_hmin.attrs
CanLEAD_hmax_hmin_masked.attrs['regrid_git_id'] = tracking_id
CanLEAD_hmax_hmin_masked.attrs['regrid_history'] = f"Generated by {os.path.basename(sys.argv[0])}"
CanLEAD_hmax_hmin_masked.attrs['regrid_description'] = f'Regrid from NAM-44 to NAM-44i using xesmf {xe.__version__}. regrid_method: nearest_s2d.'

# Save regridded 
encoding = {var: {'dtype': 'float32'} for var in CanLEAD_hmax_hmin_masked.data_vars} # save as float32 and compress to save space
CanLEAD_hmax_hmin_masked.sel(canada_bounds).to_netcdf(f'{fwipaths.working_data}/CanLEAD_offsets_tmin_tmax_month_1971_2000_all_realization_circmean.nc', encoding=encoding) 

# Convert temp_offsets grouped by day of year, to a dataset with time dimensions matching other input variables when used in calculate_noon_rh_t.py

def generate_full_time_array(offset_data, template_data):
    days_in_month = template_data.isel(lat=1, lon=1).squeeze().time.dt.daysinmonth.groupby('time.month').first() # get array of days in month
    offset_data_year = np.repeat(offset_data, 
                                 days_in_month,
                                 axis=np.argmin(offset_data.shape)) # repeat values based on days in each month, e.g., np.repeat([1,2,3],[4,1,2]) returns [1,1,1,1,2,3,3], along the smallest axis ('month')
    offset_data_all = np.tile(offset_data_year,
                              (151,1,1)) # tile values for each of 151 years , e.g., np.tile([1,2,3],2) returns [1,2,3,1,2,3]
    return offset_data_all

template = xr.open_dataset(f"{fwipaths.input_data}CanLEAD/CanRCM4-S14FD-MBCn/r1_r1i1p1/prAdjust{flnm_a}r1_r1i1p1{flnm_b}").sel(**canada_bounds)
CanLEAD_hmax_hmin_masked_full = xr.Dataset(data_vars=dict(hmax_offset=(["time", "lat", "lon"], 
                                                                        generate_full_time_array(CanLEAD_hmax_hmin_masked.hmax_offset.values, template)),
                                                          hmin_offset=(["time", "lat", "lon"], 
                                                                        generate_full_time_array(CanLEAD_hmax_hmin_masked.hmin_offset.values, template))), 
                                          coords=template.coords)

# Double check monthly values are all equal, as they should be
xr.testing.assert_allclose(CanLEAD_hmax_hmin_masked.drop('time'), 
                           CanLEAD_hmax_hmin_masked_full.groupby('time.month').mean(),
                           rtol=1e-4, atol=1e-4) # will raise assertion error if false 

for var in CanLEAD_hmax_hmin_masked_full.data_vars:
    CanLEAD_hmax_hmin_masked_full[var].attrs = CanLEAD_hmax_hmin_masked[var].attrs
CanLEAD_hmax_hmin_masked_full.attrs = CanLEAD_hmax_hmin_masked.attrs                              
                              
CanLEAD_hmax_hmin_masked_full.to_netcdf(f'{fwipaths.working_data}/CanLEAD_offsets_tmin_tmax_1971_2000_all_realization_circmean_1950_2100_daily.nc', encoding=encoding) 

# Save land and ocean masks
CanLEAD_land.sel(canada_bounds).to_netcdf(f'{fwipaths.working_data}/CanLEAD_sftlf_nearest.nc') 
CanLEAD_glac.sel(canada_bounds).to_netcdf(f'{fwipaths.working_data}/CanLEAD_sftgif_nearest.nc')
    


