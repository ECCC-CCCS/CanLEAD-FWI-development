'''
Apply domain mask (Canada, excluding the high Arctic) to annual and climatological metric files.
Add missing attributes and update filenames.
'''

import xarray as xr
import sys
import os
import glob
import tqdm
from filepaths import fwipaths

## Mask annual metric files with Canada mask, excluding northern Arctic ecozone. 30 year and annual
rcp = sys.argv[1] # one of: 'RCP85', 'constructed_RCP26', 'constructed_RCP45'        
test_statistics = ['MJJAS_mean_fillna',
                   'MJJAS_quantile_fillna',
                   'fire_season_length',
                   'exceedances_moderate',
                   'exceedances_high',
                   'exceedances_extreme',
                   'annual_exceedances_1971_2000_MJJAS_95th_quantile_fillna'
                   ]

final_mask = xr.open_dataset(f'{fwipaths.input_data}/CanLEAD_FWI_final_mask.nc')['CanLEAD_FWI_mask'] # get final mask to apply

for test_stat in test_statistics: 
    
    outpath = f'{fwipaths.output_data}CanLEAD-FWI-EWEMBI-v1/summary_stats/{rcp}/{test_stat}/'
    fls = glob.glob(f'{outpath}no_mask/*{test_stat}.nc') # for annual files in directory

    for fl in tqdm.tqdm(fls):

        ds = xr.open_dataset(fl)

        # apply mask and add RCP to all metric files
        ds = ds.where(final_mask==100) # mask with Canadian boundaries and ecozone mask
        # add RCP as attr
        if rcp == 'RCP85':
            ds.attrs['rcp'] = 'RCP8.5' 
        elif rcp == 'constructed_RCP26':
            ds.attrs['rcp'] = 'Constructed RCP2.6' 
        elif rcp == 'constructed_RCP45':
            ds.attrs['rcp'] = 'Constructed RCP4.5' 
        
        # add RCP to filename to some files where it's missing
        if rcp == 'RCP85':
            name_components = os.path.basename(fl).split('_')
            name_components.insert(2, 'rcp85') # insert RCP into filename
            if test_stat in ['exceedances_moderate',  'exceedances_high', 'exceedances_extreme']:
                name_components.remove('annual')
            outname = '_'.join(name_components) 
        else: 
            outname = os.path.basename(fl)
        ds.to_netcdf(outpath + outname) # save file