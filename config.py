# canada_bounds_rotated, for NUMBERED INDEXING ONLY
canada_bounds_rotated_index = dict(rlon=slice(20, 145),
                                   rlat=slice(50, 130)) # 130 is max
# outer lat-lon bounds of "canada_bounds_rotated_index" selection: lat: 34.05, 75.66
#                                                                  lon: -159.87, -28.46
                                     
# approximate lat-lon bounds of Canada domain
canada_bounds = {'lat': slice(42.25, 75.5), # NAM44 grid has max 76.25, but CanRCM4 run on smaller grid w max latitude of 75.6
                 'lon': slice(-144.75, -50.25)}

# wider domain to clip before regridding, clip to regular canada_bounds post-regrid
canada_bounds_wide = {'lat': slice(38, 76.25), 
                      'lon': slice(-150, -45)}

# LV: currently not used, but could be considered for final data
def get_data_packing(vals, n=16): # data packing params if wanting to save as int16
    # as per: https://www.unidata.ucar.edu/software/netcdf/workshops/2010/bestpractices/Packing.html
    add_offset = vals.min().values
    scale_factor = (vals.max().values - vals.min().values) / (2**(n - 1))
    return {'add_offset': add_offset, 
            'scale_factor': scale_factor,
            '_FillValue': 32767, # int16 specific params
            'dtype': 'int16'}