# CanLEAD-FWI
Fire weather projections for Canada

Scripts to create CanLEAD-FWI-v1. To be run in order presented below.

---------------------- IN FOLDER: noontime_estimates ---------------------- 

utc_sunrise_noon.py
  Find timing (time offset in decimal hours) of sunrise and solar noon in UTC for each grid cell,
  for both CanLEAD and CanRCM4 grids from 1950-2100.

diurnal_estimates.py
  Determine unknown temperature offset parameters (offset of tmin and tmax from sunrise and solar noon, respectively);
  required to estimate noontime values of FWI inputs.
  Determine values from CanRCM4 hourly temperature data.

regrid_diurnal_estimates.py 
  Take ensemble average of temperature offset parameters found in diurnal_estimates.py.
  Regrid from CanRCM4 grid to CanLEAD grid (NAM-44 to NAM-44i) using nearest neighbour. 
  Regrid land-sea and land-glacier mask of CanRCM4 to CanLEAD grid using nearest neighbour.	

calculate_rh_noon_t.py 
  Calculate noontime estimated values of temperature and RH from daily maximum and minimum temperature and daily average RH, 
  using temperature offset parameters and time of solar noon determined above. 

----------------------  IN FOLDER: main ---------------------- 

config.py
  Canada bounds definitions, data packing (scale/offset). Does not need to be run.

calculate_gridded_fwi.py
  Calculate gridded Canadian Forest Fire Weather Index System projections using xclim. 
