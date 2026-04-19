========================================================================
README: Beijing Multi-Site Air Quality Dataset Analysis
========================================================================

1. DATASET OVERVIEW
-------------------
This dataset contains hourly atmospheric data from 12 nationally-controlled 
air-quality monitoring sites in Beijing. The data covers a 4-year period 
from March 1st, 2013, to February 28th, 2017.

- Characteristics: Multivariate, Time-Series
- Subject Area: Climate and Environment
- Total Instances: 420,768 (aggregated from all sites)
- Missing Values: Yes (denoted as 'NA')

2. DATA SOURCES
---------------
- Air Quality Data: Beijing Municipal Environmental Monitoring Center.
- Meteorological Data: China Meteorological Administration.

3. VARIABLE DESCRIPTIONS
------------------------
No: row number 
year: year of data in this row 
month: month of data in this row 
day: day of data in this row 
hour: hour of data in this row 
PM2.5: PM2.5 concentration (ug/m^3)
PM10: PM10 concentration (ug/m^3)
SO2: SO2 concentration (ug/m^3)
NO2: NO2 concentration (ug/m^3)
CO: CO concentration (ug/m^3)
O3: O3 concentration (ug/m^3)
TEMP: temperature (degree Celsius) 
PRES: pressure (hPa)
DEWP: dew point temperature (degree Celsius)
RAIN: precipitation (mm)
wd: wind direction
WSPM: wind speed (m/s)
station: name of the air-quality monitoring site

4. SCRIPT FUNCTIONALITY (Code Overview)
---------------------------------------
The provided Python script is designed to handle multiple monitoring sites 
automatically by performing the following operations:

- FILE DISCOVERY: Uses 'glob' to scan the directory for all .csv files.
- DATA AGGREGATION: Iteratively loads each file and uses 'pd.concat' to 
  merge them into a single unified DataFrame.
- DATA VALIDATION: 
    * Displays the first 5 rows to confirm correct merging.
    * Uses '.info()' to audit data types and non-null counts.
    * Reports final dimensions (Total Rows vs. Columns).