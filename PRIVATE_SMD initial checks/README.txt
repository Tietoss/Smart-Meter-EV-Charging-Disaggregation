Author: Lukas Elmiger, luki.elmiger@hotmail.com
Date of initial commit: 21.10.2024

- read_export.py is used to get a grasp of the real SMD (Smart Meter Data) and its quality. 
It reads .csv or .gz files and generates useful plots, as can be seen in the results directory of this repository.

- private_smd_preprocessing.py: Based on the results of read_export.py, we can now filter the SMD for the data which is of interest (in this case, residential customers). 

- EV_charging_session_identification: Main script, detects EV charging sessions in aggregated SMD and plots important metrics. 

- spatio_temporal_plotting: Generates map plot, movie for charging activity. Requires .shp files (for DSO area), (approximated) GPS-locations for each ID and identfied charging time series.
