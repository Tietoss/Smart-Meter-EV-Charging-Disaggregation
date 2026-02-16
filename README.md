# 2442_Lukas_Elmiger Study of the EV Charging Behavior and its Impact on Real Low/Medium Voltage Power Grids

## Description
This project contains all files belonging to my master thesis on the "Study of the EV Charging Behavior and its Impact on Real Low/Medium Voltage Power Grids".
In my thesis, I investigated the EV charging behavior in real smart-meter data. The smart-meter data also contained other residential appliances. Therefore, an algorithm was developed to extract charging from aggregated load signals. Having extracted the EV charging load time series, a statistical analysis was performed to extract certain interesting metrics for the identified load time series. Finally, different charging strategies and their impact were investigated using Monte-Carlo simulations. 

The project is organized as follows:
- CKW Open Data Quality Check: Name is a bit misleading - contains files that do much more than just a quality check on CKW Open data. Files to preprocess open data, add synthetic charging sessions and optimization routine to find best parameters for session identification.
- EV_charging_optimization: Given some extracted charging timeseries, we can investigate the effect of different EV charging strategies from a DSO perspective. No "optimization" is done here - it's only an assessment of different strategies using parallelized Monte Carlo simulations
- PRIVATE_SMD initial checks: On the private smart meter data that was used to analyze real customer behavior, some preprocessing had to be done. 
- Smart_Charging_App_CKW: At the beginning, the CKW Smart Charging App (controlling charging sessions of certain customers) was available. After preprocessing, it turned out, that the data is not representing typical customer behavior.
- Weekly meeting: Contains the slides of the weekly exchange with the supervisor + monthly exchange with the supervisors from the company. 

## Installation
Python 3.11,
also see requirements file

## Support
lukas.elmiger@ckw.ch
luki.elmiger@hotmail.com

Feel free to contact me - I will try to answer questions related to this code and/or my thesis.

## Authors and acknowledgment
Special thanks to my project supervisor, María Parajeles Herrera from PSL ETH Zurich.

## License
MIT License

Copyright (c) [2025] [Lukas Elmiger]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
