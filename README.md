# Semantic-Kernel-Plugin-for-Multi-Time-Series-Problem
## Introduction
This is to show how we can use Semantic Kernel and its plug-in mechanism to conduct analysis based multiple data sources and multiple time series with defined business logic and well-established time series analytic tool/model to generate a Root Cause Analysis (ROA) report with Action Item suggestion. The use case is a portion of a drinking water quality management system, which identifies the root cause of pH value lower than threshold due to temperature and suggests the best treatment based on the time series Pearson correlation coefficient analysis. 

## Repo Structure
This repo has the following parts:
- light_plugin.py: this is the semantic kernel model to access GTP-4o with a plug-in class CVS which read in all the data we need and conduct the time series correlation analysis.
- sk_agent_dep_inj.py: this is the semantic kernel based and several commonly used services are injected and used by agent with access to GTP-4o from completion. Compared with plugin approach, DI/Agent give you more control on software stacking and flows of data and work.
- app.py: a Flask API framework which synchronically calls the main() of light_plugin on binder of a user call.
- report.html: A user uses this file to call the Flask API to get the ROA report in its External Status column.
- sensor_config.csv
- sensor_calibration.csv
- time_series_data.csv
- The ROA report is generated by the following steps:
1. Use function read_sensor_config read the content from the CSV file from .\data\sensor_config.csv. This file has four columns: ID, Measure, LowAlarm, UpperAlarm. This file has two rows. The first row is the column names, and the second row is the values.
2. Use function read_sensor_calibration read the content from the CSV file from .\data\sensor_calibration.csv. This file has three columns: ID, Measure, Status. This file has two rows. The first row is the column names, and the second row is the values.
3. Use function time_series_data_1 read the content from the CSV file from .\data\time_series_data_1.csv. This file has six columns: Time, Temp, Tr1, Tr2, Tr3, pH. This file has 26 rows. The first row is the column names, and the other rows are hourly measurements. After reading this function will compute the correlation coefficients between the Temp and pH, Tr1 and pH, Tr2 and pH, and Tr3 and pH from the time series.
4. Once all the data being collected, conduct the following analysis:
    1. If the last three pH values are lower than LowAlert, and pH highly correlated to the Temp (say correlation efficient is greater than 0.9), and the calibration status is Normal, Then generate a HTML report that contains a table of 3 rows and 2 columns. The first rows have two cells. One has a string 'Subject' and the other cell has a string 'pH Low ROA: Low Temperature'. The second row has two cells. One has a string 'Root cause analysis' and in the other cell you state that the system is in Abnormal status and the pH values are lower than the threshold due to the low temperature. The third row has two cells. One has a string 'Suggested Actions' and in the other cell you suggest to increasing the Treatment Tr1, Tr2 or Tr3 which has the highest correlation with pH. Remember to include the treatment name and its correlation coefficient with pH in your suggestion.
    2. Otherwise, generate the same HTML report. The first rows have two cells. One has a string 'Subject' and the other cell has a string 'pH Low ROA: Unknown'. The second row has two cells. One has a string 'Root cause analysis' and in the other cell you state that the system is in Abnormal status and the pH values are lower than the threshold but may not be due to the low temperature. The third row has two cells. One has a string 'Suggested Actions' and in the other cell you suggest to increasing the Treatment Tr1, Tr2 or Tr3 which has the highest correlation with pH. Remember to include the treatment name and its correlation coefficient with pH in your suggestion.
- Demoware.ppt: A short discussion on this project.

## Reference
~~~
https://www.healthline.com/health/ph-of-drinking-water
https://www.wikihow.com/Raise-the-pH-of-Water
https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/?pivots=programming-language-python
~~~

