#Anomaly Detection Test Program

What is it?
-----------
This is a test environment for anomaly detection algorithms. 

[NOTE: PROGRAM NOT YET EXECUTABLE, BECAUSE DATA.CSV FILES STILL MISSNG]

For the actual test of the methods on a dataset we use the test function "execute_job" which is placed in the file "TestAnomalyMethods.py" in the program folder. The function "execute_job(config_name)" executes the anomaly detection test, using the parameters described in the configuration file 
[config\_name].ini, which needs to be located in the "config"-folder of the 
program. 

When starting the python file "GUI.py", a window opens up that can be used for performing tests. On can specify the different parameters of the test in the corresponding tabs. Clicking the "Run!"-button in the "Run" tab triggers the program to collect
the information from the different tabs and to form a configuration file for the test. This file gets saved in the "configs"-folder and then the test is executed with the "execute\_job" function mentioned above.

A rought overview over the dependencies of the files can be found in the document "dependencies.pdf".


Folders
-------
- AnomalyModels: Contains the python file "AnomalyModels.py"
  which contains the classes for the different anomaly models
  and a "config.ini" file that contains the default parameters
  for the anomaly detection algorithms.

- data: Contains folders with the names of the data sets. In
  each of these folders there is a "data.csv" file that 
  contains the corresponding data set. Further the folder
  {data} contains a "config.ini" file, in which important
  information about the different datasets (eg. name of the
  label column: "label_col" or a list of the names of the
  categorical features: "fac_cols") is stored.

- DataGeneration: Contains a python file "DataGeneration.py",
  that contains the classes for the data models with functions
  that allow to simulate data for the anomaly detection tasks.
  Further the folder contains a "config.ini" file which 
  contains the details for the data simulation of each data 
  set.

- AnomalyDataSet: Contains a python file "AnomalyDataSet.py",
  that contains a class for anomaly detection datasets with
  different functions that allow for example to generate
  train-test-splits.

- Log: This is the folder where the output files get saved,
  after the test method "execute_job" has been executed.

- OptimizeParameters: Contains a python file
  "OptimizeParameters.py", that handles the optimization of
  the hyper parameters of the anomaly models. Further there is
  a "config.ini" file in the folder, that contains the
  parameters for the Bayesian optimization algorithm as well
  as the ranges, in which the optimization algorithm should 
  optimize the parameters.

- RepresentationModels: Contains the python file 
  "RepresentationModels.py", which contains the classes of the
  representation models. 

- Visualization: Contains the python file "Visualization.py",
  that contains the functions to visualize data.


Dependencies
------------
The following things need to be installed:
- Anaconda 2.7
- h2o-3
- Bayesian Optimization python module
  type in anaconda prompt: pip install bayesian-optimization
- Least-squares anomaly python detection module 
  type in anaconda prompt: pip install lsanomaly


Version
-------
Version 1.2
