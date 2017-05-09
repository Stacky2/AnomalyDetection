# AnomalyDetection
Code of my master thesis "A Comparison of Mixed-Type Anomaly Detection Methods"


What is it?
-----------
This is a test environment for anomaly detection algorithms. 

The main files are "TestNoveltyMethods.py" and "TestOutlierMethods.py" which use different other modules to test the algorithms. When these scripts get executed they use (mainly) the configuration parameters which are located in the same folder and that have the name "config_Nov.ini" or "config_Out.ini" respectively. The terms "Novelty" and "Outlier" refert to the scenario in which the models are used, 
i.e. novelty detection / semi-supervised anomaly detection or an outlier detection / unsupervised anomaly detection 
scenario.

A rought overview over the dependencies of the files can be found in the document "dependencies.pdf".


Folders
-------
- AnomalyModels: Contains the python file "AnomalyModels.py"
  which contains the classes for the different anomaly models
  and a "config.ini" file that contains the default parameters
  for the anomaly detection algorithms.

- data: Contains folders with the names of the data sets. In
  each of these folders there is a "data.csv" file that
  contains the data set. Further the folder data contains a
  "config.ini" file, in which contains important information
  about the different data sets (eg. name of the label column:
  "label_col" or a list of the names of the categorical
  features: "fac_cols".

- DataGeneration: contains a python file "DataGeneration.py"
  that contains the classes for the data models with functions
  that allow to simulate data for the anomaly detection tasks.
  Further the folder contains a "config.ini" file which 
  contains the details for the data simulation of each data
  set.

- DataSplit: contains a python file "DataSplit.py" that
  contains functions that allow to split data for the novelty 
  and outlier scenario.

- Log: is the folder where the output files get saved after
  the main file "TestNoveltyMethod.py" or
  "TestNoveltyMethod.py" has been executed.

- OptimizeParameters: contains a python file
  "OptimizeParameters.py" that handles the optimization of the
  hyper parameters of the anomaly models. Further there is a 
  "config.ini" file in the folder that contains the parameters
  for the bayesian optimization algorithm as well as the
  ranges in which the optimization algorithm should optimize 
  the parameters of the anomaly detection models.

- RepresentationModels: Contains the python file 
  "RepresentationModels.py" which contains the classes of the
  representation models. 

- Visualization: contains the python file "Visualization.py"
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
Version 1.0
