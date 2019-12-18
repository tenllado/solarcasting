This file explains which files and in which order to execute, including the different jsons associated.

0 - estaciones/stations_map.py (Optional)
	This file plots the map with the stations labeled where it belongs.
	It computes the distance in meters providing latitude and longitude in stations.txt

1 - generacionMatrices/merger.py
	This file gets the raw data (previously downloaded from https://midcdmz.nrel.gov/oahu_archive/ ).
	It computes the azimuth, elevation and the theoretical GHI for every station to get the GHI relative.
	It creates a single file for every day into a folder for every station.
	merger.json file to select:
		Input/Output folders.
		Clear Sky Model to use (ideally Haurwitz).
		Hours to take into account (for Hawaii, ideally 7:30-17:29).
		Station names and latitude and longitude.

2 - generacionMatrices/scores.py
	This file computes different scores for every day in every station to "evaluate" the cloudiness of it.
	The scores are the average difference from the max value to every value and the average points over some
	arbitrary thresholds.
	scores.json to select:
		Thresholds.
		Folders and files to load.

3 - generacionMatrices/plots.py (Optional, recommended)
	This file plots the theoretical relative GHI (always 1) and the real rel GHI for every day for every station.
	It also adds the scores for every day to easily compare the values and the graphs.
	config_plots.json to select:
		Use of relative or real values.
		Folders and files to load.

4 - generacionMatrices/nts.py
	This file creates the matrices ready to be trained in every ML algorithm.
	It creates a single file for every day.
	(originally the number of the file was because of the Number of Time Samples we wanted to select)
	nts_**.json (i.e. nts_dh6_haurwitz.json) to select:
		Target station.
		Stations to compute.
		Number of samples.
		Time granularity.
		Offset.
		Latitude and longitude for every station.
		Aggregation (mean or skip).
		Folders and files to load.

5 - generacionMatrices/nts_der.py (Optional)
	This file adds another feature, the "derivative", the variation between samples.
	--ExperimentFolder to select the folder where nts.py saved the files.

6 - generacionMatrices/NN_matrices.py
	This file merges randomly the different days into a X_tr_val & Y_tr_val and X_test & Y_test.
	It takes into account the score for an arbitrary threshold to remove the "cloudless" or "weird" days.
	It creates different files:
		days_info.csv:
			Number of samples for a day.
			Number of days.
			Number of days for test and train and seed.
		test_days.csv, train_days.csv, total_days.csv: the date of every day of the set.
	nnmatrices_**.json (i.e. nnmatrices_paper.json) to select:
		Folders and files to load.
		Threshold.
		Whether it takes the "derivative" or not.
	
---------------------------------

After that all the matrices are ready to train selecting different models. 
Check, for example, ../ModelosGuillermoYepes for different models (NN, RegForest...) that could be trained with the output of the last script
