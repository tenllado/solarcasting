This file explains which files and in which order to execute, including the different jsons associated.

Once the input matrix (for train and test, including labels) is generated using the scripts in ../TratamientoDatos,
you are ready to train selecting different models:

-NEURAL NETWORKS:

1 - NN_models.py
	This file sets the architecture of the network and trains the data already prepared.
	It saves the models and trained days.
	It computes different metrics for the global set (not individual days).
	NN_models_**.json (i.e. NN_models_300_300_300.json) to select:
		Hidden layers and its neurons.
		Stations to select as features.
		Regularization factor (Alpha).
		Train size.
		Seed.
		Time granularity.
		Forecasting horizon.
		Folders and files to load.
2 - NN_prediction_graphs.py
	This file loads the saved models and applies it to the test subset, as well as it plots any number of days.
	It computes the "skill" as well as other metrics for every single day. It saves it as a daily_skills_**.csv.
	NN_prediction_graphs.json to select:
		Hidden layers and its neurons.
		Stations to select as features.
		Regularization factor (Alpha).
		Train size.
		Seed.
		Time granularity.
		Forecasting horizon.
		Folders and files to load.
	If the network architecture doesn't belong with the models saved it will return an error.
3 - skill_plot.py
	This file loads the skill_report files and plots the skill vs forecasting prediction.
	skill_plot.json to select:
		Architecture of the neural network.
		Folders and files to load.
4 - skill_test_days.py
	This file loads the daily reports for the test days in NN_prediction_graphs.py.
	It computes the percentage of test days with negative skill.
	skill_test_days.json to select:
		Architecture of the neural network.
		Folders and files to load.



-RANDOM FOREST:

1 - RFR_models.py
	This file sets the parameters of the random forest regressor and trains the data already prepared.
	It saves the models and trained days.
	It computes different metrics for the global set (not individual days).
	RFR_**.json (i.e. RFR_paper_best.json) to select:
		Depth of the forest and its seed.
		Stations to select as features.
		Train size.
		Seed.
		Time granularity.
		Forecasting horizon.
		Folders and files to load.
2 - RFR_prediction_graphs.py
	This file loads the saved models and applies it to the test subset, as well as it plots any number of days.
	It computes the "skill" as well as other metrics for every single day. It saves it as a .csv file.
	RFR_prediction_graphs.json to select:
		Depth of forest.
		Number of days to test.
		Train size.
		Stations to select as features.
		Forecasting horizon.
		Folders and files to load.
3 - RFR_feature_importances.py
	This file loads the models saved and sort the features by importance for the RFR.
	It saves it in a single .csv for every model in an '/importances' folder.
	RFR_feature_importances_**.json (i.e. RFR_feature_importances_best.json) to select:
		Folders and files to load.

-LINEAR REGRESSOR:

1 - LinReg.py
	This file performs a simple linear regression over the data already prepared.
	It saves the models and trained days.
	LinReg_**.json (i.e. LinReg_paper_best.json) to select:
		Stations to select as features.
		Train size.
		Seed.
		Time granularity.
		Forecasting horizon.
		Folders and files to load.
2 - LinReg_feature_importances.py
	This file loads the trained models and sorts the coefficient for every feature using its absolute value.
	It saves it in a single .csv for every model in an '/importances' folder.
	LR_feature_importances_**.json (i.e. LR_feature_importances_best.json) to select:
		Folders and files to load.

---------------------------------------------------------------
After that all there is to do is to get the top/bottom features and train again.

1 - common_features.py
	This file get the top/bottom n features for RFR and LR and saves them in a file for every forecasting horizon:
	common_features.json to select:
		Stations to select as features.
		Number of features to save.
		Folders and files to load.
2 - best_features_models.py
	This file performs a NN training for the top/bottom features previously selected.
	It saves the models and trained days.
	It computes different metrics for the global set (not individual days).
	NN_best_worst_**.json (i.e. NN_best_worst_300_300_300.json) to select:
		Hidden layers and its neurons.
		Number of features to select.
		Regularization factor (Alpha).
		Train size.
		Seed.
		Time granularity.
		Forecasting horizon.
		Folders and files to load.
3 - skill_plot_best_worst.py
	This file plots the skill vs forecasting horizon for best and worst features.
	skill_plot_best_worst.json to select:
		Number of features.
		Architecture of the neural network.
		Folders and files to load.
