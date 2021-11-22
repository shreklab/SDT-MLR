# StMLR

Overview:
StMLR ~ Soft Tree - Multinomial Logistic Regression (MLR) hybrid model \
A Soft Tree (Frosst and Hinton) is used to separate a space into a number of subspaces. MLR submodels are fit on each of the different subspaces.

Example Usage: Neural activity forecasting \
Recent neural activity is used to predict future neural activity. Recent neural activity is fed into both the Soft Tree and the MLR submodels. The Soft Tree divides the space of neural activity trajectories into different subspaces. It then tries to model neural activity evolution as a multinomial linear model in each of these subspaces.

Project Structure: \
-models: contains st_mlr engine \
-model_wrappers: contains scripts that perform experiments using st_mlr engine. master_timing performs forecasting. \
-run_scripts: contains scripts that run experiments \

Examples: \
sample_run_script.py in run_scripts. Goes into detail about model specification. 

Run the sample script: \
Example: [run from outside the project directory] \
PYTHONPATH=StMLR python -m StMLR.run_scripts.sample_run_script
