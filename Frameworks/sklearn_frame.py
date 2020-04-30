"""
General framework to tune hyperparameters on available sklearn based model from Model module.
"""

## MODULES IMPORTATION SECTION ##---------------------------------------------//

# Importation of helpful generic modules
import sys
import os
from numpy import linspace

# Importation of available model based on existing sklearn models
from GetTuned import MLP, SVM

# Importation of the HPtuner modules needed for the HPO process
from GetTuned import HPtuner

# Importation of domain object needed to define each hyperparameter's search space
from GetTuned import DiscreteDomain, ContinuousDomain

# Importation of helpful available functions from DataManager module
# to generate data to practice hyperparameter tuning with GetTuned
from GetTuned import DataGenerator, load_digits_dataset, load_breast_cancer_dataset, \
                     load_forest_covertypes_dataset, load_forest_covertypes_dataset, load_iris_dataset, plot_data



## RESULT SAVING SETTINGS ##--------------------------------------------------//

save_enabled = True
display_results = False
saving_path  =  os.path.join(os.path.dirname(os.getcwd()), 'Results')
experiment_title = '' # (**Optional** -> Will be used as a directory within Results/)
name_of_dataset = ''  # (**Optional** -> Will only be written in summary.txt)  



## BUDGET SETTINGS ##---------------------------------------------------------//

# Total number of epochs that we're allowed to execute during the HPO process
total_epochs_budget = 100000

# Total number of epochs that we're allowed to execute
# while testing a single configuration of hyperparameters
max_budget_per_config = 500

# Number of cross validation that we want to execute to compute the
# score of a single configuration of hyperparameters
nb_cross_validation = 2

# Percentage of training data to take for validation 
# when computing the score of a single configuration of hyperparameters
valid_size = 0.2



## TUNING METHOD CHOICE ##----------------------------------------------------//

"""
 List of the current available methods :

['grid_search', 'random_search', 'gaussian_process', 'tpe', 'annealing', 'hyperband', 'BOHB']

"""

method = 'annealing'


## **OPTIONAL PARAMETER FOR GAUSSIAN PROCESS METHOD** ##----------------------//

variant = 'GP'                      # One among these : ['GP', 'GP_MCMC']
acquisition_function = 'EI'         # One among these : ['EI', 'MPI']
nb_inital_evaluation_points = 5     # Number of points to evaluate before the beginning of the optimization


## LETS GET TUNED! ## --------------------------------------------------------//

# We generate data for our tests and global variables for all tests
x_train, t_train, x_test, t_test = load_digits_dataset()

# We initialize our model (See Model.py for more informations on hyperparameter options)
model = SVM()

# We define our search space dictionary 
search_space = {'C': ContinuousDomain(-8, 0, log_scaled=True),      # From 10^-8 to 10^0
                'gamma': ContinuousDomain(-8, 0, log_scaled=True)}

# We initialize our HPtuner and set the hyperparameters search space
hp_tuner = HPtuner(model=model, method=method, total_budget=total_epochs_budget,
                   max_budget_per_config=max_budget_per_config, test_default_hyperparam=False)

hp_tuner.set_search_space(search_space)

# We execute the hyperparameter optimization (the tuning!)
tuning_analyst = hp_tuner.tune(X=x_train, t=t_train, nb_cross_validation=nb_cross_validation, valid_size=valid_size,
                               nbr_initial_evals=nb_inital_evaluation_points, method_type=variant, acquisition_function=acquisition_function)

final_accuracy_score = model.score(X=x_test, t=t_test)

if save_enabled:
    tuning_analyst.save_all_results(path=saving_path, experiment_title=experiment_title, dataset_name=name_of_dataset,
                                    training_size=len(x_train), test_accuracy=final_accuracy_score)

print("\n\n*******************************")
print("\nTuning completed!!", "\n\n")

if display_results:
    
    print("Best hyper-parameters found : %s \n" % str(tuning_analyst.best_hyperparameters))
    print("Test accuracy : %g \n\n" % final_accuracy_score)
    print("See tuning_summary.txt for more details \n")
    tuning_analyst.plot_accuracy_history(best_accuracy=False)
    tuning_analyst.plot_accuracy_history(best_accuracy=True)

print("*******************************\n\n")






















