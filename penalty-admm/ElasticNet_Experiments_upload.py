#%%
import sys
sys.path.append('/Users/persevere/Downloads/penalty-admm-upload')

import numpy as np
import pandas as pd

from Data_Generator import Data_Generator_Wrapper, ElasticNet_Data_Generator, ElasticNet_Setting
from ElasticNet_Algorithms_upload import iP_DCA, Bayesian_Method, Grid_Search, Random_Search, IGJO, LDMMA,LDPM

from utils import performance_reporter, results_printer

#%%
def main():
    num_repeat = 1

    num_train = 100
    num_validate = 100
    num_test = 250
    num_features = 450


    unique_marker = "sth"
    marker = unique_marker + "_%d_%d_%d_%d" % (num_train, num_validate, num_test, num_features)

    result_path = "results"

    Methods = ["GS", "RS", "Bayes", "DC", "HC",'MM','PM']
    
    PM_Setting = {
    'gd_step': 0.0005,
    "MAX_ITERATION":100,
    "TOL":0.01,#
    "beta0":10,
    }
    
    MM_Setting = {
        "MAX_ITERATION":100,
        "TOL":0.1,
        "epsilon":1e-2*np.ones(1),
        "TOLEC":0.1,
        "ITEREC":50
        }
    
    
    DC_Setting = {
        "TOL": 1e-2,
        "initial_guess": np.array([10, 5]),
        "MAX_ITERATION": 100,
        "rho": 1e-3,
        "delta": 5,
        "c": 0.5
    }

    HC_Setting = {
        "num_iters": 50,
        "step_size_min": 1e-12,
        "initial_guess": 1e-2*np.ones(2),
        "decr_enough_threshold": 1e-12
    }

    IF_Setting = {
        "n_outer": 50,
        "alpha0": 1e-2*np.ones(2)
    }
     
    ALM_Setting = {
        "alpha":0.01
        }

    settings = ElasticNet_Setting(num_train, num_validate, num_test, num_features)
    list_data_info = [
        Data_Generator_Wrapper(ElasticNet_Data_Generator, settings, i+1)
        for i in range(num_repeat)
    ]
    

    for i in range(num_repeat):
        print("Experiments: " + str(i+1) + "/" + str(num_repeat))
        data_info = list_data_info[i]
        
        if "MM" in Methods:
            Result_MM = LDMMA(data_info,MM_Setting)
            performance_reporter(Result_MM, 'LDMMA', "latest")
            Result_MM.to_pickle(result_path + "/elasticnet/MM_" + str(data_info.data_index) + marker + ".pkl")

        if "PM" in Methods:
            Result_PM = LDPM(data_info,PM_Setting)
            performance_reporter(Result_PM, 'PM', "latest")
            Result_PM.to_pickle(result_path + "/elasticnet/PM_" + str(data_info.data_index) + marker + ".pkl")


        if "GS" in Methods:
            Result_GS = Grid_Search(data_info)
            performance_reporter(Result_GS, 'Grid Search', "best")
            Result_GS.to_pickle(result_path + "/elasticnet/GS_" + str(data_info.data_index) + marker + ".pkl")

        if "RS" in Methods:
            Result_RS = Random_Search(data_info)
            performance_reporter(Result_RS, 'Random Search', "best")
            Result_RS.to_pickle(result_path + "/elasticnet/RS_" + str(data_info.data_index) + marker + ".pkl")

        if "Bayes" in Methods:
            Result_Bayes = Bayesian_Method(data_info)
            performance_reporter(Result_Bayes, 'Bayesian Method', "best")
            Result_Bayes.to_pickle(result_path + "/elasticnet/Bayes_" + str(data_info.data_index) + marker + ".pkl")
        
        if "HC" in Methods:
            Result_HC = IGJO(data_info, HC_Setting)
            performance_reporter(Result_HC, 'HC method', "latest")
            Result_HC.to_pickle(result_path + "/elasticnet/HC_" + str(data_info.data_index) + marker + ".pkl")


        if "DC" in Methods:
            Result_DC = iP_DCA(data_info, DC_Setting)
            performance_reporter(Result_DC, 'VF-iDCA', "latest")
            Result_DC.to_pickle(result_path + "/elasticnet/DC_" + str(data_info.data_index) + marker + ".pkl")



       
    results_printer(num_repeat, "elasticnet", Methods, result_path, suffix=marker, latex=False)

#%%
if __name__ == "__main__":
    main()