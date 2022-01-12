""" Auxiliary code for section 4 of the main notebook """

# All notebook dependencies:
import cvxpy as cp
import numpy as np
import pandas as pd
import numpy.linalg as LA
import statsmodels.api as sm
import plotly.graph_objs as go
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds

from auxiliary.section3_SCM import SCM

def basque_weights():
    w_nested_basque = output_object_basque[1]

    weights_compare_basque = pd.DataFrame({'Region':x0_Basque.columns.unique(),
                                        'W_Basque_Nested': np.round(w_nested_basque.ravel(), decimals=4),
                                        'W_Basque_Abadie':[0, 0, 0, 0, 0, 0, 0, 0, 0.8508, 0, 0, 0, 0.1492,0,0,0]})


   
    display(weights_compare_basque.T)


def reunification_weights():
    w_reunification = output_object_reunification[1]
    control_units_reuni = df.drop(df[df.country == "West Germany"].index)

    weights_compare_reunification = pd.DataFrame({'Country':control_units_reuni.country.unique(), 
                                        'W_Reunification_Nested': np.round(w_reunification.ravel(), decimals=3),
                                        'W_Reunification_Abadie':[0.22, 0, 0.42, 0, 0, 0, 0, 0.09, 0, 0.11, 0.16, 0, 0, 0, 0,0]})

    def RMSPE(w,Z0,Z1):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))
    
    display(weights_compare_reunification.T)

    print('\nRMSPE Nested:    {} \nRMSPE Abadie:    {}'\
              .format(np.round(RMSPE(w = w_reunification,Z0 = Z0_reuni, Z1 = Z1_reuni),5), 
              np.round(RMSPE(w = w_abadie, Z0 = Z0_reuni,Z1 = Z1_reuni),5)))


def numerical_instability(Z0,Z1,X0,X1,data,v_pinotti,v_becker,w_becker,w_pinotti):
    
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2)) 
    
    Z0_float = Z0.astype('float64')
    Z1_float = Z1.astype('float64')

    nvarsV = X0.shape[0]
    big_dataframe = pd.concat([pd.DataFrame(X0), pd.DataFrame(X1)], axis=1)
    divisor = np.sqrt(big_dataframe.apply(np.var, axis=1))
    V = np.zeros(shape=(8, 8))
    np.fill_diagonal(V, np.diag(np.repeat(big_dataframe.shape[0],1)))
    scaled_matrix = np.array(((big_dataframe.T) @ (np.array(1/(divisor)).reshape(8,1) * V)).T)

    df_matrix = pd.DataFrame(scaled_matrix)
    df_matrix.columns =['PIE', 'VDA', 'LOM', 'TAA', 'VEN', 'FVG', 'LIG', 'EMR', 'TOS',
           'UMB', 'MAR', 'LAZ', 'ABR', 'MOL', 'SAR','NEW']

    df_matrix.index = ['GDP per Capita','Investment Rate','Industry VA','Agriculture VA',
                                          'Market Services VA','Non-market Services VA','Human Capital',
                                          'Population Density']

    unit_identifier     = 'reg'
    time_identifier     = 'year'
    matching_period     = list(range(1951, 1961))
    treat_unit          = 21
    control_units       = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
    outcome_variable    = ['gdppercap']
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    entire_period       = list(range(1951, 2008))
    
    n = 10
    weights_WV = []

    def random_ordering(x):

        np.random.seed(x)
        df_shuffle = np.take(df_matrix,np.random.permutation(df_matrix.shape[0]),axis=0)
        v_order    = list(df_shuffle.index)
        X0_scaled  = np.array(df_shuffle.iloc[:,0:15])
        X1_scaled  = np.array(df_shuffle.iloc[:,15:16])

        output_object = SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,
                            control_units,outcome_variable,predictor_variables,dataprep=False,
                            x0=X0_scaled,x1=X1_scaled,z0=Z0_float,z1=Z1_float)

        output_vec  = [output_object[1],output_object[2],v_order]
        return output_vec

    weights_WV = Parallel(n_jobs=-1)(delayed(random_ordering)(x) for x in list(range(1,n+1)))
    
    weights_WV_frame = pd.DataFrame(weights_WV)
    v_weights = pd.DataFrame(weights_WV[0][1], index = weights_WV[0][2], columns=[0])
    control_units = data[(data.reg <= 14) | (data.reg ==20)]
    w_weights = pd.DataFrame(weights_WV[0][0], index = list(control_units.region.unique()), columns=[0])

    for i in range(1,n):
        v2 = pd.DataFrame(weights_WV[i][1], index = weights_WV[i][2], columns=[i])
        v_weights = pd.merge(v_weights, v2, left_index=True, right_index=True)

        w2 = pd.DataFrame(weights_WV[i][0], index = list(control_units.region.unique()), columns=[i])
        w_weights = pd.merge(w_weights, w2, left_index=True, right_index=True)
    
    v_weights = v_weights.reindex(['GDP per Capita','Investment Rate','Industry VA','Agriculture VA',
                                      'Market Services VA','Non-market Services VA','Human Capital',
                                      'Population Density'])

    RMSPE_orders = pd.DataFrame(w_weights.apply(lambda x: np.sqrt(np.mean((Z1 - Z0 @ np.array(x).reshape(15,1))**2)),axis=0))

    reorderingV_result  = pd.DataFrame({'Pinotti (Synth)': np.round(v_pinotti,3),
                                        'Becker (MSCMT)': np.round(v_becker,3),
                                        'SCM/Minimum': np.round(v_weights.min(axis=1).values,3),
                                        'SCM/Mean': np.round(v_weights.mean(axis=1).values,3),
                                        'SCM/Maximum':np.round(v_weights.max(axis=1).values,3)},
                                          index= v_weights.index)

    reorderingW_result  = pd.DataFrame({'Pinotti (Synth)': np.round(w_pinotti[12:15].ravel(),3),
                                        'Becker (MSCMT)': np.round(w_becker[12:15].ravel(),3),
                                        'SCM/Minimum': np.round(w_weights.min(axis=1).values[12:15],3),
                                        'SCM/Mean': np.round(w_weights.mean(axis=1).values[12:15],3),
                                        'SCM/Maximum':np.round(w_weights.max(axis=1).values[12:15],3)},
                                          index= ['ABR','MOL','SAR'])

    reorderingRMSPE_result  = pd.DataFrame({'Pinotti (Synth)': RMSPE(w_pinotti),
                                            'Becker (MSCMT)': RMSPE(w_becker),
                                            'SCM/Minimum': np.round(RMSPE_orders.min().values,3),
                                            'SCM/Mean': np.round(RMSPE_orders.mean().values,3),
                                            'SCM/Maximum':np.round(RMSPE_orders.max().values,3)},
                                          index= ['RMSPE'])
    
    reordered_table = pd.concat([reorderingV_result,reorderingW_result,reorderingRMSPE_result], axis=0)

    display(reordered_table.round(3))
