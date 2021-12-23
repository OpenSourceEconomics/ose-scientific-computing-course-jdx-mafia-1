""" Auxiliary code for section 3 of the main notebook.

    Contents include functions for:
        - data preparation
        - dynamic graphs
        - optimization with CVXPY and scipy 
        - dataframes for RMSPE and outputs """


import numpy as np
import pandas as pd
import cvxpy as cp
import numpy.linalg as LA
import statsmodels.api as sm
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds


def data_prep(data):
    """ Specify conditions for treated unit and control units as per Pinotti's paper (c.f. F216), 
        where 21 are regions "NEW" with recent mafia presence: Apulia and Basilicata """
    
    dtafile = './dataset/Pinotti-replication/dataset.dta'
    data = pd.read_stata(dtafile)
    
    treat_unit     = data[data.reg == 21]
    treat_unit     = treat_unit[treat_unit.year <= 1960]                 # Matching period: 1951 to 1960
    treat_unit_all = data[data.reg == 21]                                # Entire period:   1951 to 2007

    control_units     = data[(data.reg <= 14) | (data.reg ==20)]
    control_units     = control_units[control_units.year <= 1960]
    control_units_all = data[(data.reg <= 14) | (data.reg ==20)]

    
    y_treat     = np.array(treat_unit.gdppercap).reshape(1, 10)              # Matching period: 1951 to 1960
    y_treat_all = np.array(treat_unit_all.gdppercap).reshape(1, 57)          # Entire period:   1951 to 2007

    y_control     = np.array(control_units.gdppercap).reshape(15, 10)
    y_control_all = np.array(control_units_all.gdppercap).reshape(15, 57)

    Z1 = y_treat.T
    Z0 = y_control.T

    # Prepare matrices with only the relevant variables into CVXPY format, predictors k = 8
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    X = data.loc[data['year'].isin(list(range(1951, 1961)))]
    X.index = X.loc[:,'reg']

    # k x J matrix: mean values of k predictors for J untreated units
    X0 = X.loc[(X.index <= 14) | (X.index ==20),(predictor_variables)] 
    X0 = X0.groupby(X0.index).mean().values.T

    # k x 1 vector: mean values of k predictors for 1 treated unit
    X1 = X.loc[(X.index == 21),(predictor_variables)]
    X1 = X1.groupby(X1.index).mean().values.T
    
    return (treat_unit, treat_unit_all, control_units, control_units_all, y_treat, y_treat_all, y_control, y_control_all, Z1, Z0, X0, X1)

    
    

    

def cvxpy_basic_solution(control_units, X0, X1):
    """Initial simple CVXPY setup: Defines function to call and output a vector of weights function """
    
    #data_prep()
    
    def w_optimize(v=None):
        V = np.zeros(shape=(8, 8))
        if v is None:
            np.fill_diagonal(V, [1/8]*8)
        else:
            np.fill_diagonal(V, v)
            
        #X0,X1 = data_prep()
        W = cp.Variable((15, 1), nonneg=True) ## Creates a 15x1 positive nonnegative variable
        objective_function    = cp.Minimize(cp.sum(V @ cp.square(X1 - X0 @ W)))
        objective_constraints = [cp.sum(W) == 1]
        objective_solution    = cp.Problem(objective_function, objective_constraints).solve(verbose=False)
    
        return (W.value,objective_solution)

    # CVXPY Solution
    w_basic, objective_solution = w_optimize()
    print('\nObjective Value: ', objective_solution)
    #print('\nObjective Value: ', objective_solution, '\n\nOptimal Weights: ', w_basic.T)
    solution_frame_1 = pd.DataFrame({'Region':control_units.region.unique(), 
                           'Weights': np.round(w_basic.T[0], decimals=3)})

    return display(solution_frame_1)

    
    

    

def dynamic_graph_1(y_control_all, y_treat_all, data):
    
    """ Plots Figure 3.1: Synthetic Control Optimizer vs. Treated unit 
        for CVXPY initial optimizer, Pinotti, Becker and Klößner against the treated unit outcomes """
    
    dtafile = './dataset/Pinotti-replication/dataset.dta'
    data = pd.read_stata(dtafile)
    
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    w_becker = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4303541, 0.4893414, 0.0803045]).reshape(15,1)
    w_basic = np.array([0., 0., 0., 0., 0.15165999, 0., 0., 0., 0., 0., 0., 0., 0., 0.84834001, 0.]).reshape(15,1)

    y_synth_pinotti = w_pinotti.T @ y_control_all
    y_synth_becker = w_becker.T @ y_control_all
    y_synth_basic = w_basic.T @ y_control_all

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_basic[0],
                    mode='lines', name='Optimizer'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_pinotti[0],
                    mode='lines', name='Pinotti'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_becker[0],
                    mode='lines', name='Becker and Klößner'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_treat_all[0],
                    mode='lines', name='Treated unit'))

    fig.add_shape(dict(type="line", x0=1960, y0=0, x1=1960, y1=11000,
                   line=dict(color="Black", width=1)))

    fig.add_shape(dict(type="line", x0=1974, y0=0, x1=1974, y1=11000,
                   line=dict(color="Black", width=1)))

    fig.add_shape(dict(type="line", x0=1980, y0=0, x1=1980, y1=11000,
                   line=dict(color="Black", width=1)))

    fig.add_trace(go.Scatter(x=[1960], y=[12000], mode="text",
         name="Matching", text=["End of Matching<br>Period"]))
  
    fig.add_trace(go.Scatter(x=[1974], y=[12000], mode="text",
         name="Event 1", text=["Drug<br>Smuggling"]))

    fig.add_trace(go.Scatter(x=[1981], y=[12000], mode="text",
         name="Event 2", text=["Basilicata<br>Earthquake"]))

    fig.update_layout(title='Figure 3.1: Synthetic Control Optimizer vs. Treated unit',
                   xaxis_title='Time', yaxis_title='GDP per Capita')

    return fig.show()

    
    

    

def RMSPE_compare_1(Z1, Z0):
    """ Defines function for Root Mean Squared Prediction Error (RMSPE)
        and generates dataframe for RMSPE values comparison between CVXPY output, Pinotti, Becker """
    
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    w_becker = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4303541, 0.4893414, 0.0803045]).reshape(15,1)
    w_basic = np.array([0., 0., 0., 0., 0.15165999, 0., 0., 0., 0., 0., 0., 0., 0., 0.84834001, 0.]).reshape(15,1)
    
    # Function to obtain RMSPE
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))
    
    # Dataframe to compare RMSPE values
    RMSPE_values = [RMSPE(w_basic), RMSPE(w_pinotti), RMSPE(w_becker)]
    method = ['RMSPE CVXPY','RMSPE Pinotti','RMSPE Becker']
    RMSPE_compare = pd.DataFrame({'Outcome':RMSPE_values}, index=method)
    return display(RMSPE_compare)

    
    

    

def table_predicted_actual(X1, X0):
    """ Dataframe to show predicted vs. actual values of variables """
    dtafile = './dataset/Pinotti-replication/dataset.dta'
    data = pd.read_stata(dtafile)
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    w_basic = np.array([0., 0., 0., 0., 0.15165999, 0., 0., 0., 0., 0., 0., 0., 0., 0.84834001, 0.]).reshape(15,1)
    
    x_pred_pinotti = (X0 @ w_pinotti)
    x_pred_basic = (X0 @ w_basic)

    pred_error_pinotti = x_pred_pinotti - X1
    pred_error_basic = x_pred_basic - X1

    data_compare = pd.DataFrame({'Observed':X1.T[0],
                             'Pinotti Predicted':x_pred_pinotti.T[0],
                             'Optimizer Predicted':x_pred_basic.T[0],
                             'Pinotti Differential': pred_error_pinotti.T[0],
                             'Optimizer Differential': pred_error_basic.T[0]},
                              index= data.columns[[3,16,11,12,13,14,26,28]])

    return display(data_compare)

    
    

    

##############################
##   CVXPY implementation   ##
##############################

def CVXPY_iterative():
    """ CVXPY iterative implementation 
        Approach 1: Iterating over the solution set by generating  𝑉  from a Dirichlet distribution"""

    n = 100000
    iteration_2 = []

    def f(x):
        np.random.seed(x)
        v_diag  = np.random.dirichlet(np.ones(8), size=1)
        w_cvxpy = w_optimize(v_diag)[0]
        print(w_cvxpy.shape)
        prediction_error =  RMSPE(w_cvxpy) 
        output_vec = [prediction_error, v_diag, w_cvxpy]
        return output_vec
    
    iteration_2 = Parallel(n_jobs=-1)(delayed(f)(x) for x in list(range(1,n+1)))
    # Function to run in parallel 
        ## use Parallel() to save time
        ## n_jobs=-1 -> all CPU used
        ## delayed(f)(x) for x in list(range(1,n+1))  -> computes function f in parallel, for var x from 1 to n+1
    
    # Organize output into dataframe
    solution_frame_2 = pd.DataFrame(iteration_2)
    solution_frame_2.columns =['Error', 'Relative Importance', 'Weights']

    solution_frame_2 = solution_frame_2.sort_values(by='Error', ascending=True)

    w_cvxpy = solution_frame_2.iloc[0][2]
    v_cvxpy = solution_frame_2.iloc[0][1][0]

    best_weights_region = pd.DataFrame({'Region':control_units.region.unique(),
                                    'W(V*)': np.round(w_cvxpy.ravel(), decimals=3)})

    best_weights_importance = pd.DataFrame({'Predictors': data.columns[[3,16,11,12,13,14,26,28]],
                                        'V*': np.round(v_cvxpy, 3)})

    display(best_weights_region)
    display(best_weights_importance)

    
    

    

##############################
##   scipy implementation   ##
##############################

def scipy_weights():
    """ scipy implementation """
    
    A = X0
    b = X1.ravel()  ## .ravel() returns continuous flattened array [[a,b],[c,d]]->[a,b,c,d]
    iteration_3 = []
    init_w = [0]*15

    bnds = ((0, 1),)*15
    cons = ({'type': 'eq', 'fun': lambda x: 1.0 -  np.sum(x)})   ## constraint

    def fmin(x,A,b,v):         ## function we want to min
        c = np.dot(A, x) - b   ## np.dot(a,b) multiplies a and b => X0*x - X1
        d = c ** 2
        y = np.multiply(v,d)   ## y = v * (X0*x - X1)^2
        return np.sum(y)

    def g(x):
    
        np.random.seed(x)    ## deterministic random number generation by setting seed
        v = np.random.dirichlet(np.ones(8), size=1).T
        args = (A,b,v)
        res = optimize.minimize(fmin,init_w,args,method='SLSQP',bounds=bnds,
                            constraints=cons,tol=1e-10,options={'disp': False})
        ## optimize.minimize(objective, initial guess, arguments, 'SLSPQ'=sequential least squares programming,
        ##                   bounds, constraints, tolerance, )
    
        prediction_error =  RMSPE(res.x) 
        output_vec = [prediction_error, v, res.x]
        return output_vec
    
    iteration_3 = Parallel(n_jobs=-1)(delayed(g)(x) for x in list(range(1,n+1)))

    # Organize output into dataframe
    solution_frame_3 = pd.DataFrame(iteration_3)
    solution_frame_3.columns =['Error', 'Relative Importance', 'Weights']

    solution_frame_3 = solution_frame_3.sort_values(by='Error', ascending=True)

    w_scipy = solution_frame_3.iloc[0][2].reshape(15,1)
    v_scipy = solution_frame_3.iloc[0][1].T[0]

    best_weights_region2 = pd.DataFrame({'Region':control_units.region.unique(), 
                                        'W(V*)': np.round(w_scipy.ravel(), decimals=3)})

    best_weights_importance2 = pd.DataFrame({'Predictors': data.columns[[3,16,11,12,13,14,26,28]],
                                            'V*': np.round(v_scipy, 3)})

    display(best_weights_importance2)
    display(best_weights_region2)

    
    

    

def dynamic_graph_2():
    """ Dynamic plot for Figure 3.2: Synthetic Control Optimizer vs. Treated unit
        Plots iterative CVXPY, scipy, Pinotti and Becker versus treated unit outcome """
    
    y_synth_scipy = w_scipy.T @ y_control_all
    y_synth_cvxpy = w_cvxpy.T @ y_control_all
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_cvxpy[0],
                        mode='lines', name='Optimizer CVXPY'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_scipy[0],
                        mode='lines', name='Optimizer SciPY'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_pinotti[0],
                        mode='lines', name='Pinotti'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_treat_all[0],
                        mode='lines', name='Treated unit'))
    fig.add_shape(dict(type="line", x0=1960, y0=0, x1=1960, y1=11000,
                       line=dict(color="Black", width=1)))
    
    fig.add_trace(go.Scatter(x=[1960], y=[12000], mode="text",
        name="Matching", text=["End of Matching<br>Period"]))

    fig.update_layout(title='Figure 3.2: Synthetic Control Optimizer vs. Treated unit',
                       xaxis_title='Time', yaxis_title='GDP per Capita')
    fig.show()

    
    

    
    
def RMSPE_compare2():
    """ Defines function for Root Mean Squared Prediction Error (RMSPE)
        and generates dataframe for RMSPE values comparison between iterative CVXPY output, scipy and Pinotti """
    
    # Function to obtain Root Mean Squared Prediction Error (RMSPE)
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))
    
    # Dataframe to compare RMSPE values
    RMSPE_values2 = [RMSPE(w_cvxpy), RMSPE(w_scipy), RMSPE(w_pinotti)]
    method2 = ['RMSPE CVXPY','RMSPE scipy','RMSPE Pinotti']
    RMSPE_compare2 = pd.DataFrame({'RMSE':RMSPE_values2}, index=method2)
    display(RMSPE_compare2)
    
    #print('\nRMSPE CVXPY: {} \nRMSPE scipy: {} \nRMSPE Pinotti: {}'\
    #      .format(RMSPE(w_cvxpy),RMSPE(w_scipy),RMSPE(w_pinotti)))

    
    

    
    
def nested_data_prep():
    """ Data preparation to proceed with nested optimization """
    
    def data_prep(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
                  predictor_variables):
        
        X = data.loc[data[time_identifier].isin(matching_period)]
        X.index = X.loc[:,unit_identifier]
        
        X0 = X.loc[(X.index.isin(control_units)),(predictor_variables)] 
        X0 = X0.groupby(X0.index).mean().values.T                         #control predictors
        
        X1 = X.loc[(X.index == treat_unit),(predictor_variables)]
        X1 = X1.groupby(X1.index).mean().values.T                         #treated predictors
    
        Z0 = np.array(X.loc[(X.index.isin(control_units)),(outcome_variable)]).reshape(len(control_units),len(matching_period)).T  #control outcome
        Z1 = np.array(X.loc[(X.index == treat_unit),(outcome_variable)]).reshape(len(matching_period),1)                           #treated outcome
        return X0, X1, Z0, Z1

    
    

    


#######################################################
##   CVXPY CODE, SETTINGS AND OUTPUT VISUALIZATION   ##
#######################################################
    
def CVXPY_nested():
    """ Approach 2: Nested Optimization: Combination of outer optimization ( 𝑊∗(𝑉) ) via Differential Evolution 
        and inner optimization  (𝑊)  via CVXPY convex minimization """
    
    nested_data_prep()
    
    ## CVXPY ##
    def SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
                  predictor_variables,reps = 1):
        X0, X1, Z0, Z1 = data_prep(data,unit_identifier,time_identifier,matching_period,treat_unit,
                                   control_units,outcome_variable,predictor_variables)

        # Inner optimization
        def w_optimize(v):

            W = cp.Variable((len(control_units), 1), nonneg=True)
            objective_function    = cp.Minimize(cp.norm(cp.multiply(v, X1 - X0 @ W)))
            objective_constraints = [cp.sum(W) == 1]
            objective_solution    = cp.Problem(objective_function, objective_constraints).solve(verbose=False)
            return (W.value)

        # Outer optimization
        def vmin(v): 
            v = v.reshape(len(predictor_variables),1)
            W = w_optimize(v)
            return ((Z1 - Z0 @ W).T @ (Z1 - Z0 @ W)).ravel()

        def constr_f(v):
            return float(np.sum(v))

        def constr_hess(x,v):
            v=len(predictor_variables)
            return np.zeros([v,v])

        def constr_jac(v):
            v=len(predictor_variables)
            return np.ones(v)

        def RMSPE_f(w):
            return np.sqrt(np.mean((w.T @ Z0.T - Z1.T)**2))

        def v_optimize(i):
            bounds  = [(0,1)]*len(predictor_variables)
            nlc     = NonlinearConstraint(constr_f, 1, 1, constr_jac, constr_hess)
            result  = differential_evolution(vmin, bounds, constraints=(nlc),seed=i,tol=0.01)
            v_estim = result.x.reshape(len(predictor_variables),1)  
            return (v_estim)

        def h(x):
            v_estim1 = v_optimize(x)
            w_estim1 = w_optimize(v_estim1)
            prediction_error = RMSPE_f(w_estim1)
            output_vec = [prediction_error, v_estim1, w_estim1]
            return output_vec

        iterations = []
        iterations = Parallel(n_jobs=-1)(delayed(h)(x) for x in list(range(1,reps+1)))

        solution_frame = pd.DataFrame(iterations)
        solution_frame.columns =['Error', 'Relative Importance', 'Weights']
        solution_frame = solution_frame.sort_values(by='Error', ascending=True)

        w_nested = solution_frame.iloc[0][2]
        v_nested = solution_frame.iloc[0][1].T[0]

        output = [solution_frame,w_nested,v_nested,RMSPE_f(w_nested)]
        return output
    
    ## SETTINGS
    unit_identifier     = 'reg'
    time_identifier     = 'year'
    matching_period     = list(range(1951, 1961))
    treat_unit          = 21
    control_units       = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
    outcome_variable    = ['gdppercap']
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    reps                = 1
    entire_period       = list(range(1951, 2008))

    output_object = SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,
                        control_units,outcome_variable,predictor_variables,reps)
    
    ## ORGANIZE OUTPUT INTO DATAFRAME
    solution_frame_4 = output_object[0]
    w_nested = output_object[1]
    v_nested = output_object[2]
    control_units = data[(data.reg <= 14) | (data.reg == 20)]

    best_weights_region3 = pd.DataFrame({'Region':control_units.region.unique(), 
                                        'W(V*)': np.round(w_nested.ravel(), decimals=3)})

    best_weights_importance3 = pd.DataFrame({'Predictors': data.columns[[3,16,11,12,13,14,26,28]],
                                            'V*': np.round(v_nested, 3)})

    display(best_weights_importance3)
    display(best_weights_region3)

    print('\nOptimizer Weights: {} \nPaper Weights:  {}'\
          .format(np.round(w_nested.T,3), np.round(w_pinotti,3).T))

    print('\nRMSPE Nested:    {} \nRMSPE Pinotti:   {}'\
          .format(np.round(RMSPE(w_nested),5), np.round(RMSPE(w_pinotti),5)))

    
    

    


def global_optimum():
    """ Checks feasibility of unconstrained solution: unrestricted outer optimum """
    
    W = cp.Variable((15, 1), nonneg=True)
    objective_function    = cp.Minimize(np.mean(cp.norm(Z1 - Z0 @ W)))
    objective_constraints = [cp.sum(W) == 1]
    objective_solution    = cp.Problem(objective_function, objective_constraints).solve(verbose=False)

    V = cp.Variable((8, 1), nonneg=True)
    objective_function    = cp.Minimize(cp.norm(cp.multiply(V, X1 - X0 @ W.value)))
    objective_constraints = [cp.sum(V) == 1]
    objective_solution    = cp.Problem(objective_function, objective_constraints).solve(verbose=False)

    v_global = V.value.ravel()
    w_global = W.value

    print('\nOptimizer Weights: {} \nOptimal Weights:  {}'\
          .format(np.round(w_global.T,5), np.round(w_becker,5).T))

    print('\nRMSPE Global:   {} \nRMSPE Becker:    {}'\
          .format(np.round(RMSPE(w_global),6), np.round(RMSPE(w_becker),6)))

    
    

    


def dynamic_graph_3():
    """ Dynamic plot of Figure 3.3: Synthetic Control Optimizer vs. Treated unit 
        Plots nested CVXPY optimizer, Pinotti, Becker and Klößner versus treated unit outcome """
    
    y_synth_becker = w_becker.T @ y_control_all
    y_synth_nested = w_nested.T @ y_control_all

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_nested[0],
                        mode='lines', name='Nested Optimizer'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_becker[0],
                        mode='lines', name='Becker and Klößner'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_synth_pinotti[0],
                        mode='lines', name='Pinotti'))
    fig.add_trace(go.Scatter(x=list(data.year.unique()), y=y_treat_all[0],
                        mode='lines', name='Treated unit'))
    fig.add_shape(dict(type="line", x0=1960, y0=0, x1=1960, y1=11000,
                       line=dict(color="Black", width=1)))

    fig.add_trace(go.Scatter(x=[1960], y=[12000], mode="text",
        name="Matching", text=["End of Matching<br>Period"]))

    fig.update_layout(title='Figure 3.3: Synthetic Control Optimizer vs. Treated unit',
                       xaxis_title='Time', yaxis_title='GDP per Capita')
    fig.show()

    
    

    


def matching_characteristics_table():
    """ Dataframe with matching period characteristics for Apulia and Basilicata, Synthetic Control, Control Units """
    
    v_pinotti = [0.006141563, 0.464413137, 0.006141563, 0.013106925, 0.006141563, 0.033500548, 0.006141563, 0.464413137]

    x_pred_global = (X0 @ w_global).ravel()
    control_stats = X.loc[(X.index <= 14) | (X.index ==20),
                          (predictor_variables)].describe().drop(['count','25%', '50%','75%'], axis=0).T
    control_stats = np.round(control_stats,2)
    data_compare  = pd.DataFrame({'Global Opt Weights': np.round(v_global,3),
                                  'Pinotti Weights': np.round(v_pinotti,3),
                                  'Apulia and Basilicata':np.round(X1.T[0],3),
                                  'Synthetic Control':x_pred_global},
                               index= data.columns[[3,16,11,12,13,14,26,28]])
    frames = [data_compare, control_stats]
    result = pd.concat(frames,axis=1)
    #print ('\nMatching Period Characteristics: Apulia and Basilicata, Synthetic Control, Control Units')
    display(result)

    
    

    


def diff_figure_4():
    """ Generates Figure 3.4: Actual vs Synthetic Differences over time: GDP per capita and Murders 
        Shows differences in evolution of murder rates and GDP per capita between the actual realizations of Apulia 
        and Basilicata and the ones predicted by the synthetic control unit """
    
    murd_treat_all      = np.array(treat_unit_all.murd).reshape(1, 57)
    murd_control_all    = np.array(control_units_all.murd).reshape(15, 57)
    synth_murd = w_nested.T @ murd_control_all
    synth_gdp = w_nested.T @ y_control_all
    diff_GDP = (((y_treat_all-synth_gdp)/(synth_gdp))*100).ravel()
    diff_murder = (murd_treat_all - synth_murd).ravel()
    diff_data = pd.DataFrame({'Murder Gap':diff_murder,
                             'GDP Gap': diff_GDP},
                             index=data.year.unique())

    year = diff_data.index.values
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDP per capita, % Gap')
    ax1.bar(year,diff_data['GDP Gap'],width = 0.5,label = 'GDP per capita')
    ax1.axhline(0)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Murder Rate, Difference')
    ax2.plot(diff_data['Murder Gap'],color='black',label = 'Murders')
    ax2.axhline(0)
    ax2.tick_params(axis='y')

    plt.axvspan(1975, 1980, color='y', alpha=0.5, lw=0,label='Mafia Outbreak')
    ax1.set_ylim(-20,20)
    ax2.set_ylim(-4.5,4.5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2,loc = 'upper center', bbox_to_anchor = (0.5, -0.15), shadow = True, ncol = 2)
    fig.tight_layout() 
    plt.title('Fig 3.4: Actual vs Synthetic Differences over time: GDP per capita and Murders')
    plt.show()