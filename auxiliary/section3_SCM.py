""" Auxiliary code for section 3 of the main notebook.

    Contents include functions for:
        - data preparation
        - dynamic graphs
        - optimization with CVXPY and scipy 
        - dataframes for RMSPE and outputs """

# All notebook dependencies:
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




def data_prep_1(data):
    
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
    
    """Initial CVXPY setup: defines function to call and output a vector of weights function """
    
    def w_optimize(v_diag,solver=cp.ECOS):

        V = np.zeros(shape=(8, 8))
        np.fill_diagonal(V,v_diag)

        W = cp.Variable((15, 1), nonneg=True) ## Creates a 15x1 positive nonnegative variable
        objective_function    = cp.Minimize(cp.norm(V @ (X1 - X0 @ W)))
        objective_constraints = [cp.sum(W) == 1]
        objective_problem     = cp.Problem(objective_function, objective_constraints)
        objective_solution    = objective_problem.solve(solver)

        return (W.value,objective_problem.constraints[0].violation(),objective_solution)

    # CVXPY Solution
    v_diag = [1/8]*8
    w_basic, constraint_violation, objective_solution = w_optimize(v_diag)

    print('\nObjective Value: ', objective_solution)

    solution_frame_1 = pd.DataFrame({'Region':control_units.region.unique(), 
                               'Weights': np.round(w_basic.T[0], decimals=3)})

    display(solution_frame_1.transpose())
    return w_basic

    
    

    
def data_compare_df(w_basic, X0, X1):
    """Outputs a dataframe to show predicted versus actual values of variables"""
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    w_becker = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4303541, 0.4893414, 0.0803045]).reshape(15,1)
    v_pinotti = [0.464413137,0.006141563,0.464413137,0.006141563,0.013106925,0.006141563,0.033500548,0.006141563]
    v_becker  = [0,0.000000005,0,0.499999948,0.000000088,0.499999948,0.000000005,0.000000005]

    ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    x_pred_pinotti = (X0 @ w_pinotti)
    x_pred_basic = (X0 @ w_basic)

    pred_error_pinotti = x_pred_pinotti - X1
    pred_error_basic = x_pred_basic - X1

    data_compare = pd.DataFrame({'Observed':X1.T[0],
                                 'Pinotti Predicted':x_pred_pinotti.T[0],
                                 'Optimizer Predicted':x_pred_basic.T[0],
                                 'Pinotti Differential': pred_error_pinotti.T[0],
                                 'Optimizer Differential': pred_error_basic.T[0]},
                                  index= ['GDP per Capita','Investment Rate','Industry VA','Agriculture VA',
                                          'Market Services VA','Non-market Services VA','Human Capital',
                                          'Population Density'])

    #print ('\nBreakdown across predictors:')

    display(data_compare)
    return w_pinotti, w_becker
    
    

    

def dynamic_graph_1(w_basic, w_pinotti, w_becker, y_control_all, y_treat_all, data):
    
    """ Plots Figure 3.1: Synthetic Control Optimizer vs. Treated unit 
        for CVXPY initial optimizer, Pinotti, Becker and Klößner against the treated unit outcomes """
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

    fig.update_layout(title='Fig. 3.1: Treated Unit vs Synthetic Controls with different region weights',
                       xaxis_title='Time', yaxis_title='GDP per Capita')

    # Dynamic graph
    fig.show()
    
    

    

def RMSPE_compare_df(Z1, Z0, w_basic, w_pinotti, w_becker):
    
    """ Defines function for Root Mean Squared Prediction Error (RMSPE)
        and generates dataframe for RMSPE values comparison between CVXPY output, Pinotti, Becker """

    # Function to obtain RMSPE
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))
    
    # Dataframe to compare RMSPE values
    RMSPE_values = [RMSPE(w_basic), RMSPE(w_pinotti), RMSPE(w_becker)]
    method = ['RMSPE CVXPY','RMSPE Pinotti','RMSPE Becker']
    RMSPE_compare = pd.DataFrame({'Outcome':RMSPE_values}, index=method)
    
    display(RMSPE_compare)


    
    

    

##############################
##   CVXPY implementation   ##   leave it the main notebook?
##############################

def CVXPY_iterative(X0, X1,):
    
    """ CVXPY iterative implementation 
        Approach 1: Iterating over a subset of V"""
    
    initial_w = [0]*15
    bnds = ((0, 1),)*15
    objective_constraint = ({'type': 'eq', 'fun': lambda x: 1.0 -  np.sum(x)})   ## constraint
    iteration_2 = []

    # Function to run in parallel 
        ## use Parallel() to save time
        ## n_jobs=-1 -> all CPU used
        ## delayed(f)(x) for x in list(range(1,n+1))  -> computes function f in parallel, for var x from 1 to n+1

    def objective_function(W,X1,X0,v_diag):    ## function we want to min
        V = np.zeros(shape=(8, 8))
        np.fill_diagonal(V,v_diag)
        obj_value = LA.norm(V @ (X1.ravel() - X0 @ W))
        return obj_value

    def iterate_function(x):

        np.random.seed(x)                   ## deterministic random number generation by setting seed
        v_diag  = np.random.dirichlet(np.ones(8), size=1)

        w_ECOS,  csv_ECOS   = w_optimize(v_diag,solver=cp.ECOS)[0:2]
        w_SCS,   csv_SCS    = w_optimize(v_diag,solver=cp.SCS)[0:2]
        w_CPLEX, csv_CPLEX  = w_optimize(v_diag,solver=cp.CPLEX)[0:2]

        w_SLSQP = optimize.minimize(objective_function,initial_w,args=(X1,X0,v_diag),method='SLSQP',bounds=bnds,
                            constraints=objective_constraint,tol=1e-10,options={'disp': False}).x.reshape(15,1)

            ## optimize.minimize(objective, initial guess, arguments, 'SLSPQ'=sequential least squares programming,
            ##                  bounds, constraints, tolerance, )

        RMSPE_ECOS    = RMSPE(w_ECOS) 
        RMSPE_SCS     = RMSPE(w_SCS) 
        RMSPE_SLSQP   = RMSPE(w_SLSQP)
        RMSPE_CPLEX   = RMSPE(w_CPLEX)

        output_vec  = [v_diag, RMSPE_ECOS,RMSPE_SCS,RMSPE_CPLEX,RMSPE_SLSQP,w_ECOS,w_SCS,w_CPLEX,w_SLSQP,
                      csv_ECOS,csv_SCS,csv_CPLEX]

        return output_vec

    iteration_2 = Parallel(n_jobs=-1)(delayed(iterate_function)(x) for x in list(range(1,n+1)))
    solution_frame_2 = pd.DataFrame(iteration_2)
    solution_frame_2.columns =['Predictor Importance', 'RMSPE ECOS','RMSPE SCS','RMSPE CPLEX','RMSPE SLSQP',
                               'ECOS Weights', 'SCS Weights', 'CPLEX Weights', 'SLSQP Weights',
                               'ECOS Violation', 'SCS Violation', 'CPLEX Violation']    
    """ side by side dataframes display attempt 
    Use HTML+CSS ??? https://python.engineering/38783027-jupyter-notebook-display-two-pandas-tables-side-by-side/
    """
    #return display(best_weights_importance + best_weights_region) #FAIL: outputs NaN values

    
    

    

##############################
##   scipy implementation   ##
##############################

def scipy_weights(n, X0, X1, Z0, Z1, control_units, dtafile):
    
    """ scipy implementation """
    
    data = pd.read_stata(dtafile)
    
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))
    
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
    
    return (w_scipy, v_scipy)

    
    

    

def dynamic_graph_2(w_scipy, w_cvxpy, y_control_all, y_treat_all, dtafile):
    
    """ Dynamic plot for Figure 3.2: Synthetic Control Optimizer vs. Treated unit
        Plots iterative CVXPY, scipy, Pinotti and Becker versus treated unit outcome """
    
    data = pd.read_stata(dtafile)
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    y_synth_scipy = w_scipy.T @ y_control_all
    y_synth_cvxpy = w_cvxpy.T @ y_control_all
    y_synth_pinotti = w_pinotti.T @ y_control_all
    
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

   
    

    
    
def RMSPE_compare2(w_cvxpy, w_scipy, Z1, Z0):
    
    """ Defines function for Root Mean Squared Prediction Error (RMSPE)
        and generates dataframe for RMSPE values comparison between iterative CVXPY output, scipy and Pinotti """
    
    # Function to obtain Root Mean Squared Prediction Error (RMSPE)
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))
    
    # Dataframe to compare RMSPE values
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    RMSPE_values2 = [RMSPE(w_cvxpy), RMSPE(w_scipy), RMSPE(w_pinotti)]
    method2 = ['RMSPE CVXPY','RMSPE scipy','RMSPE Pinotti']
    RMSPE_compare2 = pd.DataFrame({'RMSE':RMSPE_values2}, index=method2)
    display(RMSPE_compare2)
    
    #print('\nRMSPE CVXPY: {} \nRMSPE scipy: {} \nRMSPE Pinotti: {}'\
    #      .format(RMSPE(w_cvxpy),RMSPE(w_scipy),RMSPE(w_pinotti)))

    
    

    
    


    
    

    


#######################################################
##   CVXPY CODE, SETTINGS AND OUTPUT VISUALIZATION   ##
#######################################################

""" Approach 2: Nested Optimization: Combination of outer optimization ( 𝑊∗(𝑉) ) via Differential Evolution 
    and inner optimization  (𝑊)  via CVXPY convex minimization """



def data_prep_2(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
              predictor_variables):
    
    """ Generates the necessary matrices and vectors to proceed with nested optimization """
    
    dtafile = './dataset/Pinotti-replication/dataset.dta'
    data = pd.read_stata(dtafile)
    
    X = data.loc[data[time_identifier].isin(matching_period)]
    X.index = X.loc[:,unit_identifier]
    
    X0 = X.loc[(X.index.isin(control_units)),(predictor_variables)] 
    X0 = X0.groupby(X0.index).mean().values.T                         #control predictors
    
    X1 = X.loc[(X.index == treat_unit),(predictor_variables)]
    X1 = X1.groupby(X1.index).mean().values.T                         #treated predictors

    Z0 = np.array(X.loc[(X.index.isin(control_units)),(outcome_variable)]).reshape(len(control_units),len(matching_period)).T  #control outcome
    Z1 = np.array(X.loc[(X.index == treat_unit),(outcome_variable)]).reshape(len(matching_period),1)                           #treated outcome
    
    return X0, X1, Z0, Z1






def SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
              predictor_variables,reps = 1):
    
    X0, X1, Z0, Z1 = data_prep_2(data,unit_identifier,time_identifier,matching_period,treat_unit,
                               control_units,outcome_variable,predictor_variables)
    
    #inner optimization
    def w_optimize(v):

        W = cp.Variable((len(control_units), 1), nonneg=True)
        objective_function    = cp.Minimize(cp.norm(cp.multiply(v, X1 - X0 @ W)))
        objective_constraints = [cp.sum(W) == 1]
        objective_solution    = cp.Problem(objective_function, objective_constraints).solve(verbose=False)
        return (W.value)
    
    #outer optimization
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





def settings():
    
    """ sets necessary parameters for SCM function, solution_output_SCM function and graphs in section 4 """
    
    unit_identifier     = 'reg'
    time_identifier     = 'year'
    matching_period     = list(range(1951, 1961))
    treat_unit          = 21
    control_units       = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
    outcome_variable    = ['gdppercap']
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    reps                = 1
    entire_period       = list(range(1951, 2008))
    return unit_identifier, time_identifier, matching_period, treat_unit, control_units, outcome_variable, predictor_variables, reps, entire_period







def solution_output_SCM(data, output_object, Z0, Z1):
    
    """ Builds dataframes to display the solution """
    
    solution_frame_4 = output_object[0]
    w_nested = output_object[1]
    v_nested = output_object[2]
    control_units = data[(data.reg <= 14) | (data.reg == 20)]
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)


    best_weights_region3 = pd.DataFrame({'Region':control_units.region.unique(), 
                                        'W(V*)': np.round(w_nested.ravel(), decimals=3)})

    best_weights_importance3 = pd.DataFrame({'Predictors': data.columns[[3,16,11,12,13,14,26,28]],
                                            'V*': np.round(v_nested, 3)})

    display(best_weights_importance3)
    display(best_weights_region3)

    print('\nOptimizer Weights: {} \nPaper Weights:  {}'\
          .format(np.round(w_nested.T,3), np.round(w_pinotti,3).T))

    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))

    print('\nRMSPE Nested:    {} \nRMSPE Pinotti:   {}'\
          .format(np.round(RMSPE(w_nested),5), np.round(RMSPE(w_pinotti),5)))


    
    
############################################################################################################################################


def global_optimum(Z1, Z0, X1, X0):
    
    """ Checks feasibility of unconstrained solution: unrestricted outer optimum in SECTION 3.4 """
    
    w_becker = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4303541, 0.4893414, 0.0803045]).reshape(15,1)
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    
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
    
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))

    print('\nRMSPE Global:   {} \nRMSPE Becker:    {}'\
          .format(np.round(RMSPE(w_global),6), np.round(RMSPE(w_becker),6)))
    return (w_global, v_global)
    
    

    


def dynamic_graph_3(y_control_all, y_treat_all, output_object, dtafile):
    
    """ Dynamic plot of Figure 3.3: Synthetic Control Optimizer vs. Treated unit 
        Plots nested CVXPY optimizer, Pinotti, Becker and Klößner versus treated unit outcome 
        SECTION 3.5 """
    
    data = pd.read_stata(dtafile)
    
    w_becker = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4303541, 0.4893414, 0.0803045]).reshape(15,1)
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    w_nested = output_object[1]
    
    
    y_synth_becker = w_becker.T @ y_control_all
    y_synth_pinotti = w_pinotti.T @ y_control_all
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

    
    

    


def matching_characteristics_table(X0, X1, w_global, predictor_variables, v_global, dtafile):
    
    """ Dataframe with matching period characteristics for Apulia and Basilicata, Synthetic Control, Control Units """
    
    data = pd.read_stata(dtafile)
    v_pinotti = [0.006141563, 0.464413137, 0.006141563, 0.013106925, 0.006141563, 0.033500548, 0.006141563, 0.464413137]

    X = data.loc[data['year'].isin(list(range(1951, 1961)))]
    X.index = X.loc[:,'reg']
    
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

    
    

    


def diff_figure_4(control_units_all, treat_unit_all, y_control_all, y_treat_all, output_object, dtafile):
    """ Generates Figure 3.4: Actual vs Synthetic Differences over time: GDP per capita and Murders.
        Shows differences in evolution of murder rates and GDP per capita between the actual realizations of Apulia 
        and Basilicata and the ones predicted by the synthetic control unit """
    
    data = pd.read_stata(dtafile)
    murd_treat_all      = np.array(treat_unit_all.murd).reshape(1, 57)
    murd_control_all    = np.array(control_units_all.murd).reshape(15, 57)
    w_nested = output_object[1]
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