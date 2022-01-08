""" Auxiliary code for section 3 of the main notebook.

    Contents include functions for:
        - data preparation
        - dynamic graphs
        - optimization with CVXPY and scipy 
        - dataframes for RMSPE and outputs """

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


def data_prep_1(data):
    
    """ Specify conditions for treated unit and control units as per Pinotti's paper (c.f. F216), 
        where 21 are regions "NEW" with recent mafia presence: Apulia and Basilicata """
    
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
    
    w_pinotti = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6244443, 0.3755557, 0]).reshape(15, 1)
    w_becker = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4303541, 0.4893414, 0.0803045]).reshape(15,1)
    v_pinotti = [0.464413137,0.006141563,0.464413137,0.006141563,0.013106925,0.006141563,0.033500548,0.006141563]
    v_becker  = [0,0.000000005,0,0.499999948,0.000000088,0.499999948,0.000000005,0.000000005]
    
    return (treat_unit, treat_unit_all, control_units, control_units_all, y_treat, y_treat_all, y_control, y_control_all, Z1,        Z0, X0, X1, w_pinotti, w_becker, v_pinotti, v_becker)

    
    
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

      
def data_compare_df(w_basic, X0, X1, w_pinotti):
    """Outputs a dataframe to show predicted versus actual values of variables"""

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

    fig.update_layout(title='Fig. 3: Treated Unit vs Synthetic Controls with different region weights',
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
##   Iterative implementation   ##  
##############################

def V_iterative(solution_frame_2,control_units):

    """ CVXPY iterative implementation 
    Approach 1: Iterating over a subset of V"""

    class Solution:
        def solve(self, nums):
            if not nums:
                return []
            j=nums[0]
            nums[0]=nums[0]
            for i in range(1,len(nums)):
                k=nums[i]
                nums[i]=j
                j=min(j,k)
            return nums

    ob = Solution()
    ECOS_min  = ob.solve(list(solution_frame_2['RMSPE ECOS'].values))
    SCS_min   = ob.solve(list(solution_frame_2['RMSPE SCS'].values))
    CPLEX_min = ob.solve(list(solution_frame_2['RMSPE CPLEX'].values))
    SLSQP_min = ob.solve(list(solution_frame_2['RMSPE SLSQP'].values))

    figure, axes = plt.subplots(1, 2,figsize=(10,5))
    ax1 = axes[0]
    ax2 = axes[1]

    ax1.plot(ECOS_min,label = 'ECOS')
    ax1.plot(SCS_min,label = 'SCS')
    ax1.plot(CPLEX_min,label = 'CPLEX')
    ax1.plot(SLSQP_min,label = 'SLSQP')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Value of Outer Objective Function')
    ax1.set_ylim(129.75,133)  

    ax1.title.set_text('Fig 4(a) Convergence of  RMSPE')
    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.13), shadow = True, ncol = 4)

    methods = ['ECOS', 'SCS', 'CPLEX']
    violations = [solution_frame_2['ECOS Violation'].mean(),
    solution_frame_2['SCS Violation'].mean(),
    solution_frame_2['CPLEX Violation'].mean()]
    ax2.set_yscale("log")
    ax2.bar(methods,violations,color=['blue', 'orange','green'])

    ax2.set_xlabel('Methods')
    ax2.set_ylabel('Constraint Violation')
    ax2.set_ylim(10**-15,10**-2)
    ax2.title.set_text('Fig 4(b) Average Constraint Violation')

    plt.show()
    
    sorted_df = solution_frame_2.sort_values(by='RMSPE ECOS', ascending=True)

    w_ECOS  = sorted_df.iloc[0][5]
    w_SCS   = sorted_df.iloc[0][6]
    w_CPLEX = sorted_df.iloc[0][7]
    w_SLSQP = sorted_df.iloc[0][8]

    best_weights_region = pd.DataFrame({'ECOS'  : np.round(w_ECOS.ravel(), decimals=3),
                                        'SCS'   : np.round(w_SCS.ravel(), decimals=3),
                                        'SLSQP' : np.round(w_SLSQP.ravel(), decimals=3),
                                        'CPLEX' : np.round(w_CPLEX.ravel(), decimals=3)},
                                         index  = control_units.region.unique())

    best_weights_predictor = pd.DataFrame({'v*': np.round(sorted_df.iloc[0][0].ravel(),decimals=3)},
                                  index= ['GDP per Capita','Investment Rate','Industry VA','Agriculture VA',
                                          'Market Services VA','Non-market Services VA','Human Capital',
                                          'Population Density'])

    display(best_weights_predictor.T)

    display(best_weights_region.T)

    # Organize output into dataframe
    sorted_df = solution_frame_2.sort_values(by='RMSPE ECOS', ascending=True)
    print("\noptimal value with SCS:  ",np.round(sorted_df.iloc[0][2],3), "| constraint violation: ", sorted_df.iloc[0][9])
    print("optimal value with ECOS: ",np.round(sorted_df.iloc[0][1],3), "| constraint violation: ", sorted_df.iloc[0][10])
    print("optimal value with CPLEX:",np.round(sorted_df.iloc[0][3],3), "| constraint violation: ", sorted_df.iloc[0][11])
    print("optimal value with SLSQP:",np.round(sorted_df.iloc[0][4],3), "| constraint violation: n.a.")

    
    
# Data Preparation Function with 8 arguments
def data_prep(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
              predictor_variables, normalize=False):
    
    X = data.loc[data[time_identifier].isin(matching_period)]
    X.index = X.loc[:,unit_identifier]
    
    X0 = X.loc[(X.index.isin(control_units)),(predictor_variables)] 
    X0 = X0.groupby(X0.index).mean().values.T                        #control predictors
    
    X1 = X.loc[(X.index == treat_unit),(predictor_variables)]
    X1 = X1.groupby(X1.index).mean().values.T                        #treated predictors
    
    # outcome variable realizations in matching period -  Z0: control, Z1: treated
    Z0 = np.array(X.loc[(X.index.isin(control_units)),(outcome_variable)]).reshape(len(control_units),len(matching_period)).T  #control outcome
    Z1 = np.array(X.loc[(X.index == treat_unit),(outcome_variable)]).reshape(len(matching_period),1) #treated outcome
    
    if normalize == True:
        # Scaling 
        nvarsV = X0.shape[0]
        big_dataframe = pd.concat([pd.DataFrame(X0), pd.DataFrame(X1)], axis=1)
        divisor = np.sqrt(big_dataframe.apply(np.var, axis=1))
        V = np.zeros(shape=(len(predictor_variables), len(predictor_variables)))
        np.fill_diagonal(V, np.diag(np.repeat(big_dataframe.shape[0],1)))
        scaled_matrix = ((big_dataframe.T) @ (np.array(1/(divisor)).reshape(len(predictor_variables),1) * V)).T

        X0 = np.array(scaled_matrix.iloc[:,0:len(control_units)])
        X1 = np.array(scaled_matrix.iloc[:,len(control_units):(len(control_units)+1)])

        Z0 = Z0.astype('float64')
        Z1 = Z1.astype('float64')
    
    return X0, X1, Z0, Z1

def SCM_print(data,output_object,w_pinotti,Z1,Z0):
    
    # Organize output from SCM into dataframe
    
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2))

    solution_frame_4 = output_object[0]
    w_nested = output_object[1]
    v_nested = output_object[2]
    control_units = data[(data.reg <= 14) | (data.reg == 20)]

    best_weights_region3 = pd.DataFrame({'Region':control_units.region.unique(), 
                                    'W(V*)': np.round(w_nested.ravel(), decimals=3)})

    best_weights_importance3 = pd.DataFrame({'v*': np.round(v_nested, 3)},
                              index= ['GDP per Capita','Investment Rate','Industry VA','Agriculture VA',
                                      'Market Services VA','Non-market Services VA','Human Capital',
                                      'Population Density'])

    display(best_weights_importance3.T)
    display(best_weights_region3.T)

    print('\nV* Constraint Violation: {} \nW*(V*) Constraint Violation:  {}'\
      .format(1-sum(v_nested.ravel()), abs(1-sum(w_nested.ravel()))))

    print('\nOptimizer Weights: {} \nPinotti Weights:  {}'\
      .format(np.round(w_nested.T,3), np.round(w_pinotti,3).T))

    print('\nRMSPE Nested:    {} \nRMSPE Pinotti:   {}'\
      .format(np.round(RMSPE(w_nested),5), np.round(RMSPE(w_pinotti),5)))
  
    
def SCM_v1(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
              predictor_variables,reps=1,solver=cp.ECOS,seed=1):
    
    X0, X1, Z0, Z1 = data_prep(data,unit_identifier,time_identifier,matching_period,treat_unit,
                               control_units,outcome_variable,predictor_variables)
    
    #inner optimization
    def w_optimize(v):
        V = np.zeros(shape=(len(predictor_variables), len(predictor_variables)))
        np.fill_diagonal(V,v)
        W = cp.Variable((len(control_units), 1), nonneg=True)
        objective_function    = cp.Minimize(cp.norm(V @ (X1 - X0 @ W)))
        objective_constraints = [cp.sum(W) == 1]
        objective_problem     = cp.Problem(objective_function, objective_constraints)
        objective_solution    = objective_problem.solve(solver)
        return (W.value)
    
    #outer optimization
    def vmin(v): 
        v = v.reshape(len(predictor_variables),1)
        W = w_optimize(v)
        return ((Z1 - Z0 @ W).T @ (Z1 - Z0 @ W)).ravel()
    
    # constraint on the sum of predictor weights.
    def constr_f(v):
        return float(np.sum(v))

    # Setting Hessian to zero as advised by Diff Evolution to improve performance
    def constr_hess(x,v):
        v=len(predictor_variables)
        return np.zeros([v,v])

    # Must also set Jacobian to zero when setting Hessian to avoid errors
    def constr_jac(v):
        v=len(predictor_variables)
        return np.ones(v)
    
    def RMSPE_f(w):          # RMSPE Calculator
        return np.sqrt(np.mean((w.T @ Z0.T - Z1.T)**2))
    
    # Differential Evolution optimizes the outer objective function vmin()
    def v_optimize(i):
    
        bounds  = [(0,1)]*len(predictor_variables)
        nlc     = NonlinearConstraint(constr_f, 1, 1, constr_jac, constr_hess)
        result  = differential_evolution(vmin, bounds, constraints=(nlc), seed=i)
        v_estim = result.x.reshape(len(predictor_variables),1)  
        return (v_estim)
    
    # Function that brings it all together step-by-step
    def h(x):
    
        v_estim1 = v_optimize(x)              # finding v* once Diff Evolution converges at default tolerance
        w_estim1 = w_optimize(v_estim1)       # finding w*(v*)
        prediction_error = RMSPE_f(w_estim1)
        output_vec = [prediction_error, v_estim1, w_estim1]
        return output_vec

    iterations = []
    iterations = Parallel(n_jobs=-1)(delayed(h)(x) for x in list(range(seed,reps+seed))) # seed for replicability
                                                                                         # can increase repititions
    solution_frame = pd.DataFrame(iterations)
    solution_frame.columns =['Error', 'Relative Importance', 'Weights']
    solution_frame = solution_frame.sort_values(by='Error', ascending=True)

    w_nested = solution_frame.iloc[0][2]
    v_nested = solution_frame.iloc[0][1].T[0]
    
    output = [solution_frame,w_nested,v_nested,RMSPE_f(w_nested)]  # [all repititions, W*, V*, RMSPE]
    
    return output



def global_feasible(Z1,Z0,w_cvxopt,L1_cvxopt,control_units,data,w_becker,v_OSQP,v_GUROBI,v_CPLEX,v_ECOS,v_SCS):
    
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2)) 
    
    print('\nUnrestricted Global Optimum:')

    control_units = data[(data.reg <= 14) | (data.reg ==20)]
    best_weights_region4 = pd.DataFrame({'Region':control_units.region.unique(), 
                                     'CVXOPT W**': np.round(w_cvxopt.ravel(), decimals=6),
                                     'Becker and Klößner R/MSCMT W**': np.round(w_becker.ravel(), decimals=6)})
    
    display(best_weights_region4.T)
    
    print('\nCVXOPT W** Constraint Violation: {}'\
          .format(1 - sum(w_cvxopt.ravel())))

    print('\nRMSPE CVXOPT Optimizer:   {} \nRMSPE Becker and Klößner: {}'\
          .format(np.round(RMSPE(w_cvxopt),5), np.round(RMSPE(w_becker),5)))
    
    print('\nFeasibility Check:')

    best_weights_importance4 = pd.DataFrame({'OSQP V(W**)': np.round(v_OSQP, 3)},
                                  index= ['GDP per Capita','Investment Rate','Industry VA','Agriculture VA',
                                      'Market Services VA','Non-market Services VA','Human Capital',
                                      'Population Density'])
    
    display(best_weights_importance4.T)

    print('\nOSQP V* Constraint Violation:  {}  \nGUROBI V* Constraint Violation: {} \nCPLEX V* Constraint Violation: {}\
          \nECOS V* Constraint Violation: {} \nSCS V* Constraint Violation: {}'\
          .format(1 - sum(v_OSQP.ravel()), 1 - sum(v_GUROBI.ravel()), 1 - sum(v_CPLEX.ravel()), 
                  1 - sum(v_ECOS.ravel()), 1 - sum(v_SCS.ravel())))

    
########    ######

def SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,control_units,outcome_variable,
              predictor_variables,reps=1,solver=cp.ECOS,seed=1,check_global=False,normalize=False,dataprep=True,
              x0=None,x1=None,z0=None,z1=None):
    
    def RMSPE(w):
        return np.sqrt(np.mean((Z1 - Z0 @ w)**2)) 
    
    if dataprep == True:
        X0, X1, Z0, Z1 = data_prep(data,unit_identifier,time_identifier,matching_period,treat_unit,
                                   control_units,outcome_variable,predictor_variables,normalize=normalize)
        
    else:
        X0, X1, Z0, Z1 = x0, x1, z0, z1
        
    #inner optimization
    def w_optimize(v):
        V = np.zeros(shape=(len(predictor_variables), len(predictor_variables)))
        np.fill_diagonal(V,v)
        W = cp.Variable((len(control_units), 1), nonneg=True)
        objective_function    = cp.Minimize(cp.norm(V @ (X1 - X0 @ W)))
        objective_constraints = [cp.sum(W) == 1]
        objective_problem     = cp.Problem(objective_function, objective_constraints)
        objective_solution    = objective_problem.solve(solver)
        return (W.value)
    
    #outer optimization
    def vmin(v): 
        v = v.reshape(len(predictor_variables),1)
        W = w_optimize(v)
        return ((Z1 - Z0 @ W).T @ (Z1 - Z0 @ W)).ravel()
    
    # constraint on the sum of predictor weights.
    def constr_f(v):
        return float(np.sum(v))

    # Setting Hessian to zero as advised by Diff Evolution to improve performance
    def constr_hess(x,v):
        v=len(predictor_variables)
        return np.zeros([v,v])

    # Must also set Jacobian to zero when setting Hessian to avoid errors
    def constr_jac(v):
        v=len(predictor_variables)
        return np.ones(v)
    
    def RMSPE_f(w):          # RMSPE Calculator
        return np.sqrt(np.mean((w.T @ Z0.T - Z1.T)**2))
    
    # Differential Evolution optimizes the outer objective function vmin()
    def v_optimize(i):
    
        bounds  = [(0,1)]*len(predictor_variables)
        nlc     = NonlinearConstraint(constr_f, 1, 1, constr_jac, constr_hess)
        result  = differential_evolution(vmin, bounds, constraints=(nlc), seed=i)
        v_estim = result.x.reshape(len(predictor_variables),1)  
        return (v_estim)
    
    # Function that brings it all together step-by-step
    def h(x):
        
        if check_global == True:
            # Parameter Setup
            Tpre     = Z0.shape[0] 
            nDonors  = Z0.shape[1]
            nvarsV   = X0.shape[0]

            # Quadratic Programming setup
            c1 = (-Z0.T @ Z1).reshape(nDonors,)
            H1 = Z0.T @ Z0  
            A1 = np.ones((nDonors,1)).reshape(nDonors,)
            b1 = np.ones((1,1)).reshape(1,)
            l1 = np.zeros((nDonors, 1)).reshape(nDonors,)
            u1 = np.ones((nDonors, 1)).reshape(nDonors,)

            # Quadratic Programming Execution: Outer Objective Function 
            w_cvxopt = solve_qp(P = H1, q = c1, A = A1, b = b1, lb = l1, ub = u1, solver='cvxopt').reshape(nDonors,1)

            L1_cvxopt = (Z0.T @ Z1)/Tpre + 2/Tpre * (c1.T @ w_cvxopt + 0.5 * w_cvxopt.T @ H1 @ w_cvxopt) 

            V = cp.Variable((8, 1), nonneg=True)
            objective_function    = cp.Minimize(cp.sum_squares(cp.multiply(V,X1 - X0 @ w_cvxopt)))
            objective_constraints = [cp.sum(V) == 1]
            objective_problem     = cp.Problem(objective_function, objective_constraints).solve(solver=cp.OSQP)
            v_OSQP = V.value.ravel()
            prediction_error = RMSPE(w_cvxopt)
            output_vec = [prediction_error, v_OSQP, w_cvxopt]
                
        else:
            v_estim1 = v_optimize(x)              # finding v* once Diff Evolution converges at default tolerance
            w_estim1 = w_optimize(v_estim1)       # finding w*(v*)
            prediction_error = RMSPE_f(w_estim1)
            output_vec = [prediction_error, v_estim1, w_estim1]
        
        return output_vec

    iterations = []
    iterations = Parallel(n_jobs=-1)(delayed(h)(x) for x in list(range(seed,reps+seed))) # seed for replicability
                                                                                         # can increase repititions
    solution_frame = pd.DataFrame(iterations)
    solution_frame.columns =['Error', 'Relative Importance', 'Weights']
    solution_frame = solution_frame.sort_values(by='Error', ascending=True)

    w_nested = solution_frame.iloc[0][2]
    v_nested = solution_frame.iloc[0][1].T[0]
    
    output = [solution_frame,w_nested,v_nested,RMSPE_f(w_nested)]  # [all repititions, W*, V*, RMSPE]
    
    return output

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
    
    n = 20
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

    reorderingV_result  = pd.DataFrame({'Pinotti': np.round(v_pinotti,3),
                                        'Becker': np.round(v_becker,3),
                                       'Minimum': np.round(v_weights.min(axis=1).values,3),
                                       'Mean': np.round(v_weights.mean(axis=1).values,3),
                                       'Maximum':np.round(v_weights.max(axis=1).values,3)},
                                          index= v_weights.index)

    reorderingW_result  = pd.DataFrame({'Pinotti': np.round(w_pinotti[12:15].ravel(),3),
                                        'Becker': np.round(w_becker[12:15].ravel(),3),
                                       'Minimum': np.round(w_weights.min(axis=1).values[12:15],3),
                                       'Mean': np.round(w_weights.mean(axis=1).values[12:15],3),
                                       'Maximum':np.round(w_weights.max(axis=1).values[12:15],3)},
                                          index= ['ABR','MOL','SAR'])

    reorderingRMSPE_result  = pd.DataFrame({'Pinotti': RMSPE(w_pinotti),
                                            'Becker': RMSPE(w_becker),
                                       'Minimum': np.round(RMSPE_orders.min().values,3),
                                       'Mean': np.round(RMSPE_orders.mean().values,3),
                                       'Maximum':np.round(RMSPE_orders.max().values,3)},
                                          index= ['RMSPE'])

    display(pd.concat([reorderingV_result,reorderingW_result,reorderingRMSPE_result], axis=0))


def dynamic_graph_2(w_becker,w_pinotti,w_nested, y_control_all, y_treat_all, data):
    
    """ Dynamic plot for Figure 5: Synthetic Control Optimizer vs. Treated unit
         Plots iterative CVXPY, scipy, Pinotti and Becker versus treated unit outcome """
    y_synth_pinotti = w_pinotti.T @ y_control_all
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

    fig.update_layout(title='Fig 5: Synthetic Control Optimizer vs. Treated unit',
                       xaxis_title='Time', yaxis_title='GDP per Capita')
    fig.show()

    
def table_comapre_2(w_nested,data,predictor_variables,v_nested,v_pinotti,X1,X0):   
#      """ Dataframe with matching period characteristics for Apulia and Basilicata, Synthetic Control, Control Units """ 
        
    x_pred_global = (X0 @ w_nested).ravel()
    X = data.loc[data['year'].isin(list(range(1951, 1961)))]
    control_stats = X.loc[(X.index <= 14) | (X.index ==20),
                      (predictor_variables)].describe().drop(['count','25%', '50%','75%'], axis=0).T
    control_stats = np.round(control_stats,2)
    data_compare  = pd.DataFrame({'Global Opt Weights': np.round(v_nested,3),
                              'Pinotti Weights': np.round(v_pinotti,3),
                              'Apulia and Basilicata':np.round(X1.T[0],3),
                              'Synthetic Control':x_pred_global},
                           index= data.columns[[3,16,11,12,13,14,26,28]])
    
    frames = [data_compare, control_stats]
    result = pd.concat(frames,axis=1)
    print ('\nMatching Period Characteristics: Apulia and Basilicata, Synthetic Control, Control Units')
    display(result)
    

    
def diff_figure_2(w_nested,control_units_all,treat_unit_all,y_control_all,y_treat_all,data):
#      """ Generates Fig 6: Actual vs Synthetic Differences over time: GDP per capita and Murders.
#          Shows differences in evolution of murder rates and GDP per capita between the actual realizations of Apulia 
#          and Basilicata and the ones predicted by the synthetic control unit """
  
    murd_treat_all      = np.array(treat_unit_all.murd).reshape(1, 57)
    murd_control_all    = np.array(control_units_all.murd).reshape(15, 57)
    synth_murd = w_nested.T @ murd_control_all

    synth_gdp = w_nested.T @ y_control_all

    diff_GDP = (((y_treat_all-synth_gdp)/(synth_gdp))*100).ravel()
    diff_murder = (murd_treat_all - synth_murd).ravel()

    diff_data_0 = pd.DataFrame({'Murder Gap':diff_murder,
                             'GDP Gap': diff_GDP},
                             index=data.year.unique())

    year = diff_data_0.index.values
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDP per capita, % Gap')
    ax1.bar(year,diff_data_0['GDP Gap'],width = 0.5,label = 'GDP per capita')
    ax1.axhline(0)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Murder Rate, Difference')
    ax2.plot(diff_data_0['Murder Gap'],color='black',label = 'Murders')
    ax2.axhline(0)
    ax2.tick_params(axis='y')

    plt.axvspan(1975, 1980, color='y', alpha=0.5, lw=0,label='Mafia Outbreak')
    ax1.set_ylim(-20,20)
    ax2.set_ylim(-4.5,4.5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2,loc = 'upper center', bbox_to_anchor = (0.5, -0.15), shadow = True, ncol = 2)
    fig.tight_layout() 
    plt.title('Fig 6: Actual vs Synthetic Differences over time: GDP per capita and Murders')
    plt.show()
    
    return diff_data_0
    
    
# #######################################################
# ##   CVXPY CODE, SETTINGS AND OUTPUT VISUALIZATION   ##
# #######################################################

# def settings():
    
#     """ sets necessary parameters for SCM function, solution_output_SCM function and graphs in section 4 """
    
#     unit_identifier     = 'reg'
#     time_identifier     = 'year'
#     matching_period     = list(range(1951, 1961))
#     treat_unit          = 21
#     control_units       = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
#     outcome_variable    = ['gdppercap']
#     predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
#     reps                = 1
#     entire_period       = list(range(1951, 2008))
#     return unit_identifier, time_identifier, matching_period, treat_unit, control_units, outcome_variable, predictor_variables, reps, entire_period

    
