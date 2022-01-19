""" Auxiliary code for section 6. Robustness Checks of the main notebook """

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
dtafile = './dataset/Pinotti-replication/dataset.dta'
data = pd.read_stata(dtafile)






########## SENSITIVITY ANALYSIS ############
    
def multiplot(SCM, data, unit_identifier, time_identifier, matching_period, treat_unit, control_units, outcome_variable, predictor_variables, reps, entire_period):
      """
      Plots Figure 7: Sensitivity of observed treatment effect to different specifications of the synthetic control
      """
    
    # Conducting the checks: Setting and resetting initial conditions only when needed
    fig, fig_axes = plt.subplots(ncols=3, nrows=3,figsize=(10,10))

    # Only Apulia in treatment group: Changes treat_unit to region number 16
    treat_unit = 16
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,
                        control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)

    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(a) Only Apulia in treatment group',fig_axes[0,0])

    # Only Basilicata in treatment group: Changes treat_unit to region number 17
    treat_unit = 17
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)

    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(b) Only Basilicata in treatment group',fig_axes[0,1])

    # No Molise in control group: Removes region 14 from control_unit
    treat_unit = 21
    control_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20]

    output_object = SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,
                        control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(14,1)

    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(c) No Molise in control group',fig_axes[0,2])

    # No Abruzzo in control group: Removes region 13 from control_unit
    control_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 20]
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(14,1)

    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(d) No Abruzzo in control group',fig_axes[1,0])

    # No Sardinia in control group: Removes region 20 from control_unit
    control_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(14,1)

    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(e) No Sardinia in control group',fig_axes[1,1])

    # Include crimes in predictor variables: add variable 'robkidext' in predictor_variables
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density','robkidext']
    control_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)

    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(f) Include crime rates in predictors',fig_axes[1,2])

    # Include Electricity Consumption and Theft in predictors, remove pop density: add variables 'kwpop, theft', remove 'density'
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'kwpop','theft']
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)

    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(g) Electricity consumption and theft',fig_axes[2,0])

    # Match over 1951 to 1965: change matching_period from (1951,1960) to (1951, 1965)
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    matching_period = list(range(1951, 1966))
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)

    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(h) Matching period 1951-1965',fig_axes[2,1])

    # Match over 1951 to 1975: change matching_period from (1951,1961) to (1951, 1975)
    matching_period = list(range(1951, 1976))
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)

    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(i) Matching period 1951-1975',fig_axes[2,2])

    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
########## PLACEBO TESTING ############
placebo_groups =  [[1,2],[1,7],[1,3],[3,4],[3,5],[3,8],[4,5],[5,6],[5,8],[7,8],[7,9],[8,9],[8,11],
                    [9,11],[9,10],[9,12],[10,11],[10,12],[11,13],[11,12],[12,13],[12,14],[13,14]]





def g(pair):
    
    df = data.copy(deep=True)                                    
    df = df[(df.reg == pair[0]) | (df.reg == pair[1])]         # restrict data to only the pair of regions
    df["murd"] = df["murd"]*df["pop"]/100000                   # convert murder rates (per 100,000) to levels

    df = df.groupby(['year'])[['murd','robkidext','gdp','pop','inv','vaag','vain','vams',
                                 'vanms','vatot','secsc','secpop','area']].sum()   # combine predictors by summing

    df = df.reset_index()
    df.insert(loc=0, column='reg', value=np.zeros(57))         # create column and define newly created region as 0

    df["density"]   = (df["pop"]/df["area"])                     # aggregate predictors for this new region
    df["shvaag"]    = (df["vaag"]/df["vatot"])
    df["shvain"]    = (df["vain"]/df["vatot"])
    df["shvams"]    = (df["vams"]/df["vatot"])
    df["shvanms"]   = (df["vanms"]/df["vatot"])
    df["invrate"]   = (df["inv"]/df["gdp"])
    df["gdppercap"] = df["gdp"]/df["pop"]*1000000
    df["shskill"]   = (df["secsc"]/df["secpop"])
    df["murd"]      = df["murd"]/df["pop"]*100000

    df = df[['year','reg','murd','shvaag','shvain','shvams','shvanms','invrate','gdppercap','shskill','density']]

    df.loc[df['year'] <= 1955, 'murd']    = np.nan          # these two lines were specified in author's STATA code
    df.loc[df['year'] <= 1959, 'invrate'] = np.nan

    X = data[['year','reg','murd','shvaag','shvain','shvams','shvanms','invrate','gdppercap','shskill','density']]

    df = df.append(X)                                      # now append the original data to the new region data
    df = df.reset_index()
    df.drop('index', axis=1, inplace=True) 

    index_names = df[(df.reg == pair[0]) | (df.reg==pair[1]) | (df.reg >=15) & (df.reg<=19) | (df.reg > 20)].index
    df.drop(index_names, inplace=True)                     # drop the pair of regions which make up the new region
    
    seed = 4
    treat_unit = 0
    unit_identifier  = 'reg'
    time_identifier  = 'year'
    matching_period  = list(range(1951, 1961))
    control_units    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
    control_units    = [e for e in control_units if e not in (pair)]
    outcome_variable = ['gdppercap']
    entire_period    = list(range(1951, 2008))
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    
    solver = cp.ECOS
    output_object = SCM(df,unit_identifier,time_identifier,matching_period,treat_unit,
                        control_units,outcome_variable,predictor_variables,solver=solver,seed=seed)
    
    region_weights = output_object[1].reshape(13,1)
    X3 = df.loc[df[time_identifier].isin(entire_period)]
    X3.index = X3.loc[:,unit_identifier]
    
    murd_treat_all   = np.array(X3.loc[(X3.index == treat_unit),('murd')]).reshape(1,len(entire_period))
    murd_control_all = np.array(X3.loc[(X3.index.isin(control_units)),('murd')]).reshape(len(control_units),len(entire_period))
    gdp_control_all  = np.array(X3.loc[(X3.index.isin(control_units)),('gdppercap')]).reshape(len(control_units),len(entire_period))
    gdp_treat_all    = np.array(X3.loc[(X3.index == treat_unit),('gdppercap')]).reshape(1,len(entire_period))
    
    synth_murd  = region_weights.T @ murd_control_all
    synth_gdp   = region_weights.T @ gdp_control_all
    diff_GDP    = (((gdp_treat_all-synth_gdp)/(synth_gdp))*100).ravel()
    diff_murder = (murd_treat_all - synth_murd).ravel()
    diff_data   = pd.DataFrame({'Murder Gap':diff_murder, 'GDP Gap': diff_GDP}, index=df.year.unique())
    
    return diff_data









def placebo_plot(g,placebo_groups,diff_data_0):
    """
    Generates Figure 8: Observed treatment effect for Apulia and Basilicata and placebo units
    """
    
    diff_list = []
    diff_list = Parallel(n_jobs=-1)(delayed(g)(pair) for pair in placebo_groups)

    # Auxiliary
    fig, axes = plt.subplots(1, 2,figsize=(13,4))
    ax1 = axes[0]
    ax2 = axes[1]
    year = diff_data_0.index.values

    for i in range(len(diff_list)):

        ax1.plot(diff_list[i]['GDP Gap'],color='gray',label = 'Placebos' if i == 1 else "")
        ax2.plot(diff_list[i]['Murder Gap'],color='gray',label = 'Placebos' if i == 1 else "")


    ax1.plot(diff_data_0['GDP Gap'],color='black',label = 'Treated Region')
    ax2.plot(diff_data_0['Murder Gap'],color='black',label = 'Treated Region')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDP per capita, % Gap')
    ax1.tick_params(axis='y')
    ax1.set_ylim(-30,30)    
    ax1.title.set_text('Fig 8(a) GDP per capita')
    ax1.axhline(0)

    ax1.set_xlabel('Year')
    ax2.set_ylabel('Murder Rate, Difference')
    ax2.tick_params(axis='y')
    ax2.set_ylim(-4,4)
    ax2.title.set_text('Fig 8(b) Murder Rate')
    ax2.axhline(0)

    ax1.axvspan(1975, 1980, color='y', alpha=0.5, lw=0,label='Mafia Outbreak')
    ax2.axvspan(1975, 1980, color='y', alpha=0.5, lw=0,label='Mafia Outbreak')

    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.22), shadow = True, ncol = 2)
    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.22), shadow = True, ncol = 2)

    plt.show()