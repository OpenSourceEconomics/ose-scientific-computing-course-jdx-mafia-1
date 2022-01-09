""" Auxiliary code for section 5 of the main notebook """

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
    
    
def table_compare_2(w_nested,w_global,data,predictor_variables,w_becker,w_pinotti,X1,X0):   
#      """ Dataframe with matching period characteristics for Apulia and Basilicata, Synthetic Control, Control Units """ 
        
    x_pred_nested  = (X0 @ w_nested).ravel()
    x_pred_global  = (X0 @ w_global).ravel()
    x_pred_pinotti = (X0 @ w_pinotti).ravel()
    x_pred_becker  = (X0 @ w_becker).ravel()
    X = data.loc[data['year'].isin(list(range(1951, 1961)))]
    
    control_stats = X.loc[(X.index <= 14) | (X.index ==20),
                      (predictor_variables)].describe().drop(['std','count','25%', '50%','75%'], axis=0).T
    control_stats = np.round(control_stats,2)
    rounded_x1  = np.array([2395.0, 0.32, 0.22, 0.15, 0.4, 0.23, 0.17, 134.78])
    data_compare  = pd.DataFrame({'Treated Actual':rounded_x1,
                                  'Pinotti Synth': x_pred_pinotti,
                                  'Becker MSCMT': x_pred_becker,
                                  'SCM/Nested': x_pred_nested,
                                  'SCM/Global': x_pred_global},
                           index= data.columns[[3,16,11,12,13,14,26,28]])
                             
    
    frames = [data_compare, control_stats]
    result = pd.concat(frames,axis=1)
    result = result.round(2)
    result.index = ['GDP per Capita','Investment Rate','Industry VA','Agriculture VA','Market Services VA',
                    'Non-market Services VA','Human Capital','Population Density']
    
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