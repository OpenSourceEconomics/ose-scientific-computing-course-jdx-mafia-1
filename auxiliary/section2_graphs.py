""" Auxiliary code with functions for plotting static graphs in section 2 of the main notebook """

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
dtafile = './dataset/Pinotti-replication/dataset.dta'
data = pd.read_stata(dtafile)





def plot_prep(frame_name):
    """ 
    Defines necessary dataframes for future plots. 
    Example: data("df2") generates necessary dataframe for figure 2.1
    """
    
    data = pd.read_stata(dtafile)
    if frame_name == "df1":
        df = data[data['year'] >= 1983]
    elif frame_name == "df2":
        df = data[data['year'] >= 1983]
        df = df.groupby(['region', 'reg'])[['gdppercap', 'mafia', 'murd', 'ext', 'fire', 'kidnap', 'rob', 'smug', 'drug', 'theft', 'orgcrime']].mean()
        df = df.reset_index()
    elif frame_name == "df3":
        grouped = (data['reg'] > 20) & (data['reg'] < 25)
        df = data.loc[grouped, ['murd', 'year', 'region']]
        df = df[df['year'] >= 1956]
        df = df[['murd', 'year', 'region']]
        df = df.pivot(index = 'year', columns = 'region', values = 'murd')
        # rename df3 columns for a nice looking legend
        df = df.rename(columns = {'HIS':'Sicily, Campania, Calabria', 'NEW':'Apulia, Basilicata',
                                   'NTH':'Centre-North', 'STH':'Rest of South'})
    return df
        


    
    


def fig1_mafia_presence_avg(df2):
    """ 
    Plots Figure 1: GDP per capita and mafia presence, averaged over 1983–2007
    """

    color = np.where((df2['reg'] == 15) | (df2['reg'] == 18) | (df2['reg'] == 19), 'midnightblue',           # EXCLUDED
                 np.where((df2['reg'] == 16) | (df2['reg'] == 17), 'mediumslateblue',                    # TREATED
                 np.where((df2['reg'] <= 12) | (df2['reg'] == 20), 'salmon', 'none')))                   # THE REST

    df2.plot.scatter('mafia', 'gdppercap', c = color, s = 10, linewidth = 3, 
                 xlabel = 'Presence of mafia organisations', ylabel = 'GDP per capita', ylim = [7000,15000], xlim = [0,2.25])
    n = ['Basilicata', 'Calabria', 'Campania', 'Apulia', 'Sicily']
    j, z = 0, [1, 2, 3, 16, 18]
    for i in z:
        plt.annotate(n[j], (df2.mafia[i], df2.gdppercap[i]), xytext = (0,1), 
                     textcoords = 'offset points', ha = 'left', va = 'bottom', rotation = 24)
        j += 1
        
    return plt.show()
    


def fig2_murder_rate_graphs(df3, df2):
    """ 
    Plots Fig 2(a): Murder rate time series plot 1956-2007
    and   Fig 2(b): Organized Crime and Murder 1983-2007 
    """

    color = np.where((df2['reg'] == 15) | (df2['reg'] == 18) | (df2['reg'] == 19), 'midnightblue',           # EXCLUDED
                     np.where((df2['reg'] == 16) | (df2['reg'] == 17), 'mediumslateblue',                    # TREATED
                     np.where((df2['reg'] <= 12) | (df2['reg'] == 20), 'salmon', 'none')))                   # THE REST

    figure, axes = plt.subplots(1, 2,figsize=(10,5))

    ### Figure 2(a) ###
    ax1 = df3.plot(colormap = 'seismic', rot = 'vertical',
               xticks   = [1956,1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010],
               yticks   = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9],
               xlabel   = 'Year', ylabel = 'Homicides x 100,000 Inhabitants', 
               title    = 'Fig 2(a): Murder rate time series plot 1956-2007', ax = axes[0])
    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.22), shadow = True, ncol = 2)

    ### Figure 2(b) ###
    ax2 = df2.plot.scatter('mafia', 'murd', c = color, s = 10, linewidth = 3, 
                       xlabel = 'Mafia Allegations ex Art. 416-bis × 100,000 Inhabitants', 
                       ylabel = 'Homicides x 100,000 Inhabitants', 
                       ylim = [0,7], xlim = [0,2.1], title = 'Fig 2(b): Organized Crime and Murder 1983-2007', ax = axes[1])
    n = ['Basilicata', 'Calabria','Campania','Apulia','Sicily']
    j, z = 0, [1, 2, 3, 16, 18]
    for i in z:
        plt.annotate(n[j], (df2.mafia[i], df2.murd[i]), xytext = (0,1),
                     textcoords = 'offset points', ha = 'left', va = 'bottom')
        j += 1

    plt.tight_layout()
    return plt.show()
