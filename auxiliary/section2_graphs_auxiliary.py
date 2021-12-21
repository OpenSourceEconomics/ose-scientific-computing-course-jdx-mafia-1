""" Auxiliary code with functions for plotting static graphs in section 2 of the main notebook """

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


%matplotlib inline
%config InlineBackend.figure_format = 'svg'
#plt.rcParams['figure.figsize'] = [5.5, 3.5]
plt.rcParams['figure.figsize'] = [6, 4.0]
#plt.rcParams['figure.dpi'] = 80


dtafile = './dataset/Pinotti-replication/dataset.dta'
data = pd.read_stata(dtafile)




def dataframes():  
    """ Define necesary dataframes for future plots """
    
    df1 = data[data['year'] >= 1983]
    
    df2 = df1.groupby(['region', 'reg'])[['gdppercap', 'mafia', 'murd', 'ext', 'fire', 'kidnap', 'rob', 'smug', 'drug', 'theft', 'orgcrime']].mean()
    df2 = df2.reset_index()
    grouped = (data['reg'] > 20) & (data['reg'] < 25)

    df3 = data.loc[grouped, ['murd', 'year', 'region']]
    df3 = df3[df3['year'] >= 1956]
    df3 = df3[['murd', 'year', 'region']]
    df3 = df3.pivot(index = 'year', columns = 'region', values = 'murd')
    # rename df3 columns for a nice looking legend
    df3 = df3.rename(columns = {'HIS':'Sicily, Campania, Calabria', 'NEW':'Apulia, Basilicata',
                            'NTH':'Centre-North', 'STH':'Rest of South'})
    color = np.where((df2['reg'] == 15) | (df2['reg'] == 18) | (df2['reg'] == 19), 'midnightblue',           # EXCLUDED
                     np.where((df2['reg'] == 16) | (df2['reg'] == 17), 'mediumslateblue',                    # TREATED
                     np.where((df2['reg'] <= 12) | (df2['reg'] == 20), 'salmon', 'none')))                   # THE REST





def mafia_presence_avg():
    """ Plots Figure 2.1: GDP per capita and mafia presence, 1983–2007 average """
    
    df2.plot.scatter('mafia', 'gdppercap', c = color, s = 10, linewidth = 3, 
                 xlabel = 'Presence of mafia organisations', ylabel = 'GDP per capita', ylim = [7000,15000], xlim = [0,2.25],
                 title = 'Figure 2.1: GDP per capita and mafia presence, 1983–2007 average')
    n = ['Basilicata', 'Calabria', 'Campania', 'Apulia', 'Sicily']
    j, z = 0, [1, 2, 3, 16, 18]
    for i in z:
        plt.annotate(n[j], (df2.mafia[i], df2.gdppercap[i]), xytext = (0,1), 
                 textcoords = 'offset points', ha = 'left', va = 'bottom', rotation = 15)
        j += 1
        


def murder_rate_graphs():
    """ Plots Figure 2.2: Murder rate time series plot 1956-2007 
    and Figure 2.3: Organized Crime and Averge Murder 1983-2007 """
    
    figure, axes = plt.subplots(1, 2,figsize=(10,5))

    ax1 = df3.plot(colormap = 'seismic', rot = 'vertical',
               xticks   = [1956,1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010],
               yticks   = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9],
               xlabel   = 'Year', ylabel = 'Homicides x 100,000 Inhabitants', 
               title    = 'Fig 2.2: Murder rate time series plot 1956-2007', ax = axes[0])
    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.22), shadow = True, ncol = 2)

    ax2 = df2.plot.scatter('mafia', 'murd', c = color, s = 10, linewidth = 3, 
                       xlabel = 'Mafia Allegations ex Art. 416-bis × 100,000 Inhabitants', 
                       ylabel = 'Homicides x 100,000 Inhabitants', 
                       ylim = [0,7], xlim = [0,2.1], title = 'Fig 2.3: Organized Crime and Averge Murder 1983-2007', ax = axes[1])
    n = ['Basilicata', 'Calabria','Campania','Apulia','Sicily']
    j, z = 0, [1, 2, 3, 16, 18]
    for i in z:
        plt.annotate(n[j], (df2.mafia[i], df2.murd[i]), xytext = (0,1),
                 textcoords = 'offset points', ha = 'left', va = 'bottom')
        j += 1

    plt.tight_layout()
    plt.show()
