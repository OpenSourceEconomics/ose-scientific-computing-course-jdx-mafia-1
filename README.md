# Working title: Constrained optimization of the synthetic control method with application to the Economic Costs of Organized Crime (Pinotti, 2015)
###### Authors: Cremonese J., Syed M.D., Wang X.

<a href="https://nbviewer.org/github/OpenSourceEconomics/ose-scientific-computing-course-jdx-mafia-1/blob/master/Replication%20notebook.ipynb"
   target="_parent">
   <img align="center" 
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png" 
      width="109" height="20"> 
</a> 
<a href="https://gesis.mybinder.org/binder/v2/gh/OpenSourceEconomics/ose-scientific-computing-course-jdx-mafia-1/07a0ac8130a315254cd1257509ca52ab7ec10678?urlpath=lab%2Ftree%2FReplication%20notebook.ipynb"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>

[![Continuous Integration](https://github.com/OpenSourceEconomics/ose-scientific-computing-course-jdx-mafia-1/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenSourceEconomics/ose-scientific-computing-course-jdx-mafia-1/actions/workflows/ci.yml)

This repository contains our replication project of [The Economic Costs of Organised Crime: Evidence from Southern Italy](https://onlinelibrary.wiley.com/doi/abs/10.1111/ecoj.12235) (Pinotti, 2015) for the OSE Scientific Computing class at Bonn University held during the Winter Semester 2021-2022. <br>
Authors: Jessica Cremonese, Muhammad-Danial Syed, Xue Wang.

## The Economic Costs of Organised Crime (Pinotti, 2015)
The issue of organized crime represents a source of potentially adverse socio-economic repercussions across a plethora of communities worldwide, in part due to its prevalence in some form or the other in almost every country. To quantify its impact in the case of the infamous Italian mafia, we replicate the Pinotti (2015) investigation into the economic cost of organized crime in Italy. In doing so, we recreate its main results showing the effect of organized crime on GDP and, in the process, we gain insight into the complexities underlying the Synthetic Control Method (SCM). 

This project begins with an overview of the topic of organized crime followed by an analysis of the paper featuring descriptive statistics, graphs, and a battery of robustness checks. We use this empirical context to motivate an implementation of the SCM, with emphasis on gradually building up the several optimization steps. This allows us to showcase the flaws of our earlier SCM optimization functions and gain some economic intuition in the process. Our implementations draw on the variety of algorithms available in Python's rich scientific libraries which can be used for our intermediate computations. Specifically, we draw on some of the state-of-the-art quadratic programming solvers including CVXOPT, OSQP, GUROBI, CPLEX, ECOS, and SLSQP using convenient wrappers in the form of CVXPY, qpsolvers, and scipy.optimize. 

A final point of departure is the examination of a conflicting result by Becker and Klößner's (2017) response paper to Pinotti (2015). We attempt to reconcile their different findings by investigating some unexpected, recent developments in the literature on the computational challenges of SCM.

## Replication
***> summary of what we implement and why it is worthwhile***

## Repository guide
***> repository folders and files guide***

## References
Pinotti, Paolo. The economic consequences of organized crime: Evidence from Southern Italy. Bank of Italy (2011)

Becker, Martin, and Stefan Klößner. Estimating the economic costs of organized crime by synthetic control methods. Journal of Applied Econometrics 32.7 (2017): 1367-1369.

## Acknowledgements

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/OpenSourceEconomics/ose-scientific-computing-course-jdx-mafia-1/blob/37c9f20c82fa9f328bce4efe5a858feca1d18bbe/LICENSE) 
</a> 
