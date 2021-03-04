# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:41:01 2021

@author: laach

Used to generate data for the simpler functions
"""

from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd

sqrt = np.sqrt
sin = np.sin
cos = np.cos
exp = np.exp

function = 3 #1,2 or 3

####Different functions

def dx_dt1(t,x):
    return (sqrt(x)*sin(x)) + x

def dx_dt2(t,x):
    return (exp(-x)*sin(x)) + cos(x)

def dx_dt3(t,x):
    return ((1/sqrt(x))*sin(x)) + 1/x


#### Solve ODE

t_val = np.round(np.linspace(0, 10, 101),1)
y0 = np.linspace(0.1, 1, 10)

if function == 1:
    sol = solve_ivp(dx_dt1,[0,10],y0,t_eval=t_val)
if function == 2:
    sol = solve_ivp(dx_dt2,[0,10],y0,t_eval=t_val)
if function == 3:
    sol = solve_ivp(dx_dt3,[0,10],y0,t_eval=t_val)


### Set data on form [x,x_dot]

data_prepre = sol.y
data_pre = []

for i in range(0,len(data_prepre)):
    for j in range(0,len(data_prepre[0])):
        if j < len(data_prepre[0])-1:
            data_pre.append([data_prepre[i][j], data_prepre[i][j+1] - data_prepre[i][j]]) #[x,x_dot]



#### Get data on form [x_1,...,x_n, x_dot]

b_len = 100 #batch length
input_window = 5

x_data = []
y_data = []


for ii in range(0,len(data_pre)):
    
    if (ii + input_window-1 < (b_len*(ii//b_len+1))):
        add_list = []
        for jj in range(0,input_window):
            add_list.append(data_pre[ii+jj][0])
        y_data.append(data_pre[ii+input_window-1][1]) #x_dot
        x_data.append(add_list)  #x_1 to x_inputWindow


dfx = pd.DataFrame(x_data,columns=['x1','x2','x3','x4','x5'])
dfy = pd.DataFrame(y_data,columns=['x_dot'])
df = dfx.join(dfy)

if function == 1:
    df.to_csv('test-data-func1.csv')
if function == 2:  
    df.to_csv('test-data-func2.csv')
if function == 3:
    df.to_csv('test-data-func3.csv')


