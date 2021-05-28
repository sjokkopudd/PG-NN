# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
import math

input_window = 5

function = 3

if function == 1:

    def lotkavolterra(t, z, a, b, c, d):
        x, y = z
        return [a*x - b*x*y, -c*y + d*x*y]
    
    data_train = []
    data_test = []
    
    for init_value in range(7,30):
        
        if init_value == 29:
            sol = solve_ivp(lotkavolterra, [0, 15], [init_value+11, 2], args=(0.6, 0.1, 0.1, 0.01),
                            dense_output=True,max_step=0.05) 
        else: 
            sol = solve_ivp(lotkavolterra, [0, 15], [init_value, 2], args=(0.6, 0.1, 0.1, 0.01),
                        dense_output=True,max_step=0.05) 
            
        y = sol.y
        t = sol.t
        
        for i in range(0,len(t)-input_window):
            
            new_input = []
            new_input.extend(y[0][i:i+input_window]) #x
            new_input.extend(y[1][i:i+input_window]) #y
            new_input.extend(t[i:i+input_window]) # time
            target_x = (y[0][i+input_window]-y[0][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) 
            target_y = (y[1][i+input_window]-y[1][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) 
            new_input.append(target_x)
            new_input.append(target_y)
            if init_value == 29:
                data_test.append(new_input)
            else:
                data_train.append(new_input)
            
    df_train = pd.DataFrame(data_train)
    df_train.to_csv('train-lotkav-outside.csv')
    df_test = pd.DataFrame(data_test)
    df_test.to_csv('test-lotkav-outside.csv')


if function == 2:    
    
    def duffing(t, X):
        x, x_dot = X
        x_dotdot = 2.3*math.cos(0.2*t) - x_dot - 0.5*x - x**3
        return [x_dot,x_dotdot]
    
    data_train = []
    data_test = []
    
    init_values = (np.random.random_sample((2,25))*2-1).T

    
    #init_values = [[1,1],[1,-1],[-1,1],[1,0],[0,1],[-1,-1],[0.5,-0.5]]
    
    for j,values in enumerate(init_values):
        
        sol = solve_ivp(duffing, [0, 25], values, dense_output=True,max_step=0.05) 
        y = sol.y
        t = sol.t
        
        for i in range(0,len(t)-input_window):
            
            new_input = []
            new_input.extend(y[1][i:i+input_window]) #x_dot
            new_input.extend(y[0][i:i+input_window]) #x
            new_input.extend(t[i:i+input_window]) # time
            #target_x = (y[0][i+input_window]-y[0][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) 
            target_y = (y[1][i+input_window]-y[1][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) #x_dotdot 
            #new_input.append(target_x)
            new_input.append(target_y)
            if j == 6:
                data_test.append(new_input)
            else:
                data_train.append(new_input)
            
    df_train = pd.DataFrame(data_train)
    #df_train.to_csv('train-duff-redo.csv')
    df_test = pd.DataFrame(data_test)
    #df_test.to_csv('test-duff-redo.csv')
    
# =============================================================================
# duff_plot = plt.figure()
# plt.plot(t,y[0],'navy',lw=1.1,label='x(t)')
# #plt.plot(t,y[1],'navy',lw=0.7,label='x_dot')
# plt.xlabel('t')
# plt.ylabel('x(t)')
# plt.legend()
# plt.show()
# #duff_plot.savefig('duff-example.pdf')
# =============================================================================
    
if function == 3:

    def hr(t, xyz):
        x, y, z = xyz
        a,b,c,d,s,x_r,r = 1,3,1,5,4,(-8/5),0.005
        x_dot= y -a*(x**3) + b*(x**2) - z + 5
        y_dot= c - d*(x**2) - y
        z_dot= r*(s*(x-x_r)-z)
        return [x_dot,y_dot,z_dot]
    
    data_train = []
    data_test = []
    
    init_values = (np.random.random_sample((3,25))*2-1).T
    
    for j,values in enumerate(init_values):
        

        sol = solve_ivp(hr, [0, 15], values, dense_output=True,max_step=0.005) 

            
        y = sol.y
        t = sol.t
        
        for i in range(0,len(t)-input_window):
            
            new_input = []
            new_input.extend(y[0][i:i+input_window]) #x
            new_input.extend(y[1][i:i+input_window]) #y
            new_input.extend(y[2][i:i+input_window]) #z
            new_input.extend(t[i:i+input_window]) # time
            target_x = (y[0][i+input_window]-y[0][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) 
            target_y = (y[1][i+input_window]-y[1][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) 
            target_z = (y[2][i+input_window]-y[2][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) 
            new_input.append(target_x)
            new_input.append(target_y)
            new_input.append(target_z)
            if j == 6:
                data_test.append(new_input)
            else:
                data_train.append(new_input)
            
    df_train = pd.DataFrame(data_train)
    df_train.to_csv('train-HR-long.csv')
    df_test = pd.DataFrame(data_test)
    df_test.to_csv('test-HR-long.csv')  

if function == 4:

# =============================================================================
#     def trials(t, xyz):
#         x, y, z = xyz
#         k1,k_1,k2,k_2,k3,k_3,k4,k_4,k5,k_5 = 30,0.25,1,0.0001,10,0.001,1,0.001,16.5,0.5
#         x_dot= k1*x - k_1*(x**2) - k2*x*y + k_2*(y**2) - k4*x*z + k_4
#         y_dot= k2*x*y - k_2*(y**2) - k3*y + k_3
#         z_dot= -k4*x*z + k_4 + k5*z - k_5*(z**2)
#         return [x_dot,y_dot,z_dot]
# =============================================================================

    def trials(t, xyz):
        x, y, z = xyz
        a,b,c = 10,(8/3),28
        x_dot= a*(y-x)
        y_dot= x*(c-z)-y
        z_dot= x*y - b*z
        return [x_dot,y_dot,z_dot]
    
    data_train = []
    data_test = []
    
    init_values = (np.random.random_sample((3,25))*2).T
    
    for j,values in enumerate(init_values):
        

        sol = solve_ivp(trials, [0, 15], values, dense_output=True,max_step=0.005) 

            
        y = sol.y
        t = sol.t
        
        for i in range(0,len(t)-input_window):
            
            new_input = []
            new_input.extend(y[0][i:i+input_window]) #x
            new_input.extend(y[1][i:i+input_window]) #y
            new_input.extend(y[2][i:i+input_window]) #z
            new_input.extend(t[i:i+input_window]) # time
            target_x = (y[0][i+input_window]-y[0][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) 
            target_y = (y[1][i+input_window]-y[1][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) 
            target_z = (y[2][i+input_window]-y[2][i+input_window-1])/(t[i+input_window]-t[i+input_window-1]) 
            new_input.append(target_x)
            new_input.append(target_y)
            new_input.append(target_z)
            if j == 16:
                data_test.append(new_input)
            else:
                data_train.append(new_input)
            
    df_train = pd.DataFrame(data_train)
    df_train.to_csv('train-trials-lor-long.csv')
    df_test = pd.DataFrame(data_test)
    df_test.to_csv('test-trials-lor-long.csv')  
  


if function == 5:

    def plotting(t, xyz):
        x, y, z = xyz
        k1,k_1,k2,k_2,k3,k_3,k4,k_4,k5,k_5 = 30,0.25,1,0.0001,10,0.001,1,0.001,16.5,0.5
        x_dot= k1*x - k_1*(x**2) - k2*x*y + k_2*(y**2) - k4*x*z + k_4
        y_dot= k2*x*y - k_2*(y**2) - k3*y + k_3
        z_dot= -k4*x*z + k_4 + k5*z - k_5*(z**2)
        return [x_dot,y_dot,z_dot]
    
    data_train = []
    data_test = []
    
    init_values = [1,1.5,0.3]
    

    sol = solve_ivp(plotting, [0, 3], init_values, dense_output=True,max_step=0.001) 

            
    y = sol.y
    t = sol.t
    

    
    ax = plt.figure().add_subplot(projection='3d')
    
    ax.plot(y[0], y[1], y[2], 'navy', lw=0.7)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    
    plt.show()
    
    ax.figure.savefig('WR-example.pdf',bbox_inches = 'tight',pad_inches = 0)


    
     
