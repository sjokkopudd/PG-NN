# -*- coding: utf-8 -*-
"""
Created on Sat May 22 12:01:54 2021

@author: laach
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import concatenate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tensorflow.random import set_seed
import statistics
import seaborn as sns
from operator import itemgetter

#### Data Setup ####
####################

df_train = pd.read_csv("C:/Users/laach/Documents/Skole/Masteroppgave/Master2/Code2/train-HR-long.csv")
df_train = df_train.drop(columns=['Unnamed: 0'],axis=1)
df_test = pd.read_csv("C:/Users/laach/Documents/Skole/Masteroppgave/Master2/Code2/test-HR-long.csv")
df_test = df_test.drop(columns=['Unnamed: 0'],axis=1)

x_train = df_train.iloc[:,0:20]
y_train = df_train.iloc[:,20:23]
x_test = df_test.iloc[:,0:20]
y_test = df_test.iloc[:,20:23]

test_plot_x = df_test.iloc[0,0:4]
test_plot_x = test_plot_x.append(df_test.iloc[:,4])
test_plot_y = df_test.iloc[0,5:9]
test_plot_y = test_plot_y.append(df_test.iloc[:,9])
test_plot_z = df_test.iloc[0,10:14]
test_plot_z = test_plot_z.append(df_test.iloc[:,14])
test_plots = np.vstack((test_plot_x,test_plot_y,test_plot_z)).T

#### Neural networks ####

input_shape = 20
nn_number = 10
normal_nn_results = [] 

histories = [[],[],[],[],[]]

for normal_nn in range(0,nn_number): 

    seed_number = int(normal_nn*10)    
    set_seed(seed_number)

    # MODEL
    x1 = Input(shape=(input_shape,))
    x = Dense(32,activation='relu')(x1)
    x = Dense(64,activation='relu')(x)
    x = Dense(32,activation='relu')(x)
    output = Dense(3,activation='linear')(x)
    model = Model(inputs=x1, outputs=output)
    opt = tf.keras.optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=opt)

    #keras.utils.plot_model(model, "nn_model.png", show_shapes=True)

    # TRAIN
    history = model.fit( 
        x_train,
        y_train,
        batch_size=32,
        epochs=20,
        validation_split=0.2,
        verbose=0,
    )
    histories[0].append(history)

    # Predict
    predictions = [x_test.iloc[[0]].values.tolist()[0]] #get first timeseries element

    for i in range(0,len(x_test-2)):
        
        pred = model.predict(np.array([predictions[i],])) 
        dt = x_test.iloc[1,19] - x_test.iloc[1,18]
        new_input = predictions[i][1:5]
        new_input.append(new_input[-1] + dt*pred[0][0]) #x_dot
        new_input.extend(predictions[i][6:10])
        new_input.append(new_input[-1] + dt*pred[0][1]) #y_dot
        new_input.extend(predictions[i][11:15])
        new_input.append(new_input[-1] + dt*pred[0][2]) #z_dot
        new_input.extend(predictions[i][16:20])
        new_input.append(new_input[-1] + dt) #time
        predictions.append(new_input)
        
        
    pred_plot = pd.DataFrame(predictions)
    pred_plot_x = pred_plot.iloc[0,0:4]
    pred_plot_x = pred_plot_x.append(pred_plot.iloc[:,4])
    pred_plot_x = pred_plot_x.iloc[:-1].reset_index(drop=True) 
    pred_plot_y = pred_plot.iloc[0,5:9]
    pred_plot_y = pred_plot_y.append(pred_plot.iloc[:,9])
    pred_plot_y = pred_plot_y.iloc[:-1].reset_index(drop=True)
    pred_plot_z = pred_plot.iloc[0,10:14]
    pred_plot_z = pred_plot_z.append(pred_plot.iloc[:,14])
    pred_plot_z = pred_plot_z.iloc[:-1].reset_index(drop=True)
    pred_plots = np.vstack((pred_plot_x,pred_plot_y,pred_plot_z)).T
    normal_nn_results.append(pred_plots)
     
    print("Done with normal NN number: "+ str(normal_nn))


#### Guided neural network ####
############################### 

x_train2_pre = x_train[['4','9','14']]
x_train2 = []

for i2 in range(0,(len(x_train2_pre))):
    if i2 != (len(x_train2_pre)-1) :
        dt = (x_train.iloc[i2+1,19]-x_train.iloc[i2,19])
    else: 
        dt = (x_train.iloc[i2,19]-x_train.iloc[i2-1,19])
    x_train2.append(dt*(x_train2_pre.iloc[i2,0]**3))
    #x_train2.append([dt*(x_train2_pre.iloc[i2,0]**3),dt*(x_train2_pre.iloc[i2,0]**2)])
    #x_train2.append(dt*(x_train2_pre.iloc[i2,0]*x_train2_pre.iloc[i2,1]))

x_train2 = np.array(x_train2)


guided_nn_results = []

for input_layer in range(0,4):
    

    layer_pred = []
    for guided_nn in range(0,nn_number):
        
        seed_number = int(guided_nn*10)    
        set_seed(seed_number)
        
        # Decide which layer the guided input will be on
        if input_layer == 0:
            x12 = Input(shape=(input_shape,))
            x22 = Input(shape=(1,))
            x2 = concatenate(inputs=[x12, x22])
            x2 = Dense(32,activation='relu')(x2)     
            x2 = Dense(64,activation='relu')(x2)
            x2 = Dense(32,activation='relu')(x2)
            
        if input_layer == 1:
            x12 = Input(shape=(input_shape,))
            x22 = Input(shape=(1,))
            x2 = Dense(32,activation='relu')(x12)     
            x2 = concatenate(inputs=[x2, x22])
            x2 = Dense(64,activation='relu')(x2)
            x2 = Dense(32,activation='relu')(x2)
            
        if input_layer == 2:
            x12 = Input(shape=(input_shape,))
            x22 = Input(shape=(1,))
            x2 = Dense(32,activation='relu')(x12)     
            x2 = Dense(64,activation='relu')(x2)
            x2 = concatenate(inputs=[x2, x22])
            x2 = Dense(32,activation='relu')(x2)
            
        if input_layer == 3:
            x12 = Input(shape=(input_shape,))
            x22 = Input(shape=(1,))
            x2 = Dense(32,activation='relu')(x12)     
            x2 = Dense(64,activation='relu')(x2)
            x2 = Dense(32,activation='relu')(x2)
            x2 = concatenate(inputs=[x2, x22])
        
        
        output2 = Dense(3,activation='linear')(x2)
        model2 = Model(inputs=[x12, x22], outputs=output2)
        opt2 = tf.keras.optimizers.Adam()
        model2.compile(loss='mean_squared_error', optimizer=opt2)
    

        history = model2.fit(
            [x_train,x_train2],
            y_train,
            batch_size=32,
            epochs=20,
            validation_split=0.2,
            verbose=0,
        )

        histories[input_layer+1].append(history)
    
        # Predictions 
        # Predict
        predictions = [x_test.iloc[[0]].values.tolist()[0]] #get first timeseries element
    
        for i in range(0,len(x_test-2)):
            
            dt = x_test.iloc[i,19] - x_test.iloc[i,18]
            side_input = dt*(predictions[i][4]**3)
            #side_input = [dt*(predictions[i][4]**3),dt*(predictions[i][4]**2)]
            #side_input = dt*(predictions[i][4]*predictions[i][9])
          
            pred = model2.predict([np.array([predictions[i],]),np.array([side_input,])]) 
            new_input = predictions[i][1:5]
            new_input.append(new_input[-1] + dt*pred[0][0]) #x_dot
            new_input.extend(predictions[i][6:10])
            new_input.append(new_input[-1] + dt*pred[0][1]) #y_dot
            new_input.extend(predictions[i][11:15])
            new_input.append(new_input[-1] + dt*pred[0][2]) #z_dot
            new_input.extend(predictions[i][16:20])
            new_input.append(new_input[-1] + dt) #time
            predictions.append(new_input)
            
        pred_plot = pd.DataFrame(predictions)
        pred_plot_x = pred_plot.iloc[0,0:4]
        pred_plot_x = pred_plot_x.append(pred_plot.iloc[:,4])
        pred_plot_x = pred_plot_x.iloc[:-1].reset_index(drop=True) 
        pred_plot_y = pred_plot.iloc[0,5:9]
        pred_plot_y = pred_plot_y.append(pred_plot.iloc[:,9])
        pred_plot_y = pred_plot_y.iloc[:-1].reset_index(drop=True)
        pred_plot_z = pred_plot.iloc[0,10:14]
        pred_plot_z = pred_plot_z.append(pred_plot.iloc[:,14])
        pred_plot_z = pred_plot_z.iloc[:-1].reset_index(drop=True)
        pred_plots = np.vstack((pred_plot_x,pred_plot_y,pred_plot_z)).T
        layer_pred.append(pred_plots)
         
        print("Done with guided NN number: "+ str(guided_nn) + " Layer" + str(input_layer))

    guided_nn_results.append(layer_pred)

#normal_nn_results[9] = normal_nn_results[3]
#guided_nn_results[2][0] = guided_nn_results[1][0]

errors = [[],[],[],[],[]]
for nn in range(0,nn_number):
    errors[0] = errors[0] + [mean_absolute_error(test_plots,normal_nn_results[nn])]  #normal
    errors[1] = errors[1] + [mean_absolute_error(test_plots,guided_nn_results[0][nn])]
    errors[2] = errors[2] + [mean_absolute_error(test_plots,guided_nn_results[1][nn])]
    errors[3] = errors[3] + [mean_absolute_error(test_plots,guided_nn_results[2][nn])]
    errors[4] = errors[4] + [mean_absolute_error(test_plots,guided_nn_results[3][nn])]


errors_mean = [statistics.mean(errors[m]) for m in range(len(errors))]
errors_std = [statistics.stdev(errors[s]) for s in range(len(errors))]

#Plots

sns.set_theme()


ind = np.arange(5)
bar_plot = plt.figure()
plt.bar(ind,errors_mean,width=0.35,yerr=errors_std,log=True,capsize=8)
plt.ylabel('Mean Absolute Error')
plt.xlabel('NN Type')
plt.yscale('log')
#plt.title('Average errors')
plt.xticks(ind,('Normal','Layer 1','Layer 2', 'Layer 3','Output'))
plt.show()

# Line graph

df_nn_x = pd.DataFrame(normal_nn_results[0][:,0])
df_nn_y = pd.DataFrame(normal_nn_results[0][:,1])
df_nn_z = pd.DataFrame(normal_nn_results[0][:,2])
df_l0_x = pd.DataFrame(guided_nn_results[0][0][:,0])
df_l0_y = pd.DataFrame(guided_nn_results[0][0][:,1])
df_l1_x = pd.DataFrame(guided_nn_results[1][0][:,0])
df_l1_y = pd.DataFrame(guided_nn_results[1][0][:,1])
df_l1_z = pd.DataFrame(guided_nn_results[1][0][:,2])
df_l2_x = pd.DataFrame(guided_nn_results[2][0][:,0])
df_l2_y = pd.DataFrame(guided_nn_results[2][0][:,1])
df_l3_x = pd.DataFrame(guided_nn_results[3][0][:,0])
df_l3_y = pd.DataFrame(guided_nn_results[3][0][:,1])

for aa in range(1,len(normal_nn_results)):
    df_nn_x[aa] = normal_nn_results[aa][:,0]
    df_nn_y[aa] = normal_nn_results[aa][:,1]
    df_nn_z[aa] = normal_nn_results[aa][:,2]
    df_l0_x[aa] = guided_nn_results[0][aa][:,0]
    df_l0_y[aa] = guided_nn_results[0][aa][:,1]
    df_l1_x[aa] = guided_nn_results[1][aa][:,0]
    df_l1_y[aa] = guided_nn_results[1][aa][:,1]
    df_l1_z[aa] = guided_nn_results[1][aa][:,2]
    df_l2_x[aa] = guided_nn_results[2][aa][:,0]
    df_l2_y[aa] = guided_nn_results[2][aa][:,1]
    df_l3_x[aa] = guided_nn_results[3][aa][:,0]
    df_l3_y[aa] = guided_nn_results[3][aa][:,1]
    
    
#df_nn_x = df_nn_x.drop(df_nn_x.columns[[3]],axis= 1) 
#df_nn_y = df_nn_y.drop(df_nn_y.columns[[3]],axis= 1)    
   

t = np.round(np.linspace(0, 15, len(df_nn_x)),2)

line_plot_nn_x = plt.figure()
plt.plot(t,test_plot_x,'r',lw=0.7,label='True value')
plt.plot(t,df_nn_x.mean(axis=1),lw=0.7,label='Average prediction')
plt.fill_between(t, df_nn_x.quantile(0.05, axis=1), df_nn_x.quantile(0.95, axis=1), alpha=0.5)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.legend()
plt.show()

line_plot_nn_y = plt.figure()
plt.plot(t,test_plot_y,'r',lw=0.7,label='True value')
plt.plot(t,df_nn_y.mean(axis=1),lw=0.7,label='Average prediction')
plt.fill_between(t, df_nn_y.quantile(0.05, axis=1), df_nn_y.quantile(0.95, axis=1), alpha=0.5)
plt.ylabel('y(t)')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()

line_plot_nn_z = plt.figure()
plt.plot(t,test_plot_z,'r',lw=0.7,label='True value')
plt.plot(t,df_nn_z.mean(axis=1),lw=0.7,label='Average prediction')
plt.fill_between(t, df_nn_z.quantile(0.05, axis=1), df_nn_z.quantile(0.95, axis=1), alpha=0.5)
plt.ylabel('z(t)')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()


line_plot_l1_x = plt.figure()
plt.plot(t,test_plot_x,'r',lw=0.7,label='True value')
plt.plot(t,df_l1_x.mean(axis=1),lw=0.7,label='Average prediction')
plt.fill_between(t, df_l1_x.quantile(0.05, axis=1), df_l1_x.quantile(0.95, axis=1), alpha=0.5)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.legend()
plt.show()


line_plot_l1_y = plt.figure()
plt.plot(t,test_plot_y,'r',lw=0.7,label='True value')
plt.plot(t,df_l1_y.mean(axis=1),lw=0.7,label='Average prediction')
plt.fill_between(t, df_l1_y.quantile(0.05, axis=1), df_l1_y.quantile(0.95, axis=1), alpha=0.5)
plt.ylabel('y(t)')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()

line_plot_l1_z = plt.figure()
plt.plot(t,test_plot_z,'r',lw=0.7,label='True value')
plt.plot(t,df_l1_z.mean(axis=1),lw=0.7,label='Average prediction')
plt.fill_between(t, df_l1_z.quantile(0.05, axis=1), df_l1_z.quantile(0.95, axis=1), alpha=0.5)
plt.ylabel('z(t)')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()

#low sampling rate (0.05) 23.05.2021

#bar_plot.savefig('bar_plot-lotkav-ext.pdf')
line_plot_nn_x.savefig('nn_x-hr-long.pdf')
line_plot_nn_y.savefig('nn_y-hr-long.pdf')
line_plot_nn_z.savefig('nn_z-hr-long.pdf')
line_plot_l1_x.savefig('l1_x-hr-long.pdf')
line_plot_l1_y.savefig('l1_y-hr-long.pdf')
line_plot_l1_z.savefig('l1_z-hr-long.pdf')

# =============================================================================
# loss_plot = plt.figure()
# plt.plot(histories[0][0].history['loss'])
# plt.plot(histories[0][0].history['val_loss'])
# plt.title('Model loss')
# plt.yscale('log')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training loss', 'Validation loss'], loc='upper right')
# plt.show()
# =============================================================================
#loss_plot.savefig("epochs-lotkav-nn.pdf")

