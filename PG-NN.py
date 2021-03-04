# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:05:50 2021

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
from sklearn.metrics import mean_squared_log_error
from tensorflow.random import set_seed
import statistics

#### Data Setup ####
####################

df = pd.read_csv("C:/Users/laach/Documents/Skole/Masteroppgave/Master2/Code/test-data-func3.csv")
df = df.drop(columns=['Unnamed: 0'],axis=1)


n = len(df)
train_df = df[0:int(n*0.9)]
#val_df = df[int(n*0.7):int(n*0.9)].reset_index(drop=True)
test_df = df[int(n*0.9):].reset_index(drop=True)

input_shape = 5

x_train,y_train = train_df.iloc[:,0:input_shape],train_df.iloc[:,input_shape]
x_test,y_test = test_df.iloc[:,0:input_shape],test_df.iloc[:,input_shape]

# Generate Guided input, must change depending on data function

sin = np.sin
cos = np.cos
sqrt = np.sqrt
exp = np.exp

x_train2_pre = x_train['x5']
x_train2 = []

for i2 in range(0,len(x_train2_pre)):
    #x_train2.append(x_train2_pre[i2]) # func1 side_input = x 
    #x_train2.append(cos(x_train2_pre[i2])) # func2 side_input = cos(x)
    x_train2.append(1/x_train2_pre[i2]) # func3 side_input = 1/x 

x_train2 = np.array(x_train2)



#### Normal neural network ####
###############################

normal_nn_results = [] # 10 plots

for normal_nn in range(0,10): # 10 neural networks

    seed_number = int(normal_nn*10)    
    set_seed(seed_number)

    # MODEL
    x1 = Input(shape=(input_shape,))
    x = Dense(40,activation='relu')(x1)
    x = Dense(40,activation='relu')(x)
    x = Dense(40,activation='relu')(x)
    x = Dense(40,activation='relu')(x)
    output = Dense(1,activation='linear')(x)
    
    model = Model(inputs=x1, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)
    model.compile(loss='mean_squared_error', optimizer=opt)

    #keras.utils.plot_model(model, "nn_model.png", show_shapes=True)

    # TRAIN
    model.fit( 
        x_train,
        y_train,
        batch_size=32,
        epochs=200,
        validation_split=0.2,
        verbose=0,
    )

    # Predict
    
    predictions = [x_test.iloc[[0]].values.tolist()[0]] #get first timeseries element

    for i in range(0,len(x_test-1)):
    
        pred = model.predict(np.array([predictions[i],])) #x_dot
        new_input = predictions[i][1:input_shape] 
        last_elem = new_input[-1] + pred[0][0] #x + x_dot
        new_input.append(last_elem)
        predictions.append(new_input)
    
    pred_plot = pd.DataFrame(predictions)
    pred_plot = pred_plot[input_shape-1].tolist()[:-1]    
    normal_nn_results.append(pred_plot)
     
    print("Done with normal NN number: "+ str(normal_nn))


#### Guided neural network ####
############################### 

guided_nn_results = []

for input_layer in range(0,4):
    
    # Decide which layer the guided input will be 
    if input_layer == 0:
        x12 = Input(shape=(input_shape,))
        x22 = Input(shape=(1,))
        x2 = Dense(40,activation='relu')(x12)     
        x2 = concatenate(inputs=[x2, x22])
        x2 = Dense(40,activation='relu')(x2)
        x2 = Dense(40,activation='relu')(x2)
        x2 = Dense(40,activation='relu')(x2)
        
    if input_layer == 1:
        x12 = Input(shape=(input_shape,))
        x22 = Input(shape=(1,))
        x2 = Dense(40,activation='relu')(x12)     
        x2 = Dense(40,activation='relu')(x2)
        x2 = concatenate(inputs=[x2, x22])
        x2 = Dense(40,activation='relu')(x2)
        x2 = Dense(40,activation='relu')(x2)
        
    if input_layer == 2:
        x12 = Input(shape=(input_shape,))
        x22 = Input(shape=(1,))
        x2 = Dense(40,activation='relu')(x12)     
        x2 = Dense(40,activation='relu')(x2)
        x2 = Dense(40,activation='relu')(x2)
        x2 = concatenate(inputs=[x2, x22])
        x2 = Dense(40,activation='relu')(x2)
        
    if input_layer == 3:
        x12 = Input(shape=(input_shape,))
        x22 = Input(shape=(1,))
        x2 = Dense(40,activation='relu')(x12)     
        x2 = Dense(40,activation='relu')(x2)
        x2 = Dense(40,activation='relu')(x2)
        x2 = Dense(40,activation='relu')(x2)
        x2 = concatenate(inputs=[x2, x22])


    layer_pred = []
    for guided_nn in range(0,10):
        
        seed_number = int(guided_nn*10)    
        set_seed(seed_number)
        
        output2 = Dense(1,activation='linear')(x2)
        model2 = Model(inputs=[x12, x22], outputs=output2)
        opt2 = tf.keras.optimizers.Adam(learning_rate=0.01,decay=1e-4)
        model2.compile(loss='mean_squared_error', optimizer=opt2)
    
        model2.fit(
            [x_train,x_train2],
            y_train,
            batch_size=32,
            epochs=200,
            validation_split=0.2,
            verbose=0,
        )
    
        # Predictions 
        predictions = [x_test.iloc[[0]].values.tolist()[0]]
        
        for ii in range(0,len(x_test-1)):   
            #side_input = cos(predictions[ii][input_shape-1]) # func2
            #side_input = predictions[ii][input_shape-1] # func1
            side_input = 1/predictions[ii][input_shape-1] # func3
            
            pred = model2.predict([np.array([predictions[ii],]),np.array([side_input,])]) #x_dot
            new_input = predictions[ii][1:input_shape]
            last_elem = new_input[-1] + pred[0][0]
            new_input.append(last_elem)
            predictions.append(new_input)

        pred_plot = pd.DataFrame(predictions)
        pred_plot = pred_plot[input_shape-1].tolist()[:-1]
        layer_pred.append(pred_plot)
        print("Done with guided NN number: "+ str(guided_nn) + " Layer" + str(input_layer))
        
    
        
    guided_nn_results.append(layer_pred)
        
        

#### Results & Plot ####
########################

test_plot = x_test['x5'].tolist() #actual predictions


errors = [[],[],[],[],[]]
for nn in range(0,10):
    errors[0] = errors[0] + [mean_squared_error(test_plot,normal_nn_results[nn])]  #normal
    errors[1] = errors[1] + [mean_squared_error(test_plot,guided_nn_results[0][nn])]
    errors[2] = errors[2] + [mean_squared_error(test_plot,guided_nn_results[1][nn])]
    errors[3] = errors[3] + [mean_squared_error(test_plot,guided_nn_results[2][nn])]
    errors[4] = errors[4] + [mean_squared_error(test_plot,guided_nn_results[3][nn])]

errors_mean = [statistics.mean(errors[m]) for m in range(len(errors))]
errors_std = [statistics.stdev(errors[s]) for s in range(len(errors))]

# Histogram

ind = np.arange(5)

plot = plt.bar(ind,errors_mean,width=0.35,yerr=errors_std,log=True,capsize=10)
plt.ylabel('Error')
#plt.title('Function sqrt(x)*sin(x) + x') #func1
#plt.title('Function exp(-x)*sin(x) + cos(x)') #func2
plt.title('Function (1/sqrt(x))*sin(x) + 1/x') #func3
plt.xticks(ind,('normal','layer1','layer2','layer3','layer4'))


#### Plot Normal NN

# =============================================================================
# pred_x = pd.DataFrame(predictions)
# pred_plot = pred_x[input_shape-1].tolist()[:-1]
# 
# test_plot = x_test['x5'].tolist()
# t = np.round(np.linspace(0, 10, len(pred_plot)),1)
# 
# #plt.plot(t,test_plot,'r',t,pred_plot,'b')
# 
# plt.plot(t,test_plot,'r',label='Actual')
# plt.plot(t,pred_plot,'b',label='Prediction no help')
# 
# pred_x2 = pd.DataFrame(predictions2)
# pred_plot2 = pred_x2[input_shape-1].tolist()[:-1]
# 
# 
# plt.plot(t,pred_plot2,'g',label='Prediction with help')
# plt.legend()
# 
#   
# # Error comparisons  
# 
# mse = mean_squared_error(test_plot,pred_plot)
# mse2 = mean_squared_error(test_plot,pred_plot2)
# 
# try:
#     mse_log = mean_squared_log_error(test_plot,pred_plot)
#     mse2_log = mean_squared_log_error(test_plot,pred_plot2)
#     print("Log Error, without: "+ str(mse_log)+ " With: "+str(mse2_log))
# except:
#     print("no log error")
# 
# 
# if abs(mse) > abs(mse2):
#     print("NN without guidance performed WORST, MSE at :"+str(mse)+" With guidance: "+str(mse2))
# else: 
#     print("NN without guidance performed BEST, MSE at: "+str(mse)+" With guidance: "+str(mse2))
# 
# 
# =============================================================================
