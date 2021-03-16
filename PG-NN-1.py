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
from sklearn.metrics import mean_absolute_error
from tensorflow.random import set_seed
import statistics
import seaborn as sns
from operator import itemgetter

#### Data Setup ####
####################

df = pd.read_csv("C:/Users/laach/Documents/Skole/Masteroppgave/Master2/Code/test-data-func1v2.csv")
df = df.drop(columns=['Unnamed: 0'],axis=1)


n = len(df)
train_df = df[0:int(n*0.9)]
test_df = df[int(n*0.9):].reset_index(drop=True)

test_samples = 10
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

dt = 0.01

for i2 in range(0,len(x_train2_pre)):
    x_train2.append(dt*(sin(x_train2_pre[i2]))) # func1 side_input = x 
    #x_train2.append(dt*(cos(x_train2_pre[i2]))) # func2 side_input = cos(x)
    #x_train2.append(dt*(1/x_train2_pre[i2])) # func3 side_input = 1/x 

x_train2 = np.array(x_train2)

#### Normal neural network ####
###############################

nn_number = 100
normal_nn_results = [] 

histories = [[],[],[],[],[]]

for normal_nn in range(0,nn_number): 

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
    history = model.fit( 
        x_train,
        y_train,
        batch_size=32,
        epochs=150,
        validation_split=0.2,
        verbose=0,
    )

    #histories[0].append(history)

    # Predict
    
    predictions = [x_test.iloc[[0]].values.tolist()[0]] #get first timeseries element

    
    for i in range(0,len(x_test-1)):
        
        if ((i+1)%int((len(x_test))/test_samples) == 0) and (i != len(x_test)-1): #new start
            predictions.append(x_test.iloc[[i+1]].values.tolist()[0])
            
        else:
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
    

    layer_pred = []
    for guided_nn in range(0,nn_number):
        
        seed_number = int(guided_nn*10)    
        set_seed(seed_number)
        
        # Decide which layer the guided input will be on 
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
        
        
        output2 = Dense(1,activation='linear')(x2)
        model2 = Model(inputs=[x12, x22], outputs=output2)
        opt2 = tf.keras.optimizers.Adam(learning_rate=0.01,decay=1e-4)
        model2.compile(loss='mean_squared_error', optimizer=opt2)
    

        history = model2.fit(
            [x_train,x_train2],
            y_train,
            batch_size=32,
            epochs=150,
            validation_split=0.2,
            verbose=0,
        )

        #histories[input_layer+1].append(history)
    
        # Predictions 
        predictions = [x_test.iloc[[0]].values.tolist()[0]]
        
        for ii in range(0,len(x_test-1)): 
            
            if ((ii+1)%int((len(x_test))/test_samples) == 0) and (ii != len(x_test)-1): #new start
                predictions.append(x_test.iloc[[ii+1]].values.tolist()[0])
            
            else:
                #Choose Function
                side_input = dt*(sin(predictions[ii][input_shape-1])) # func1
                #side_input = dt*(cos(predictions[ii][input_shape-1])) # func2
                #side_input = dt*(1/predictions[ii][input_shape-1]) # func3
                
                #Prediction
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

sns.set_theme()

plot_title = 'Function sqrt(x)*sin(x) + x' #func1
#plot_title = 'Function exp(-x)*sin(x) + cos(x)'#func2
#plot_title = 'Function (1/sqrt(x))*sin(x) + 1/x' #func3 

errors = [[],[],[],[],[]]
for nn in range(0,nn_number):
    errors[0] = errors[0] + [mean_squared_error(test_plot,normal_nn_results[nn])]  #normal
    errors[1] = errors[1] + [mean_squared_error(test_plot,guided_nn_results[0][nn])]
    errors[2] = errors[2] + [mean_squared_error(test_plot,guided_nn_results[1][nn])]
    errors[3] = errors[3] + [mean_squared_error(test_plot,guided_nn_results[2][nn])]
    errors[4] = errors[4] + [mean_squared_error(test_plot,guided_nn_results[3][nn])]

# Keep middle 80% of sorted data
sorted_indices = [[],[],[],[],[]]
for mm in range(0,5):
    sorted_indices[mm], errors[mm] = zip(*sorted(enumerate(errors[mm]), key=itemgetter(1)))
    errors[mm] = list(errors[mm])
    errors[mm] = errors[mm][int(len(errors[mm])*0.1) : int(len(errors[mm])*0.9)]

    sorted_indices[mm] = list(sorted_indices[mm]) 
    sorted_indices[mm] = sorted_indices[mm][int(len(sorted_indices[mm])*0.1) : int(len(sorted_indices[mm])*0.9)]

normal_nn_results = [normal_nn_results[index] for index in sorted_indices[0]] 
for xx in range(1,5): 
    guided_nn_results[xx-1] = [guided_nn_results[xx-1][index2] for index2 in sorted_indices[xx]] 


errors_mean = [statistics.mean(errors[m]) for m in range(len(errors))]
errors_std = [statistics.stdev(errors[s]) for s in range(len(errors))]

#### Histogram plot ####

ind = np.arange(5)

bar_plot = plt.figure()
plt.bar(ind,errors_mean,width=0.35,yerr=errors_std,log=True,capsize=10)
plt.ylabel('Mean Squared Error')
plt.xlabel('NN Type')
plt.title(plot_title)
plt.xticks(ind,('Normal','Layer 1','Layer 2','Layer 3','Layer 4'))
plt.show()


#### Line graphs ####

nnn_df =  pd.DataFrame(normal_nn_results)
l1_df = pd.DataFrame(guided_nn_results[0])
l2_df = pd.DataFrame(guided_nn_results[1])
l3_df = pd.DataFrame(guided_nn_results[2])
l4_df = pd.DataFrame(guided_nn_results[3])

plot_nnn = pd.DataFrame(0, index=range(int(nnn_df.shape[1]/test_samples)), columns=['Avg','Max','Min'])
plot_l1 = pd.DataFrame(0, index=range(int(l1_df.shape[1]/test_samples)), columns=['Avg','Max','Min'])
plot_l2 = pd.DataFrame(0, index=range(int(l2_df.shape[1]/test_samples)), columns=['Avg','Max','Min'])
plot_l3 = pd.DataFrame(0, index=range(int(l3_df.shape[1]/test_samples)), columns=['Avg','Max','Min'])
plot_l4 = pd.DataFrame(0, index=range(int(l4_df.shape[1]/test_samples)), columns=['Avg','Max','Min'])

for tt in range(int(l1_df.shape[1]/test_samples)):

    plot_nnn.loc[tt,'Avg'] = nnn_df.iloc[:,tt].mean()
    plot_nnn.loc[tt,'Max'] = nnn_df.iloc[:,tt].max()
    plot_nnn.loc[tt,'Min'] = nnn_df.iloc[:,tt].min()

    plot_l1.loc[tt,'Avg'] = l1_df.iloc[:,tt].mean()
    plot_l1.loc[tt,'Max'] = l1_df.iloc[:,tt].max()
    plot_l1.loc[tt,'Min'] = l1_df.iloc[:,tt].min()
    
    plot_l2.loc[tt,'Avg'] = l2_df.iloc[:,tt].mean()
    plot_l2.loc[tt,'Max'] = l2_df.iloc[:,tt].max()
    plot_l2.loc[tt,'Min'] = l2_df.iloc[:,tt].min()
    
    plot_l3.loc[tt,'Avg'] = l3_df.iloc[:,tt].mean()
    plot_l3.loc[tt,'Max'] = l3_df.iloc[:,tt].max()
    plot_l3.loc[tt,'Min'] = l3_df.iloc[:,tt].min()
    
    plot_l4.loc[tt,'Avg'] = l4_df.iloc[:,tt].mean()
    plot_l4.loc[tt,'Max'] = l4_df.iloc[:,tt].max()
    plot_l4.loc[tt,'Min'] = l4_df.iloc[:,tt].min()


t = np.round(np.linspace(0, 2, int(len(test_plot)/test_samples)),2)

line_plot_nn = plt.figure()
plt.plot(t,test_plot[0:int(len(test_plot)/test_samples)],'r',lw=0.7,label='Actual')
plt.plot(t,plot_nnn['Avg'],'navy',lw=0.7,label='Layer 1')
plt.fill_between(t, plot_nnn['Min'], plot_nnn['Max'],alpha=0.8,color='cornflowerblue')
plt.title(plot_title + '  for normal network')
plt.yscale('log')
plt.legend()
plt.show()

line_plot_l1 = plt.figure()
plt.plot(t,test_plot[0:int(len(test_plot)/test_samples)],'r',lw=0.7,label='Actual')
plt.plot(t,plot_l1['Avg'],'navy',lw=0.7,label='Layer 1')
plt.fill_between(t, plot_l1['Min'], plot_l1['Max'],alpha=0.8,color='cornflowerblue')
plt.title(plot_title + '  for layer 1')
plt.yscale('log')
plt.legend()
plt.show()

line_plot_l2 = plt.figure()
plt.plot(t,test_plot[0:int(len(test_plot)/test_samples)],'r',lw=0.7,label='Actual')
plt.plot(t,plot_l2['Avg'],'navy',lw=0.7,label='Layer 1')
plt.fill_between(t, plot_l2['Min'], plot_l2['Max'],alpha=0.8,color='cornflowerblue')
plt.title(plot_title + '  for layer 2')
plt.yscale('log')
plt.legend()
plt.show()

line_plot_l3 = plt.figure()
plt.plot(t,test_plot[0:int(len(test_plot)/test_samples)],'r',lw=0.7,label='Actual')
plt.plot(t,plot_l3['Avg'],'navy',lw=0.7,label='Layer 1')
plt.fill_between(t, plot_l3['Min'], plot_l3['Max'],alpha=0.8,color='cornflowerblue')
plt.title(plot_title + '  for layer 3')
plt.yscale('log')
plt.legend()
plt.show()

line_plot_l4 = plt.figure()
plt.plot(t,test_plot[0:int(len(test_plot)/test_samples)],'r',lw=0.7,label='Actual')
plt.plot(t,plot_l4['Avg'],'navy',lw=0.7,label='Layer 1')
plt.fill_between(t, plot_l4['Min'], plot_l4['Max'],alpha=0.8,color='cornflowerblue')
plt.title(plot_title + '  for layer 4')
plt.yscale('log')
plt.legend()
plt.show()

line_plot_nn.savefig("line_figure_normal.1.pdf")
line_plot_l1.savefig("line_figure_l1.1.pdf")
line_plot_l2.savefig("line_figure_l2.1.pdf")
line_plot_l3.savefig("line_figure_l3.1.pdf")
line_plot_l4.savefig("line_figure_l4.1.pdf")

#### Training/Validation loss ####

# =============================================================================
# loss_plot = plt.figure()
# plt.plot(histories[2][1].history['loss'])
# plt.plot(histories[2][1].history['val_loss'])
# plt.title('Model loss')
# plt.yscale('log')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training loss', 'Validation loss'], loc='upper right')
# plt.show()
# loss_plot.savefig("train-val-loss-long.pdf")
# #line_plot.savefig("line_figure_func3.v2.pdf")
# =============================================================================

bar_plot.savefig("bar_figure_func1.pdf")

