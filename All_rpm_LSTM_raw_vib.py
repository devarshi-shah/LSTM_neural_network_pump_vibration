#from keras.layers import LSTM, Input, Reshape
#LSTM()
# code for all rpms models with cross validation
#import numpy as np
import matplotlib.pyplot as plt
#import pandas
import math
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Input, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

## custom initializers: code below defines custom weight initializers for kernal and bias for keras function API used for MC version of training the model
#from keras import backend as K


def prediction_1(filename,input_1,output_1,x_test):
    model_1  = Model(inputs=inputs_1, outputs=output_1)
    model_1.load_weights(filename)
    #x1=model_1.get_weights()
    model_1.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])
    ypred=model_1.predict(x_test)
    return ypred
## initializing constants and data extraction
#path_mat_file='C:\\Users\\dks0013\\Desktop\\Vibration exp-2017\\vibration-log-1\\july_endexp_finaldata\\sample_initial_data\\prelim results'
#path_mat_file='C:\\Users\\dks0013\\Desktop\\Vibration exp-2017\\vibration-log-1\\july_endexp_finaldata\\sample_initial_data\\try_all_rpm_data'
path_mat_file='C:\\Users\\dks0013\\Desktop\\Vibration exp-2017\\vibration-log-1\\july_endexp_finaldata'
#file_name='\\Results_for_2400RPM_allgpm_improved_rpi4_prelim.mat'
#file_name='\\cali_vali_data_all_rpm_var_sel.mat'
file_name='\\Raw_data_for_lstm.mat'

import hdf5storage as hdf
rpm=np.arange(1500,2600,100)
j=0
ypred=[None]*len(rpm)
sypred=[None]*len(rpm)
vloss=[None]*len(rpm)
closs=[None]*len(rpm)
rmse_rpm=[None]*len(rpm)
rmse_flo=[None]*len(rpm)
y_test=[None]*len(rpm)

for i in rpm:
    x_train=hdf.loadmat(path_mat_file+file_name)['x_{}c'.format(i)]
    x_test=hdf.loadmat(path_mat_file+file_name)['x_{}v'.format(i)]
    y_train=hdf.loadmat(path_mat_file+file_name)['y_{}c'.format(i)]
    y_test[j]=hdf.loadmat(path_mat_file+file_name)['y_{}v'.format(i)]
    #del path_mat_file, file_name
    #[ccc]=hdf.loadmat(path_mat_file+file_name,variable_names=['x1800_c','x1800_v','y1800_c','y1800_v'])

    perm=np.arange(0,len(x_train),1)
    random.seed(1000)
    random.shuffle(perm)
    nc=perm[0:int(np.floor(len(perm)*0.8))]
    nv=perm[(int(np.floor(len(perm)*0.8))+1):]
    x_cali=x_train[nc,:]
    x_vali=x_train[nv,:]
    y_cali=y_train[nc,:]
    y_vali=y_train[nv,:]
    
    batch_size=int(len(x_cali)/10)
    epochs=1
    freq_len=(len(x_cali[0])-0)
    
    # Preprocessing of data
    
    mncn_x=StandardScaler(with_std=True)     #only mean centering of data
    x_c_mn=mncn_x.fit_transform(x_cali)
    x_c_mn=x_c_mn[:,:,None]
    x_v_mn=mncn_x.transform(x_vali)
    x_v_mn=x_v_mn[:,:,None]
    #x_c_mn=x_c_mn1[perm,:]
    x_t_mn=mncn_x.transform(x_test)
    x_t_mn=x_t_mn[:,:,None]
    
    mncn_y=StandardScaler(with_std=True)     #only mean centering of data
    colsy=[1]
    y_c_mn=mncn_y.fit_transform(y_cali[:,colsy])
    y_c_mn=y_c_mn[:,:]
    y_v_mn=mncn_y.transform(y_vali[:,colsy])
    y_v_mn=y_v_mn[:,:]
    #y_c_mn=y_c_mn1[perm,:]
    #y_t_mn=mncn_y.transform(y_test)
    
   #Input layer

    inputs_1     = Input(shape=(freq_len,1))
    
    lstm_1       = LSTM(100,return_sequences=True)(inputs_1)
    
    lstm_2       = LSTM(50,return_sequences=False)(lstm_1)
#    lstm_3       = LSTM(50,return_sequences=True)(lstm_2)
#    lstm_4       = LSTM(30,return_sequences=True)(lstm_3)
#    lstm_5       = LSTM(30,return_sequences=False)(lstm_4)
    den3          = Dense(units=50)(lstm_2)
    output_1     = Dense(units = 1)(den3)
    
    model_train  = Model(inputs=inputs_1, outputs=output_1)
    
    addm         =keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
    
    model_train.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])
    model_train.summary()
    # checkpoint
    
    sav_nam='model_raw_vib_bestval_{}rpm1.hdf5'.format(i)
    filepath=sav_nam
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    #model traning
    #history=model_train.fit(x_c_mn,y_c_mn, shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_data=(x_v_mn, y_v_mn),verbose=1)

#getting predictions
#model_1  = Model(inputs=inputs_1, outputs=output_1)
#model_1.load_weights(filepath)
#x1=model_1.get_weights()
#model_1.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])
#pred=model_1.predict(x_t_mn[:,:])
##score=model_1.evaluate(x_t_mn[:,:],y_t_mn[:,:])
#rmse=math.sqrt(mean_squared_error(y_t_mn[:,:],pred))       #manual root mean squared error
#xx=np.zeros((epochs, 2))
#xx[:,1]=history.history['val_loss']      #traning history
#xx[:,0]=history.history['loss']
    ypred[j]=prediction_1(filepath,inputs_1,output_1,x_t_mn)
    sypred[j]=mncn_y.inverse_transform(ypred[j])
#    rmse_rpm[j]=math.sqrt(mean_squared_error(y_test[j][:,0],sypred[j][:,0]))       #manual root mean squared error
    rmse_flo[j]=math.sqrt(mean_squared_error(y_test[j][:,1],sypred[j][:,0]))
    #xx=np.zeros((epochs, 2))
    #vloss[j]=history.history['val_loss']      #traning history
    #closs[j]=history.history['loss']
    j=j+1
    

 #for running plot_model
#model_train.save('model_train_dense.h5')

#from keras.utils import plot_model
#plot_model(model_train,show_shapes=True,to_file='try.png')


## PLotting
##flowrate prediction comparison
#l=11;
#for k in np.arange(0,11):    
#    fl=plt.figure(l)
#    plt.scatter(np.arange(len(sypred[k])),sypred[k][:,0],label='Predicted')
#    plt.scatter(np.arange(len(sypred[k])),y_test[k][:,0],label='Measured')
#    #plt.xlabel('Sample Number',fontsize=15,fontweight='bold')
#    #plt.ylabel('Flowrate',fontsize=15,fontweight='bold')
#    #plt.title('Flowrate prediction',fontsize=17,fontweight='bold')
#    plt.xlabel('Sample Number',fontsize=15,fontweight='bold')
#    plt.ylabel('RPM',fontsize=15,fontweight='bold')
#    plt.title('RPM prediction',fontsize=17,fontweight='bold')
#    plt.legend(fontsize =17,prop=dict(weight='bold'))
#    mng = plt.get_current_fig_manager()
#    mng.window.showMaximized()
#    fl.show()
#    l=l+1
#    #RPM prediction 
#    r=plt.figure(l)
#    plt.scatter(np.arange(len(sypred[k])),sypred[k][:,0],label='Predicted')
#    plt.scatter(np.arange(len(sypred[k])),y_test[k][:,0],label='Measured')
#    plt.xlabel('Sample Number',fontsize=15,fontweight='bold')
#    plt.ylabel('RPM',fontsize=15,fontweight='bold')
#    plt.title('RPM prediction',fontsize=17,fontweight='bold')
#    plt.legend(fontsize =17,prop=dict(weight='bold'))
#    mng = plt.get_current_fig_manager()
#    mng.window.showMaximized()
#    r.show()
#    l=l+1
    #loss comparison
#    ls=plt.figure(k+2)
#    plt.plot(np.arange(epochs),closs[k][:,0],label='Cali_loss',linewidth=2)
#    plt.plot(np.arange(epochs),vloss[k][:,1],label='Vali_loss',linewidth=2)
#    plt.xlabel('Epochs',fontsize=15,fontweight='bold')
#    plt.ylabel('Loss (MSE)',fontsize=15,fontweight='bold')
#    plt.title('RPM prediction',fontsize=17,fontweight='bold')
#    plt.legend(fontsize =17,prop=dict(weight='bold'))
#    mng = plt.get_current_fig_manager()
#    mng.window.showMaximized()
#    ls.show()
    #l=l+1