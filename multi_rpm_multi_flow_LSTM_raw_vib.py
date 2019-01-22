##from keras.layers import LSTM, Input, Reshape
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
import hdf5storage as hdf
import h5py
from DataGenerator_raw import DataGenerator_raw

import time



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



def mean_std_part(filename,var_string,no_of_var,ch_size=500,var_in=0):
    #no_of_var=integer number of variables whose mean and std needs to be calculated
    f1=h5py.File(filename,'r')
    xsh=f1[var_string].shape
   
    #ch_size=200     #number of samples of data to be loaded at one time
    sx=np.zeros((ch_size,no_of_var))       #initialize sum of x
    sxsq=np.zeros((ch_size,no_of_var))#initialize sum of x^2 used later to find std.
     
    temp_len=np.arange(0,xsh[0],ch_size)
    temp_len=np.append(temp_len,xsh[0])
    if var_in==0:
        var_in1=np.arange(0,xsh[1],1)
    else:
        var_in1=var_in
    
    for i in range(len(temp_len)-1):
        xtemp=f1[var_string][temp_len[i]:temp_len[i+1],var_in1]
        sht=xtemp.shape
        if sht[0]<(ch_size-1):
            zzx=np.zeros((ch_size-sht[0],no_of_var))
            xtemp=np.concatenate((xtemp,zzx))
            
        sx=np.add(sx,xtemp)
        ex_sq=np.square(xtemp)
        sxsq=np.add(sxsq,ex_sq)
        
    
    f1.close()  
    #mean calculation    
    mnx=np.sum(sx,axis=0)/(xsh[0]-1)        #mean of dataset
    
    ##Std calculation
    stdx=np.sqrt(np.subtract((np.sum(sxsq,axis=0)/(xsh[0]-1)),np.square(mnx)))
    return mnx, stdx, xsh

file_name='\\Raw_data_for_lstm_xyz_evi_out_rem_allsamples_rand_for_python.mat'

mncyd,stdcyd,yshc=mean_std_part(path_mat_file+file_name,'y_cali',2,var_in=[0,1])
#mnvyd,stdvyd,yshv=mean_std_part(path_mat_file+file_name,'y_vali',2,var_in=[0,1])
#
#
mncx,stdcx,xshcx=mean_std_part(path_mat_file+file_name,'x_calix',800)
mncy,stdcy,xshcy=mean_std_part(path_mat_file+file_name,'x_caliy',800)
mncz,stdcz,xshcz=mean_std_part(path_mat_file+file_name,'x_caliz',800)

#
#mnvx,stdvx,xshvx=mean_std_part(path_mat_file+file_name,'x_valix',800)
#mnvy,stdvy,xshvy=mean_std_part(path_mat_file+file_name,'x_valiy',800)
#mnvz,stdvz,xshvz=mean_std_part(path_mat_file+file_name,'x_valiz',800)






batch_size=int(xshcx[0]/60)
epochs=100
freq_len=(xshcx[1])

f1=h5py.File(path_mat_file+file_name,'r')

training_gen=DataGenerator_raw(f1,xshcx,yshc,data_type='cali',batch_size=batch_size, shuffle=True,meanxx=mncx,meanxy=mncy,meanxz=mncz,meany=mncyd,stdxx=stdcx,stdxy=stdcy,stdxz=stdcz,stdy=stdcyd,n_col=[0,1])
xshvx=f1['x_valix'].shape
yshv=f1['y_vali'].shape
validation_gen=DataGenerator_raw(f1,xshvx,yshv,data_type='vali',batch_size=batch_size, shuffle=True,meanxx=mncx,meanxy=mncy,meanxz=mncz,meany=mncyd,stdxx=stdcx,stdxy=stdcy,stdxz=stdcz,stdy=stdcyd,n_col=[0,1])


## no generator
#
#x_calix=f1['x_calix']
#x_calix=(x_calix-mncx)/stdcx
#
#x_caliy=f1['x_caliy']
#x_caliy=(x_caliy-mncy)/stdcy
#
#x_caliz=f1['x_caliz']
#x_caliz=(x_caliz-mncz)/stdcz
#
#y_cali=f1['y_cali'][:,0:2]
#x_cali=np.array([x_calix,x_caliy,x_caliz])
#del x_calix,x_caliy,x_caliz
#x_cali=np.swapaxes(x_cali,1,2)
#x_cali=np.swapaxes(x_cali,0,2)
#y_cali=(y_cali-mncyd)/stdcyd
##vaidation
#x_valix=f1['x_valix']
#x_valix=(x_valix-mncx)/stdcx
#
#x_valiy=f1['x_valiy']
#x_valiy=(x_valiy-mncy)/stdcy
#
#x_valiz=f1['x_valiz']
#x_valiz=(x_valiz-mncz)/stdcz
#
#y_vali=f1['y_vali'][:,0:2]
#x_vali=np.array([x_valix,x_valiy,x_valiz])
#del x_valix,x_valiy,x_valiz
#x_vali=np.swapaxes(x_vali,1,2)
#x_vali=np.swapaxes(x_vali,0,2)
#y_vali=(y_vali-mncyd)/stdcyd

   #Input layer

inputs_1     = Input(shape=(freq_len,3))

lstm_1       = LSTM(200,return_sequences=True)(inputs_1)

lstm_2       = LSTM(100,return_sequences=False)(lstm_1)
#    lstm_3       = LSTM(50,return_sequences=True)(lstm_2)
#    lstm_4       = LSTM(30,return_sequences=True)(lstm_3)
#    lstm_5       = LSTM(30,return_sequences=False)(lstm_4)
den3          = Dense(units=100)(lstm_2)
output_1     = Dense(units = 2)(den3)

model_train  = Model(inputs=inputs_1, outputs=output_1)

addm         =keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)

model_train.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])
model_train.summary()
# checkpoint

sav_nam='model_bestval_all_rpm_models_raw_trial_200.hdf5'
filepath=sav_nam
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#model traning
#history=model_train.fit(x_c_mn,y_c_mn, shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_data=(x_v_mn, y_v_mn),verbose=1)
start_time=time.time()

history=model_train.fit_generator(generator=training_gen, shuffle=True, epochs=epochs, callbacks=callbacks_list, validation_data=validation_gen,verbose=1,max_queue_size=3)

#history=model_train.fit(x_cali,y_cali, shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_data=(x_vali, y_vali),verbose=1)



#getting predictions
model_1  = Model(inputs=inputs_1, outputs=output_1)
model_1.load_weights(filepath)
#m1=keras.models.load_model(filepath)
x1=model_1.get_weights()
model_1.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])

# validation data
x_valix=f1['x_valix']
x_valix=(x_valix-mncx)/stdcx

x_valiy=f1['x_valiy']
x_valiy=(x_valiy-mncy)/stdcy

x_valiz=f1['x_testz']
x_valiz=(x_valiz-mncz)/stdcz

y_vali=f1['y_vali'][:,:]

x_vali=np.array([x_valix,x_valiy,x_valiz])
x_vali=np.swapaxes(x_vali,1,2)
x_vali=np.swapaxes(x_vali,0,2)
f1.close()

# test data
#x_testx=f1['x_testx']
#x_testx=(x_testx-mncx)/stdcx
#
#x_testy=f1['x_testy']
#x_testy=(x_testy-mncy)/stdcy
#
#x_testz=f1['x_testz']
#x_testz=(x_testz-mncz)/stdcz
#
#y_test=f1['y_test'][:,:]
#
#x_test=np.array([x_testx,x_testy,x_testz])
#x_test=np.swapaxes(x_test,1,2)
#x_test=np.swapaxes(x_test,0,2)
#f1.close()


pred=model_1.predict(x_vali)
#pred=model_1.predict(x_test)


#score=model_1.evaluate(x_t_mn[:,:],y_t_mn[:,:])
#rmse=math.sqrt(mean_squared_error(y_t_mn[:,:],pred))       #manual root mean squared error
#xx=np.zeros((epochs, 2))
#xx[:,1]=history.history['val_loss']      #traning history
#xx[:,0]=history.history['loss']
ypred=prediction_1(filepath,inputs_1,output_1,x_test)
sypred=(ypred*stdcyd)+mncyd
#    rmse_rpm[j]=math.sqrt(mean_squared_error(y_test[j][:,0],sypred[j][:,0]))       #manual root mean squared error
rmse_flo=math.sqrt(mean_squared_error(y_test[:,1],sypred[:,1]))
    #xx=np.zeros((epochs, 2))
    #vloss[j]=history.history['val_loss']      #traning history
    #closs[j]=history.history['loss']
    
end_time=time.time()

run_time=start_time-end_time    

 #for running plot_model
#model_train.save('model_train_dense.h5')

#from keras.utils import plot_model
#plot_model(model_train,show_shapes=True,to_file='try.png')


## PLotting
#flowrate prediction comparison
l=11;

fl=plt.figure()
plt.scatter(np.arange(len(sypred)),sypred[:,0],label='Predicted')
plt.scatter(np.arange(len(sypred)),y_test[:,0],label='Measured')
#plt.xlabel('Sample Number',fontsize=15,fontweight='bold')
#plt.ylabel('Flowrate',fontsize=15,fontweight='bold')
#plt.title('Flowrate prediction',fontsize=17,fontweight='bold')
plt.xlabel('Sample Number',fontsize=15,fontweight='bold')
plt.ylabel('RPM',fontsize=15,fontweight='bold')
plt.title('RPM prediction',fontsize=17,fontweight='bold')
plt.legend(fontsize =17,prop=dict(weight='bold'))
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
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