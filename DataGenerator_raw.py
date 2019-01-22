import numpy as np
import tensorflow.keras as keras

class DataGenerator_raw(keras.utils.Sequence):
    #data generator for keras model using keras utils sequence 
    def __init__(self, file, x_shape, y_shape, batch_size=40, data_type='cali', shuffle=True,meanxx=0,meanxy=0,meanxz=0,meany=0,stdxx=1,stdxy=1,stdxz=1,stdy=1,n_col=[0,1]):
        self.f1=file  #filename
        self.x_shape, self.y_shape = x_shape, y_shape
        self.batch_size = batch_size
        self.mnxx=meanxx
        self.mnxy=meanxy
        self.mnxz=meanxz
        self.mny=meany
        self.stdxx=stdxx
        self.stdxy=stdxy
        self.stdxz=stdxz
        self.stdy=stdy
        self.shuffle=shuffle
        self.dtype=data_type
        self.n_col=n_col           #colums to be inclued for model analysis/ training
        self.on_epoch_end()
        

    def __len__(self):
        #number of batches per epochs
        return int(np.floor((self.x_shape[0]) / float(self.batch_size)))

    def __getitem__(self, idx):
        
        batch_idx=self.batch_idx[idx*self.batch_size:(idx+1)*self.batch_size]
        
        x,y=self.__data_generation(batch_idx)
        #print(x)
#        print('train',idx)
#        print(x.shape)
#        print(x)
        return x,y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print('epoch end')
        self.batch_idx = np.arange(self.x_shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.batch_idx)
    
    def __data_generation(self,batch_idx):
        xx=np.empty((self.batch_size,self.x_shape[1]))
        xy=np.empty((self.batch_size,self.x_shape[1]))
        xz=np.empty((self.batch_size,self.x_shape[1]))
        y=np.empty((self.batch_size,len(self.n_col)))
        #import h5py
        #f1=h5py.File('multi_model_data_cali_vali_test.h5','r')
        if self.dtype=='cali':
            obj_calixx=self.f1['x_calix']
            obj_calixy=self.f1['x_caliy']
            obj_calixz=self.f1['x_caliz']
            obj_caliy=self.f1['y_cali']
            
            for i in range(len(batch_idx)):
                xx[i,:]=obj_calixx[batch_idx[i],:]
                xy[i,:]=obj_calixy[batch_idx[i],:]
                xz[i,:]=obj_calixz[batch_idx[i],:]
                y[i,:]=obj_caliy[batch_idx[i],self.n_col]
                
                
        elif self.dtype=='vali':
            obj_valixx=self.f1['x_valix']
            obj_valixy=self.f1['x_valiy']
            obj_valixz=self.f1['x_valiz']
            obj_valiy=self.f1['y_vali']
            for i in range(len(batch_idx)):
                xx[i,:]=obj_valixx[batch_idx[i],:]
                xy[i,:]=obj_valixy[batch_idx[i],:]
                xz[i,:]=obj_valixz[batch_idx[i],:]
                y[i,:]=obj_valiy[batch_idx[i],self.n_col]
            
        else:
            raise ValueError("Check data type for generatore use either 'cali' or 'vali'. ")
            
        #f1.close()
        xx=(xx-self.mnxx)/self.stdxx
        #print(xx.shape)
        xy=(xx-self.mnxy)/self.stdxy
        #print(xy.shape)
        xz=(xx-self.mnxz)/self.stdxz
        #print(xz.shape)
        
        x=np.array([xx,xy,xz])
      #  print(x.shape)
        x=np.swapaxes(x,1,2)
        x=np.swapaxes(x,0,2)
        #print(x.shape)
        #x=(x-self.mnx)/self.stdx
        y=(y-self.mny)/self.stdy
        #print(y)
        return x ,y
            
    
