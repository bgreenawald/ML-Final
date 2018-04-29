
# coding: utf-8

# # Weather Prediction Using Recurrent Neural Networks
# 
# ## Adrian, Ben, and Sai

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import timeit
import random

# # Preprocessing

# ### Read in the files

# In[2]:


full_df = pd.read_csv("df_weather_scaled_encoded.csv")


# In[3]:


# Filter by the city of interest
current_city = "Charlotte"

full_df = full_df[full_df["city"] == current_city]

min_dataset = 0.54
max_dataset = 99.95

# In[4]:


years = np.array([y[0:4] for y in full_df.datetime])

train = full_df[years < '2016']
valid = full_df[years == '2016']
test = full_df[years > '2016']

if(train.shape[0] + valid.shape[0] + test.shape[0] != years.shape[0]):
    raise Exception("Partition did not work")
    
# Drop the city and timestamp for all three
train.drop(["city", "datetime"], inplace=True, axis=1)
valid.drop(["city", "datetime"], inplace=True, axis=1)
test.drop(["city", "datetime"], inplace=True, axis=1)


# In[ ]:


class DataSet(object):
    def __init__(self, x, y):
        self._num_examples = len(x)
        self._x = x
        self._y = y
        self._epochs_done = 0
        self._index_in_epoch = 0
        np.random.seed(123456)
        # Shuffle the data
        perm = np.arange(self._num_examples)
        print(perm)
        np.random.shuffle(perm)
        self._x = [self._x[i] for i in perm]
        self._y = [self._y[i] for i in perm]
        random.seed(123456)
    @property
    def features(self):
        return self._x
    @property
    def response(self):
        return self._y
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_done(self):
        return self._epochs_done

    def reset_batch_index(self):
        self._index_in_epoch = 0
    
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        done = False

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._x = self._x
            self._y = self._y 
            start = 0
            self._index_in_epoch = batch_size
            done = True
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
    
        return self._x[start:end], self._y[start:end], done
    
# In[ ]: 

# Start on the second problem
seq_len = 24

train_x = [] 
train_y = []
baseline_err = 0

for i in range(train.shape[0] - seq_len):
    train_x.append([x for x in train.iloc[i:i+seq_len, :].values])
    train_y.append([y for y in train.iloc[i+1:i+seq_len+1, -7:].values])
    

train_data2 = DataSet(train_x, train_y)
del train_x, train_y


# In[11]:

valid_x = [] 
valid_y = []

for i in range(valid.shape[0] - seq_len):
    valid_x.append([x for x in valid.iloc[i:i+seq_len, :].values])
    valid_y.append([y for y in valid.iloc[i+1:i+seq_len+1, -7:].values])

valid_data2 = DataSet(valid_x, valid_y)
del valid_x, valid_y

# In[ ]:

test_x = [] 
test_y = []
baseline_accuracy = []
for i in range(test.shape[0] - seq_len):
    test_x.append([x for x in test.iloc[i:i+seq_len, :].values])
    test_y.append([y for y in test.iloc[i+1:i+seq_len+1, -7:].values])
    baseline_accuracy.append(test.iloc[i:i+seq_len, -7:].values == test.iloc[i+1:i+seq_len+1, -7:].values)
test_data2 = DataSet(test_x, test_y)
del test_x, test_y

# Test baseline error of 0.9154835105079804


# In[ ]:

def build_and_save_d2(modelDir,train,valid,cell,cellType,input_dim=1,hidden_dim=100,
                          seq_size = 12,max_itr=200,keep_prob=0.5, batch_size=32, num_epochs=10,log=500,save=1000):
    graph2 = tf.Graph()
    with graph2.as_default():
        # input place holders
        # input Shape: [# training examples, sequence length, # features]
        x = tf.placeholder(tf.float32,[None,seq_size,input_dim],name="x_in")
        # label Shape: [# training examples, sequence length, # classes]
        y = tf.placeholder(tf.float32,[None,seq_size,7],name="y_in")
        dropout = tf.placeholder(tf.float32,name="dropout_in")
        
        # cell = tf.contrib.rnn.MultiRNNCell([cell, cell])
        cell = tf.nn.rnn_cell.DropoutWrapper(cell)
        # RNN output Shape: [# training examples, sequence length, # hidden] 
        outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
        
        
        # weights for output dense layer (i.e., after RNN)
        # W shape: [# hidden, 1]
        W_out = tf.Variable(tf.random_normal([hidden_dim,7]),name="w_out") 
        # b shape: [1]
        b_out = tf.Variable(tf.random_normal([7]),name="b_out")
    
        # output dense layer:
        num_examples = tf.shape(x)[0] 
        # convert W from [# hidden, 1] to [# training examples, # hidden, 1]
        # step 1: add a new dimension at index 0 using tf.expand_dims
        w_exp= tf.expand_dims(W_out,0)
        # step 2: duplicate W for 'num_examples' times using tf.tile
        W_repeated = tf.tile(w_exp,[num_examples,1,1])
        
        # Dense Layer calculation: 
        # [# training examples, sequence length, # hidden] *
        # [# training examples, # hidden, 1] = [# training examples, sequence length]
        
        y_pred = tf.matmul(outputs,W_repeated)
        y_pred = tf.add(y_pred, b_out, name="y_out")
        # Actually, y_pred: [# training examples, sequence length, 1]
        # Remove last dimension using tf.squeeze
        # y_pred = tf.squeeze(y_pred,name="y_pred")
        
        # Cost & Training Step
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y))
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        saver=tf.train.Saver()
        
        # Run Session
    with tf.Session(graph=graph2) as sess2:
        # initialize variables
        sess2.run(tf.global_variables_initializer())
        # Run for 1000 iterations (1000 is arbitrary, need a validation set to tune!)
        start=timeit.default_timer()
        epoch_counter = 0 # Keep track of our epochs
        i = 0 # Keep track of our iterations
        
        print('Training %s ...'%cellType)
        while True: # If we train more, would we overfit? Try 10000
            i += 1
            trainX, trainY, done = train.next_batch(batch_size)
            # See if we are done with our epochs
            if done:
                epoch_counter += 1
                print("Done with epoch " + str(epoch_counter))
                if epoch_counter + 1 > num_epochs:
                    break
                
            _, train_err = sess2.run([train_op,cost],feed_dict={x:trainX,y:trainY,dropout:keep_prob})
            if i==0:
                print('  step, train err= %6d: %8.5f' % (0,train_err)) 
            elif  (i+1) % log == 0: 
                print('  step, train err= %6d: %8.5f' % (i+1,train_err)) 
                
                # Get validation error
                valid_err = sess2.run(cost,feed_dict={x:valid.features,y:valid.response,dropout:1})
                print('  step, validation err= %6d: %8.5f' % (i+1,valid_err)) 
            if i>0 and (i+1) % save == 0:    
                modelPath= saver.save(sess2,"%s/model_%s"%(modelDir,cellType),global_step=i+1)
                print("model saved:%s"%modelPath)    
        
        # Save model at the end
        modelPath= saver.save(sess2,"%s/modelF_%s"%(modelDir,cellType),global_step=i+1)
        print("model saved:%s"%modelPath)    
        end=timeit.default_timer()        
        print("Training time : %10.5f"%(end-start))
       
    return i + 1

input_dim=19 # dim > 1 for multivariate time series
hidden_dim=100 # number of hiddent units h
max_itr=2000 # number of training iterations
keep_prob=0.5
modelDir='modelDir2'
log=100
save=5000
batch_size=16
num_epochs=1

# Different RNN Cell Types
RNNcell = rnn.BasicRNNCell(hidden_dim)
cellType = "RNN"

# Build models and save model
end_itr = build_and_save_d2(modelDir=modelDir,
                 train=train_data2,
                 valid=valid_data2,
                 cell=RNNcell,
                 cellType=cellType,
                 input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 seq_size=seq_len,
                 max_itr=max_itr,
                 keep_prob=keep_prob,
                 batch_size=batch_size,
                 num_epochs=num_epochs,
                 log=log,
                 save=save)


# In[ ]:
def load_and_predict2(test,modelDir,cellType,itr,seq_size):
    with tf.Session() as sess2:
        print ("Load model:%s-%s"%(modelDir,itr))
        saver = tf.train.import_meta_graph("%s/modelF_%s-%s.meta"%(modelDir,cellType,itr))
        saver.restore(sess2,tf.train.latest_checkpoint("%s"%modelDir))
        graph2 = tf.get_default_graph()
        
        x2 = graph2.get_tensor_by_name("x_in:0")
        dropout2= graph2.get_tensor_by_name("dropout_in:0")
        y_pred_2 = graph2.get_tensor_by_name("y_out:0")
        
        predicted_vals_all= sess2.run(y_pred_2, feed_dict={ x2: test.features, dropout2:1})
        # Get last item in each predicted sequence:
        predicted_vals = predicted_vals_all[:,seq_size-1]
    return predicted_vals
        
# Load and predict
predicted_vals_rnn=load_and_predict2(test_data2,modelDir,cellType,end_itr,seq_len)

print(predicted_vals_rnn)
# Compute MSE
# step 2: get ground-truth
actual_test= np.array([x[-1] for x in test_data2.response])

# In[ ]:

actual_test= np.array([x[-1] for x in test_data2.response])

sum(np.argmax(actual_test, axis=1) == np.argmax(predicted_vals_rnn, axis=1))/len(actual_test)