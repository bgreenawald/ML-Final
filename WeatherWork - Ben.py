
# coding: utf-8

# # Weather Prediction Using Recurrent Neural Networks
# 
# ## Adrian, Ben, and Sai

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from functools import reduce
import datetime
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.preprocessing import MinMaxScaler
import timeit
import random
import sys
# # Preprocessing

###########################################################################################################################
############# Preprocessing ##############################################################################################
###########################################################################################################################

# ### Read in the files

# In[2]:


# Filenames
city_file = 'city_attributes.csv'
temp_file = 'temperature.csv'
humid_file = 'humidity.csv'
press_file = 'pressure.csv'
desc_file = 'weather_description.csv'
wdir_file = 'wind_direction.csv'
wspeed_file = 'wind_speed.csv'
# Load the files
city_df = pd.read_csv(city_file)
city_df.rename(str.lower, axis = 'columns', inplace = True)
city_df.drop(['country'], axis = 1, inplace = True)
city_df.set_index(['city'], inplace = True)
temp_df = pd.read_csv(temp_file)
humid_df = pd.read_csv(humid_file)
press_df = pd.read_csv(press_file)
desc_df = pd.read_csv(desc_file)
wdir_df = pd.read_csv(wdir_file)
wspeed_df = pd.read_csv(wspeed_file)


# In[3]:


# These are the cities that universally have > 1% missing across all weather values
drop_city = set(temp_df.columns[temp_df.isna().sum() > 500]) & set(humid_df.columns[humid_df.isna().sum() > 500]) & set(press_df.columns[press_df.isna().sum() > 500]) & set(desc_df.columns[desc_df.isna().sum() > 500]) & set(wdir_df.columns[wdir_df.isna().sum() > 500]) & set(wspeed_df.columns[wspeed_df.isna().sum() > 500])  


# In[4]:


# Remove the undesired cities and melt the tables to be conducive for joining
alt_temp_df = pd.melt(temp_df.drop(drop_city, axis = 1), id_vars = ['datetime'], var_name = 'city', value_name = 'temperature')
alt_humid_df = pd.melt(humid_df.drop(drop_city, axis = 1), id_vars = ['datetime'], var_name = 'city', value_name = 'humidity')
alt_press_df = pd.melt(press_df.drop(drop_city, axis = 1), id_vars = ['datetime'], var_name = 'city', value_name = 'pressure')
alt_desc_df = pd.melt(desc_df.drop(drop_city, axis = 1), id_vars = ['datetime'], var_name = 'city', value_name = 'weather_description')
alt_wdir_df = pd.melt(wdir_df.drop(drop_city, axis = 1), id_vars = ['datetime'], var_name = 'city', value_name = 'wind_direction')
alt_wspeed_df = pd.melt(wspeed_df.drop(drop_city, axis = 1), id_vars = ['datetime'], var_name = 'city', value_name = 'wind_speed')

# Set proper indices
alt_temp_df = alt_temp_df.set_index(['city', 'datetime'])
alt_humid_df = alt_humid_df.set_index(['city', 'datetime'])
alt_press_df = alt_press_df.set_index(['city', 'datetime'])
alt_desc_df = alt_desc_df.set_index(['city', 'datetime'])
alt_wdir_df = alt_wdir_df.set_index(['city', 'datetime'])
alt_wspeed_df = alt_wspeed_df.set_index(['city', 'datetime'])


# ### Join tables together

# In[5]:


# Join tables on the city and datetime info
dfs = [city_df, alt_temp_df, alt_humid_df, alt_press_df, alt_wspeed_df, alt_wdir_df, alt_desc_df]
df_final = reduce(lambda left, right : pd.merge(left, right, left_index = True, right_index = True), dfs)


# ### Deal with Missing Values

# In[6]:


# INTERPOLATION HAPPENS HERE -- Break up by city
df_final = df_final.groupby('city').apply(lambda group: group.interpolate(limit_direction = 'both'))

# Need to do something special for weather_description
arr, cat = df_final['weather_description'].factorize()
df_final['weather_description'] = pd.Series(arr).replace(-1, np.nan).interpolate(method = 'nearest', limit_direction = 'both').interpolate(limit_direction = 'both').astype('category').cat.rename_categories(cat).astype('str').values


# In[7]:


# The whole purpose here is to encode wind direction. It's not continuous so don't really want to scale it
# Also have more granularity in wind dir if need be.
#dir_df = pd.DataFrame({'dir' : ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N'],
#                        'lower' : [348.75, 11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75, 191.25, 213.75, 236.25, 258.75, 281.25, 303.75, 326.25, 0],
#                        'upper' : [360, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75, 191.25, 213.75, 236.25, 258.75, 281.25, 303.75, 326.25, 348.75, 11.25]})
dir_df = pd.DataFrame({'dir' : ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'],
                        'lower' : [337.5, 22.5, 67.5, 112.5, 157, 202.5, 247.5, 292.5, 0],
                        'upper' : [360, 67.5, 112.5, 157, 202.5, 247.5, 292.5, 337.5, 22.5]})
# Make a copy to fool around in
fill_this = df_final['wind_direction'].copy()
# And overwrite the copy
for i in reversed(range(len(dir_df))):
#    print(str(dir_df.loc[i,'lower']) + " and " + str(dir_df.loc[i,'upper']))
    fill_this.loc[df_final['wind_direction'].between(dir_df.loc[i,'lower'], dir_df.loc[i,'upper'])] = i
# This is a bit ugly here; but it maintains any missing values nicely
df_final['wind_direction'] = dir_df.loc[fill_this, 'dir'].values


# In[8]:


# Go ahead and drop lat and long, we wont need them for now
df_final.drop(["latitude", "longitude"], inplace=True, axis=1)


# In[12]:


# Convert the data to Farenheit and note the min and max values
df_final["temperature"] = df_final["temperature"] * 9/5 - 459.67

# ### Normalize data through min-max scaling

# In[13]:


# Scaling happens here -- IMPUTATION MUST HAPPEN FIRST
scale_df = df_final[['temperature', 'humidity', 'pressure', 'wind_speed']].values
scaler = MinMaxScaler()
# We have access to min and max so we can transform back and forth
scale_df = scaler.fit_transform(scale_df)
df_final_scaled = df_final.copy()
df_final_scaled[['temperature', 'humidity', 'pressure', 'wind_speed']] = scale_df
df_final_scaled.head()


# In[14]:


# Collapse a lot of these groupings
weather_dict = {'scattered clouds' : 'partly_cloudy', 'sky is clear' : 'clear', 
             'few clouds' : 'partly_cloudy', 'broken clouds' : 'partly_cloudy',
           'overcast clouds' : 'cloudy', 'mist' : 'cloudy', 'haze' : 'cloudy', 
             'dust' : 'other', 'fog' : 'cloudy', 'moderate rain' : 'rain',
           'light rain' : 'rain', 'heavy intensity rain' : 'rain', 'light intensity drizzle' : 'rain',
           'heavy snow' : 'snow', 'snow' : 'snow', 'light snow' : 'snow', 'very heavy rain' : 'rain',
           'thunderstorm' : 'tstorm', 'proximity thunderstorm' : 'tstorm', 'smoke' : 'other', 'freezing rain' : 'snow',
           'thunderstorm with light rain' : 'tstorm', 'drizzle' : 'rain', 'sleet' : 'snow',
           'thunderstorm with rain' : 'tstorm', 'thunderstorm with heavy rain' : 'tstorm',
           'squalls' : 'rain', 'heavy intensity drizzle' : 'rain', 'light shower snow' : 'snow',
           'light intensity shower rain' : 'rain', 'shower rain' : 'rain',
           'heavy intensity shower rain' : 'rain', 'proximity shower rain' : 'rain',
           'proximity sand/dust whirls' : 'other', 'proximity moderate rain' : 'rain', 'sand' : 'other',
           'shower snow' : 'snow', 'proximity thunderstorm with rain' : 'tstorm',
           'sand/dust whirls' : 'other', 'proximity thunderstorm with drizzle' : 'tstorm',
           'thunderstorm with drizzle' : 'tstorm', 'thunderstorm with light drizzle' : 'tstorm',
           'light rain and snow' : 'snow', 'thunderstorm with heavy drizzle' : 'tstorm',
           'ragged thunderstorm' : 'tstorm', 'tornado' : 'other', 'volcanic ash' : 'other', 'shower drizzle' : 'rain',
           'heavy shower snow' : 'snow', 'light intensity drizzle rain' : 'rain',
           'light shower sleet' : 'snow', 'rain and snow' : 'snow'}


# In[15]:


adj_weather = [weather_dict[val] for val in df_final_scaled['weather_description']]
df_final_scaled['adj_weather'] = adj_weather
df_final_scaled = df_final_scaled.drop('weather_description', axis = 1)


# ### Make weather and wind direction dummy variables

# In[16]:


# And one-hot encode the wind_directions, NOT weather description since it is the response
df_final_scaled = pd.get_dummies(df_final_scaled, prefix=['wind_dir', 'weather'], 
                                 columns=['wind_direction', 'adj_weather'])


# In[17]:


df_final_scaled = df_final_scaled.reset_index('city')


# In[18]:


# Write for distribution
df_final_scaled.to_csv('df_weather_scaled_encoded.csv')


# In[124]:


# Clean up the local environment
get_ipython().run_line_magic('reset', '')

###########################################################################################################################
############# Part 1: Temperature Prediction ##############################################################################
###########################################################################################################################
# In[2]:

# ## Split into train, test, and validation


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import timeit
import sys

# In[2]:


full_df = pd.read_csv("df_weather_scaled_encoded.csv")


# In[3]:


# Filter by the city of interest
current_city = "Charlotte"

full_df = full_df[full_df["city"] == current_city]

min_dataset = 0.5149999994000041
max_dataset = 99.95

# In[4]:

# Extract
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

# Wrapper for data object
# Modified from Mohammad al Boni

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


# ## Create baselines

# ### Create observations using a sliding sequence window

# In[26]:

# Define size of sequence, 1 day for now
seq_len = 24

# Wrapper function to perform the entire creation of observations given the subset
# data. Can specify sequence_size, lookahead, response (temp means 'temperature'),
# and whether you want a greedy baseline.
def create_observations(train, test, valid, seq_size = 24, lookahead = 1, temp = True, baseline=False):

    train_x = [] 
    train_y = []
    # If we are doing the temperature variable, extract that feature
    if temp: 
        for i in range(train.shape[0] - seq_len - lookahead + 1):
            # Slide over input, storing each "sequence size" window
            train_x.append([x for x in train.iloc[i:i+seq_len, :].values])
            train_y.append([y for y in train.iloc[i+lookahead:i+seq_len+lookahead, 0]])
    # Otherwise, extract out the weather type
    else:
        for i in range(train.shape[0] - seq_len  - lookahead + 1):
            train_x.append([x for x in train.iloc[i:i+seq_len, :].values])
            train_y.append([y for y in train.iloc[i+lookahead:i+seq_len+lookahead, -7:].values])
    
    # Convert to a Dataset object
    train_data = DataSet(train_x, train_y)
    
    # Repeat the above process on the validation set
    valid_x = [] 
    valid_y = []
    
    # If we are doing the temperature variable, extract that feature
    if temp: 
        for i in range(valid.shape[0] - seq_len - lookahead + 1):
            # Slide over input, storing each "sequence size" window
            valid_x.append([x for x in valid.iloc[i:i+seq_len, :].values])
            valid_y.append([y for y in valid.iloc[i+lookahead:i+seq_len+lookahead, 0]])
    # Otherwise, extract out the weather type
    else:
        for i in range(valid.shape[0] - seq_len - lookahead + 1):
            valid_x.append([x for x in valid.iloc[i:i+seq_len, :].values])
            valid_y.append([y for y in valid.iloc[i+lookahead:i+seq_len+lookahead, -7:].values])
    
    valid_data = DataSet(valid_x, valid_y)


    # Repeat for test except also track the baseline prediction error
    test_x = [] 
    test_y = []
    test_baseline_err = []
    
    if temp:
        for i in range(test.shape[0] - seq_len - lookahead + 1):
            test_x.append([x for x in test.iloc[i:i+seq_len, :].values])
            test_y.append([y for y in test.iloc[i+lookahead:i+seq_len+lookahead, 0]])
            
            # Get the baseline prediction error by taking the MSE between the current hour and the 
            # temperature of the next hour. This is the trivial case where our prediction for temp
            # is just the current temp
            if baseline:
                test_baseline_err.append((np.mean(train.iloc[i+seq_len - 1, 0]*(max_dataset-min_dataset)+min_dataset) - 
                                      (train.iloc[i+seq_len + lookahead - 1, 0]*(max_dataset-min_dataset)+min_dataset)) ** 2)
        
        if baseline:
            print("Baseline error of: " + str(np.mean(test_baseline_err)))
        # Test baseline error of 2.7059362148173807
    else:
        for i in range(test.shape[0] - seq_len - lookahead + 1):
            test_x.append([x for x in test.iloc[i:i+seq_len, :].values])
            test_y.append([y for y in test.iloc[i+lookahead:i+seq_len+lookahead, 0]])
        
            if baseline:        
                # Compare current weather type with next weather type
                test_baseline_err.append(test.iloc[i + seq_len - 1, -7:].values == test.iloc[i + seq_len + lookahead - 1, -7:].values)
                
        if baseline:
            print("Baseline error of: " + str(np.mean(test_baseline_err)))
        # Test baseline error of 2.7059362148173807
    
    test_data = DataSet(test_x, test_y)
    
    return train_data, valid_data, test_data


train_data,valid_data,test_data = create_observations(train, valid, test, seq_size=seq_len, lookahead=4, baseline=True)

# In[11]:

# NEED TO
#   Perform imputation on missing values -- Probably by city and day -- DONE
#   Join the tables -- DONE
#   Do min-max scaling -- DONE
#   Roll up the values to the daily level -- NOT DOING (this isn't what we were planning on doing in our proposal)
#   Encode the weather_description and wind direction as a one-hot -- DONE
#   Get the wind direction as a categorical -- DONE

# Pretty good. Have some more to do now
#   Separate into training, testing, and validation --DONE
#   Fully break up the data into the Xtrain, Xtest, Xvalid, Ytrain, Ytest, and Yvalid



# In[ ]:
        
def build_and_save_d(modelDir,train,valid,cell,cellType,input_dim=1,hidden_dim=100,
                          seq_size = 12,max_itr=200,keep_prob=0.5, batch_size=32, num_epochs=10, log=500,
                            early_stopping=3, learning_rate=0.01):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        # input place holders
        # input Shape: [# training examples, sequence length, # features]
        x = tf.placeholder(tf.float32,[None,seq_size,input_dim],name="x_in")
        # label Shape: [# training examples, sequence length]
        y = tf.placeholder(tf.float32,[None,seq_size],name="y_in")
        dropout = tf.placeholder(tf.float32,name="dropout_in")
        
        # Function to wrap each cell with dropout
        def wrap_cell(cell, keep_prob):
            drop = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return drop
        
        cells = tf.nn.rnn_cell.MultiRNNCell(
                 [wrap_cell(cell,keep_prob) for cell in cell]
         )

        # cell = tf.nn.rnn_cell.DropoutWrapper(cell)
        # RNN output Shape: [# training examples, sequence length, # hidden] 
        outputs, _ = tf.nn.dynamic_rnn(cells,x,dtype=tf.float32)
        
        
        # weights for output dense layer (i.e., after RNN)
        # W shape: [# hidden, 1]
        W_out = tf.Variable(tf.random_normal([hidden_dim,1]),name="w_out") 
        # b shape: [1]
        b_out = tf.Variable(tf.random_normal([1]),name="b_out")
    
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
        
        y_pred = tf.matmul(outputs,W_repeated)+b_out
        # Actually, y_pred: [# training examples, sequence length, 1]
        # Remove last dimension using tf.squeeze
        y_pred = tf.squeeze(y_pred,name="y_pred")
        
        # Cost & Training Step
        cost = tf.reduce_mean(tf.square(y_pred-y))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        saver=tf.train.Saver()
        
        
        
        # Run Session
    with tf.Session(graph=graph) as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # Run for 1000 iterations (1000 is arbitrary, need a validation set to tune!)
        start=timeit.default_timer()
        epoch_counter = 0 # Keep track of our epochs
        i = 0 # Keep track of our iterations
        min_validation_err = sys.float_info.max # Start min error at biggest number
        min_validation_itr = 0 # Keep track of the smallest validation error we have seen so far
        early_stopping_counter = 0 # Counter to see if we have acheived early stopping
        min_denorm_err = None
        print('Training %s ...'%cellType)
        while True: # If we train more, would we overfit? Try 10000
            i += 1 # Increment counter
            trainX, trainY, done = train.next_batch(batch_size) # Get train batch
            # See if we are done with our epochs
            if done:
                epoch_counter += 1
                print("Done with epoch " + str(epoch_counter))
                if epoch_counter >= num_epochs:
                    break
            
            # Pass the data through the network
            _, train_err = sess.run([train_op,cost],feed_dict={x:trainX,y:trainY,dropout:keep_prob})
            if i==0:
                print('  step, train err= %6d: %8.5f' % (0,train_err))
            # Every 'log' steps, print out train error and validation error.
            # Update early stopping at these points
            elif  (i+1) % log == 0: 
                print('  step, train err= %6d: %8.5f' % (i+1,train_err)) 
                
                # Get validation error on the full validation set
                valid_err, predicted_vals_rnn = sess.run([cost, y_pred],feed_dict={x:valid.features,y:valid.response,dropout:1})
                
                # Compute denormalized MSE
                # step 1: denormalize data
                # If seq_len greater than 1, get only the last element
                if seq_size > 1:
                    predicted_vals_rnn = predicted_vals_rnn[:,seq_size-1]
                predicted_vals_dnorm_rnn=predicted_vals_rnn*(max_dataset-min_dataset)+min_dataset
                # step 2: get ground-truth, also must be denormalized
                actual_test= np.array([x[-1] for x in valid.response])*(max_dataset-min_dataset)+min_dataset
                # step 3: compute MSE
                mse_rnn= ((predicted_vals_dnorm_rnn - actual_test) ** 2).mean()
                print('  step, validation err= %6d: %8.5f' % (i+1,valid_err)) 
                print('  step, denorm validation err= %6d: %8.5f' % (i+1,mse_rnn)) 
                
                # Check early stopping
                early_stopping_counter += 1
                # If we have smaller validation error, reset counter and
                # assign new smallest validation error. Also store the
                # current iterqtion as the iteration where the current min is
                if valid_err < min_validation_err:
                    min_validation_err = valid_err
                    min_validation_itr = i + 1
                    early_stopping_counter = 0
                    min_denorm_err = mse_rnn
                    
                    # Store the current best model
                    modelPath= saver.save(sess,"%s/model_%s"%(modelDir,cellType),global_step=i+1)
                    print("model saved:%s"%modelPath) 
                
                # Break if we achieve early stopping
                if early_stopping_counter > early_stopping:
                    break
                   
        end=timeit.default_timer()        
        print("Training time : %10.5f"%(end-start))
        
        # Log the results to a file
        with open(modelDir + "/results.txt", 'a+') as file:
            file.write(cellType + "\n")
            file.write("Time taken: " + str(end - start) + "\n")
            file.write("Itr stopped: " + str(min_validation_itr) + "\n")
            file.write("Min validation error: " + str(min_validation_err) + "\n")
            file.write("Denormalized validation error: " + str(min_denorm_err) + "\n\n")
            
       
    return min_validation_itr, min_validation_err


def load_and_predict(test,modelDir,cellType,itr,seq_size):
    # Restore the session
    with tf.Session() as sess:
        print ("Load model:%s-%s"%(modelDir,itr))
        saver = tf.train.import_meta_graph("%s/model_%s-%s.meta"%(modelDir,cellType,itr))
        saver.restore(sess,tf.train.latest_checkpoint("%s"%modelDir))
        graph = tf.get_default_graph()
        # print all nodes in saved graph 
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # get tensors by name to use in prediction
        x = graph.get_tensor_by_name("x_in:0")
        dropout= graph.get_tensor_by_name("dropout_in:0")
        y_pred = graph.get_tensor_by_name("y_pred:0")
        
        # Feed entire test set to get predictions
        predicted_vals_all= sess.run(y_pred, feed_dict={ x: test.features, dropout:1})
        # Get last item in each predicted sequence:
        predicted_vals = predicted_vals_all[:,seq_size-1]
    return predicted_vals


"""
# Perform a crude grid search
from itertools import product
# Define learning rates, dropout
params = [[0.1, 0.01, 0.001], [0.25, 0.5, 0.75]]
# Iterate over all combinations and test model
# with those parameters, storing the min
min_param_val = sys.float_info.max
min_param_elems = None
for elem in product(*params):
    # Unpack the values
    learning_rate, keep_prob = elem
    RNNcell = [rnn.BasicLSTMCell(hidden_dim) for _ in range(n_layers)]
    cellType = "LSTM"
    
    # Build models and save model
    end_itr, min_err = build_and_save_d(modelDir=modelDir,
                     train=train_data,
                     valid=valid_data,
                     cell=RNNcell,
                     cellType=cellType,
                     input_dim=input_dim,
                     hidden_dim=hidden_dim,
                     seq_size=seq_len,
                     keep_prob=keep_prob,
                     batch_size=batch_size,
                     num_epochs=num_epochs,
                     log=log,
                     early_stopping=early_stopping,
                     learning_rate=learning_rate)
    # See if we have a new low error
    if min_err < min_param_val:
        min_param_val = min_err
        min_param_elems = elem
    print("Min validation error " + str(min_err) + " for elems " + str(elem))

print("Global validation error " + str(min_param_val) + " for elems " + str(min_param_elems))
"""

# Grid search on learning rate and dropout

# RNN

#Min validation error 0.015986204 for elems (0.1, 0.25)
#Min validation error 0.015794938 for elems (0.1, 0.5)
#Min validation error 0.015503254 for elems (0.1, 0.75)
#Min validation error 0.012949656 for elems (0.01, 0.25)
#Min validation error 0.006430081 for elems (0.01, 0.5)
#Min validation error 0.0046402193 for elems (0.01, 0.75)
#Min validation error 0.029264465 for elems (0.001, 0.25)
#Min validation error 0.012221504 for elems (0.001, 0.5)
#Min validation error 0.008622245 for elems (0.001, 0.75)
#Global validation error 0.0046402193 for elems (0.01, 0.75)

# GRU

#Min validation error 0.0111637125 for elems (0.1, 0.25)
#Min validation error 0.012049832 for elems (0.1, 0.5)
#Min validation error 0.017291395 for elems (0.1, 0.75)
#Min validation error 0.0037756523 for elems (0.01, 0.25)
#Min validation error 0.002122913 for elems (0.01, 0.5)
#Min validation error 0.0032095483 for elems (0.01, 0.75)
#Min validation error 0.00797302 for elems (0.001, 0.25)
#Min validation error 0.008556419 for elems (0.001, 0.5)
#Min validation error 0.0030354045 for elems (0.001, 0.75)
#Global validation error 0.002122913 for elems (0.01, 0.5)

# LSTM

#Min validation error 0.0039516427 for elems (0.1, 0.25)
#Min validation error 0.016133798 for elems (0.1, 0.5)
#Min validation error 0.008657359 for elems (0.1, 0.75)
#Min validation error 0.0010539122 for elems (0.01, 0.25)
#Min validation error 0.0023624634 for elems (0.01, 0.5)
#Min validation error 0.002788953 for elems (0.01, 0.75)
#Min validation error 0.002642741 for elems (0.001, 0.25)
#Min validation error 0.0013699796 for elems (0.001, 0.5)
#Min validation error 0.0020976907 for elems (0.001, 0.75)
#Global validation error 0.0010539122 for elems (0.01, 0.25)


input_dim=19 # dim > 1 for multivariate time series
hidden_dim=100 # number of hiddent units h
keep_prob=0.5
modelDir='modelDir'
log=500 # How often we validate
batch_size=16
num_epochs=15 # MAXIMUM number of epochs (i.e if early stopping is never achieved)
early_stopping = 5 # Number of validation steps without improvement until we stop
learning_rate = 0.01
n_layers = 2

# NEED TO MAKE DIFFERENT COPIES OF THE CELL TO AVOID SELF-REFENTIAL ERRORS
# RNNcell = [rnn.BasicRNNCell(hidden_dim) for _ in range(1)]
# cellType = "RNN"

RNNcell = [rnn.BasicRNNCell(hidden_dim) for _ in range(n_layers)]
cellType = "RNN"

# Build models and save model
end_itr, min_err = build_and_save_d(modelDir=modelDir,
                 train=train_data,
                 valid=valid_data,
                 cell=RNNcell,
                 cellType=cellType,
                 input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 seq_size=seq_len,
                 keep_prob=keep_prob,
                 batch_size=batch_size,
                 num_epochs=num_epochs,
                 log=log,
                 early_stopping=early_stopping,
                 learning_rate=learning_rate)


RNNcell = [rnn.GRUCell(hidden_dim) for _ in range(n_layers)]
cellType = "GRU"

# Build models and save model
end_itr, min_err = build_and_save_d(modelDir=modelDir,
                 train=train_data,
                 valid=valid_data,
                 cell=RNNcell,
                 cellType=cellType,
                 input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 seq_size=seq_len,
                 keep_prob=keep_prob,
                 batch_size=batch_size,
                 num_epochs=num_epochs,
                 log=log,
                 early_stopping=early_stopping,
                 learning_rate=learning_rate)

RNNcell = [rnn.BasicLSTMCell(hidden_dim) for _ in range(n_layers)]
cellType = "LSTM"

# Build models and save model
end_itr, min_err = build_and_save_d(modelDir=modelDir,
                 train=train_data,
                 valid=valid_data,
                 cell=RNNcell,
                 cellType=cellType,
                 input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 seq_size=seq_len,
                 keep_prob=keep_prob,
                 batch_size=batch_size,
                 num_epochs=num_epochs,
                 log=log,
                 early_stopping=early_stopping,
                 learning_rate=learning_rate)

# In[ ]:

modelDir="modelDir"
cellType="GRU"
end_itr=16000
# Load and predict
predicted_vals_rnn=load_and_predict(test_data,modelDir,cellType,end_itr,seq_len)

# Compute MSE
# step 1: denormalize data
predicted_vals_dnorm_rnn=predicted_vals_rnn*(max_dataset-min_dataset)+min_dataset
# step 2: get ground-truth, also must be denormalized
actual_test= np.array([x[-1] for x in test_data.response])*(max_dataset-min_dataset)+min_dataset
# step 3: compute MSE
mse_rnn= ((predicted_vals_dnorm_rnn - actual_test) ** 2).mean()
 
print("RNN MSE = %10.5f"%mse_rnn)

pred_len=len(predicted_vals_dnorm_rnn)
train_len=len(test_data.features)

pred_avg = []
actual_avg = []
# Compute the moving average of each set for visual purposes
moving_length = 24
for i in range(len(actual_test) - moving_length):
    pred_avg.append(np.mean(predicted_vals_dnorm_rnn[i:i+moving_length]))
    actual_avg.append(np.mean(actual_test[i:i+moving_length]))

# Plot the results
plt.figure()
plt.plot(list(range(len(actual_test))), predicted_vals_dnorm_rnn, color='r', label=cellType)
plt.plot(list(range(len(actual_test))), actual_test, color='g', label='Actual')
plt.plot(list(range(int(moving_length/2), len(actual_test)-int(moving_length/2))), pred_avg, color='y', label="{0} MA".format(cellType))
plt.plot(list(range(int(moving_length/2), len(actual_test)-int(moving_length/2))), actual_avg, color='b', label="Actual MA")
plt.legend()

###########################################################################################################################
############# Part 2: Weather Type Prediction #############################################################################
###########################################################################################################################
# In[ ]: 

# Start on the second problem

# Create the observations the same as in part 1, but now the
# response is a one hot encoded vector


# In[ ]:

def build_and_save_d2(modelDir,train,valid,cell,cellType,input_dim=1,hidden_dim=100,
                          seq_size = 12,max_itr=200,keep_prob=0.5, batch_size=32, num_epochs=10,log=500,save=1000):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        # input place holders, note the change in dimensions in y, which 
        # now has 7 dimensions
        
        # input Shape: [# training examples, sequence length, # features]
        x = tf.placeholder(tf.float32,[None,seq_size,input_dim],name="x_in")
        # label Shape: [# training examples, sequence length, # classes]
        y = tf.placeholder(tf.float32,[None,seq_size,7],name="y_in")
        dropout = tf.placeholder(tf.float32,name="dropout_in")
        
        # Function to wrap each cell with dropout
        def wrap_cell(cell, keep_prob):
            drop = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return drop
        
        cells = tf.nn.rnn_cell.MultiRNNCell(
                 [wrap_cell(cell,keep_prob) for cell in cell]
         )
        
        # RNN output Shape: [# training examples, sequence length, # hidden] 
        outputs, _ = tf.nn.dynamic_rnn(cells,x,dtype=tf.float32)
        
        
        # weights for output dense layer (i.e., after RNN)
        # W shape: [# hidden, 7]
        W_out = tf.Variable(tf.random_normal([hidden_dim,7]),name="w_out") 
        # b shape: [7]
        b_out = tf.Variable(tf.random_normal([7]),name="b_out")
    
        # output dense layer:
        num_examples = tf.shape(x)[0] 
        # convert W from [# hidden, 7] to [# training examples, # hidden, 7]
        # step 1: add a new dimension at index 0 using tf.expand_dims
        w_exp= tf.expand_dims(W_out,0)
        # step 2: duplicate W for 'num_examples' times using tf.tile
        W_repeated = tf.tile(w_exp,[num_examples,1,1])
        
        # Dense Layer calculation: 
        # [# training examples, sequence length, # hidden] *
        # [# training examples, # hidden, 1] = [# training examples, sequence length]
        
        y_pred = tf.matmul(outputs,W_repeated) + b_out
        
        # Cost & Training Step
        # Minimize error with softmax cross entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y))
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        saver=tf.train.Saver()
        
        # Run Session
        with tf.Session(graph=graph) as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # Run for 1000 iterations (1000 is arbitrary, need a validation set to tune!)
            start=timeit.default_timer()
            epoch_counter = 0 # Keep track of our epochs
            i = 0 # Keep track of our iterations
            min_validation_err = sys.float_info.max # Start min error at biggest number
            min_validation_itr = 0 # Keep track of the smallest validation error we have seen so far
            early_stopping_counter = 0 # Counter to see if we have acheived early stopping
            
            print('Training %s ...'%cellType)
            while True: # If we train more, would we overfit? Try 10000
                i += 1 # Increment counter
                trainX, trainY, done = train.next_batch(batch_size) # Get train batch
                # See if we are done with our epochs
                if done:
                    epoch_counter += 1
                    print("Done with epoch " + str(epoch_counter))
                    if epoch_counter >= num_epochs:
                        break
                
                # Pass the data through the network
                _, train_err = sess.run([train_op,cost],feed_dict={x:trainX,y:trainY,dropout:keep_prob})
                if i==0:
                    print('  step, train err= %6d: %8.5f' % (0,train_err))
                # Every 'log' steps, print out train error and validation error.
                # Update early stopping at these points
                elif  (i+1) % log == 0: 
                    print('  step, train err= %6d: %8.5f' % (i+1,train_err)) 
                    
                    # Get validation error on the full validation set
                    valid_err = sess.run(cost,feed_dict={x:valid.features,y:valid.response,dropout:1})
                    print('  step, validation err= %6d: %8.5f' % (i+1,valid_err)) 
                    
                    # Check early stopping
                    early_stopping_counter += 1
                    # If we have smaller validation error, reset counter and
                    # assign new smallest validation error. Also store the
                    # current iterqtion as the iteration where the current min is
                    if valid_err < min_validation_err:
                        min_validation_err = valid_err
                        min_validation_itr = i + 1
                        early_stopping_counter = 0
                        
                        # Store the current best model
                        modelPath= saver.save(sess,"%s/model_%s"%(modelDir,cellType),global_step=i+1)
                        print("model saved:%s"%modelPath) 
                    
                    # Break if we achieve early stopping
                    if early_stopping_counter > early_stopping:
                        break
       
            end=timeit.default_timer()        
            print("Training time : %10.5f"%(end-start))
        return min_validation_itr, min_validation_err

input_dim=19 # dim > 1 for multivariate time series
hidden_dim=100 # number of hiddent units h
keep_prob=0.5
modelDir='modelDir2' # Make sure to use a different model dir
log=1000 # How often we validate
batch_size=16
num_epochs=3 # MAXIMUM number of epochs (i.e if early stopping is never achieved)
early_stopping = 10 # Number of validation steps without improvement until we stop

# Different RNN Cell Types
# NEED TO MAKE DIFFERENT COPIES OF THE CELL TO AVOID SELF-REFENTIAL ERRORS
RNNcell = [rnn.BasicRNNCell(hidden_dim) for _ in range(2)]
cellType = "RNN"

# Build models and save model
end_itr, min_err = build_and_save_d2(modelDir=modelDir,
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

print("Min validation error {0}".format(str(min_err)))

# In[ ]:
def load_and_predict2(test,modelDir,cellType,itr,seq_size):
    with tf.Session() as sess:
        print ("Load model:%s-%s"%(modelDir,itr))
        saver = tf.train.import_meta_graph("%s/model_%s-%s.meta"%(modelDir,cellType,itr))
        saver.restore(sess,tf.train.latest_checkpoint("%s"%modelDir))
        graph = tf.get_default_graph()
        
        x = graph.get_tensor_by_name("x_in:0")
        dropout= graph.get_tensor_by_name("dropout_in:0")
        y_pred = graph.get_tensor_by_name("y_out:0")
        
        predicted_vals_all= sess.run(y_pred, feed_dict={ x: test.features, dropout:1})
        # Get last item in each predicted sequence:
        predicted_vals = predicted_vals_all[:,seq_size-1]
    return predicted_vals
        
# Load and predict
predicted_vals_rnn=load_and_predict2(test_data2,modelDir,cellType,end_itr,seq_len)

print(predicted_vals_rnn)
# Compute MSE
# step 2: get ground-truth
actual_test= np.array([x[-1] for x in test_data2.response])

# Get raw accuracy
sum(np.argmax(actual_test, axis=1) == np.argmax(predicted_vals_rnn, axis=1))/len(actual_test)

# In[ ]:

# Calculate f1_score

# Convert the continuous valued predictions
# to one hot
preds = np.zeros([len(actual_test), 7])
for i in range(len(actual_test)):
    preds[i, np.argmax(predicted_vals_rnn[i])] = 1
# Use the weighted version for more accuracy in the multiclass setting
f1_score(actual_test, preds, average="weighted")