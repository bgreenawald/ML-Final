
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


# # Preprocessing

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


# ## Split into train, test, and validation

# In[2]:


full_df = pd.read_csv("df_weather_scaled_encoded.csv")


# In[3]:


# Filter by the city of interest
current_city = "Charlotte"

full_df = full_df[full_df["city"] == current_city]


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


# ### Create observations using a sliding sequence window

# In[26]:


seq_len = 72

train_x = [] 
train_y = []
baseline_err = 0

for i in range(train.shape[0] - seq_len):
    train_x.append([x for x in train.iloc[i:i+seq_len, :].values])
    train_y.append([y for y in train.iloc[i+1:i+seq_len+1, 0]])
    
    # Keep a running sum of squared error
    baseline_err += (np.mean(train.iloc[i:i+seq_len, 0]) - train.iloc[i+seq_len, 0]) ** 2


# ## Wrapper for dataset object

# In[11]:


# Get the mean squared error of the naive model


# In[ ]:


# Modified from Mohammad al Boni

class DataSet(object):
    def __init__(self, data):
        self._num_examples = len(data.shape[0])
        self._x = dt
        self._y = labels
        self._epochs_done = 0
        self._index_in_epoch = 0
        np.random.seed(123456)
        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        random.seed(123456)
    @property
    def features(self):
        return self._x
    @property
    def response(self):
        return self._labels
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
            self._images = self._images[perm]
            self._labels = self._labels[perm] 
            start = 0
            self._index_in_epoch = batch_size
            done = True
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
    
        return self._images[start:end], self._labels[start:end], done


# ## Create baselines

# In[ ]:


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
        
def build_and_save_d(modelDir,trainX,trainY,cell,cellType,input_dim=1,hidden_dim=100,
                          seq_size = 12,max_itr=200,keep_prob=0.5):
    graph = tf.Graph()
    with graph.as_default():
        # input place holders
        # input Shape: [# training examples, sequence length, # features]
        x = tf.placeholder(tf.float32,[None,seq_size,input_dim],name="x_in")
        # label Shape: [# training examples, sequence length]
        y = tf.placeholder(tf.float32,[None,seq_size],name="y_in")
        dropout = tf.placeholder(tf.float32,name="dropout_in")
        
        cell = rnn_cell.DropoutWrapper(cell)
        # RNN output Shape: [# training examples, sequence length, # hidden] 
        outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
        
        
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
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        saver=tf.train.Saver()
        
        # Run Session
    with tf.Session(graph=graph) as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # Run for 1000 iterations (1000 is arbitrary, need a validation set to tune!)
        start=timeit.default_timer()
        print('Training %s ...'%cellType)
        for i in range(max_itr): # If we train more, would we overfit? Try 10000
            _, train_err = sess.run([train_op,cost],feed_dict={x:trainX,y:trainY,dropout:keep_prob})
            if i==0:
                print('  step, train err= %6d: %8.5f' % (0,train_err)) 
            elif  (i+1) % 100 == 0: 
                print('  step, train err= %6d: %8.5f' % (i+1,train_err)) 
            if i>0 and (i+1) % 100 == 0:    
                modelPath= saver.save(sess,"%s/model_%s"%(modelDir,cellType),global_step=i+1)
                print("model saved:%s"%modelPath)    
        end=timeit.default_timer()        
        print("Training time : %10.5f"%(end-start))
       
    return 


def load_and_predict(testX,modelDir,cellType,itr):
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
        
        predicted_vals_all= sess.run(y_pred, feed_dict={ x: testX, dropout:1})
        # Get last item in each predicted sequence:
        predicted_vals = predicted_vals_all[:,seq_size-1]
    return predicted_vals


input_dim=1 # dim > 1 for multivariate time series
hidden_dim=100 # number of hiddent units h
max_itr=500 # number of training iterations
keep_prob=0.5
modelDir='modelDir'
# Different RNN Cell Types
RNNcell = rnn.BasicRNNCell(hidden_dim)

# Build models and save model
build_and_save_d(modelDir,dataX_train,dataY_train,RNNcell,"RNN",input_dim,hidden_dim,seq_size,max_itr,keep_prob
                 
    # Load and predict
predicted_vals_rnn=load_and_predict(dataX_test,modelDir,"RNN",max_itr)
# Compute MSE
# step 1: denormalize data
predicted_vals_dnorm_rnn=predicted_vals_rnn*(max_dataset-min_dataset)+min_dataset
# step 2: get ground-truth
actual_test=dataset[seq_size+train_size:len(dataset_norm)]
# step 3: compute MSE
mse_rnn= ((predicted_vals_dnorm_rnn - actual_test) ** 2).mean()
 
print("RNN MSE = %10.5f"%mse_rnn)

# Plot predictions
pred_len=len(predicted_vals_dnorm_rnn)
train_len=len(dataX_train)

plt.figure()
plt.plot(list(range(seq_size+train_len,seq_size+train_len+train_len)), predicted_vals_dnorm_rnn, color='r', label='RNN')
plt.plot(list(range(len(dataset))), dataset, color='g', label='Actual')
plt.legend()