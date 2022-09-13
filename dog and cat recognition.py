#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


pip install keras


# In[3]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[4]:


tf.__version__


# # Part 1 - Data Preprocessing
# 

# ## Preprocessing the Training set

# In[5]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set =  train_datagen.flow_from_directory('dataset/Dogs_Cats/Training_set',
                                                   target_size = (64,64),
                                                   batch_size = 32,
                                                   class_mode = 'binary')


# In[6]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set =  test_datagen.flow_from_directory('dataset/Dogs_Cats/Test_set',
                                                   target_size = (64,64),
                                                   batch_size = 32,
                                                   class_mode = 'binary')


# # Part 2 - Building the CNN
# 

# ## Initialising the CNN

# In[7]:


cnn = tf.keras.models.Sequential()


# ### step 1 - Convultion
# 
# 

# In[8]:


cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape=[64,64,3]))


# ### step 2 - Pooling

# In[9]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Adding a second convultion layer

# In[10]:


cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### step 3 - Flattening

# In[11]:


cnn.add(tf.keras.layers.Flatten())


# ### step 4 - Full Connection

# In[12]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ### step 5 - Output Layer

# In[13]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# # Part 3 - Training the CNN

# ## Compiling the CNN

# In[14]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the CNN on the training set and evaluating it on the test set

# In[15]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 5)


# # Part 4 - Making a Single Prediction

# In[33]:


import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('dataset/Dogs_Cats/Single_Prediction/H.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0] [0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


# In[34]:


print(prediction)


# In[ ]:




