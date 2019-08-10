#!/usr/bin/env python
# coding: utf-8

# ## 1. Loading and preprocessing data

# In[58]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tf.__version__


# In[2]:


import os
from six.moves import cPickle as pickle
import numpy as np
import scipy.io as sio
import scipy.misc as spm
from scipy import ndimage
import datetime
import matplotlib.image as plt
from IPython.display import Image, display
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from collections import Counter



# In[3]:


def reformat_date(mat_date):
    dt = datetime.date.fromordinal(np.max([mat_date - 366, 1])).year
    return dt

def create_path(path):
    return os.path.join(IMG_DIR, path[0])


# In[4]:


IMG_DIR = 'wiki'
#MAT_FILE = os.path.expanduser("~")+'/wiki/wiki.mat'  
MAT_FILE = 'wiki/wiki.mat'  
mat_struct = sio.loadmat(MAT_FILE)
data_set = [data[0] for data in mat_struct['wiki'][0, 0]]

keys = ['dob',
    'photo_taken',
    'full_path',
    'gender',
    'name',
    'face_location',
    'face_score',
    'second_face_score',
    'celeb_names',
    'celeb_id'
]

imdb_dict = dict(zip(keys, np.asarray(data_set)))
imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]
imdb_dict['full_path'] = [create_path(path) for path in imdb_dict['full_path']]

# Add 'age' key to the dictionary
imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']

print("Dictionary created...")

raw_path = imdb_dict['full_path']
raw_age = imdb_dict['age']
raw_gender = imdb_dict['gender']
raw_name = imdb_dict['name']
raw_face_location = imdb_dict['face_location']
raw_face_score = imdb_dict['face_score']
raw_second_face_score = imdb_dict['second_face_score']


# In[8]:


for i in range(2):
    display(Image(filename=raw_path[i]))
    print("Path: " + str(raw_path[i]))
    print("Age: " + str(raw_age[i]))
    print("Gender: " + str(raw_gender[i]))
    #print("Name: " + str(raw_name[i]))
    print("Face location: " + str(raw_face_location[i]))
    print("Face score: " + str(raw_face_score[i]))
    print("Second face score: " + str(raw_second_face_score[i]))
    #print("Celeb id: " + str(raw_celeb_id[i])+"\n")


# In[9]:


n=[]
path=[]
label=[]
import cv2
for i in range(len(raw_path)):
    if raw_face_score[i]>0:
        if raw_age[i] in range(1,100):
            try:
                img = cv2.imread(raw_path[i], cv2.IMREAD_UNCHANGED)  
                up=int(raw_face_location[i][0][1])
                bottom=int(raw_face_location[i][0][3])
                left=int(raw_face_location[i][0][0])
                right=int(raw_face_location[i][0][2])
                if up==bottom or left==right:
                    cropped=img
                else:
                    cropped = img[up:bottom,left:right]


                cv2.imwrite(raw_path[i], cropped)
                label.append(raw_age[i])
                path.append(raw_path[i])
                n.append(i)
            except TypeError:
                pass


# In[266]:


IMG_DIR = 'imdb'
MAT_FILE = 'imdb/imdb.mat'  
mat_struct = sio.loadmat(MAT_FILE)
data_set = [data[0] for data in mat_struct['imdb'][0, 0]]

keys = ['dob',
    'photo_taken',
    'full_path',
    'gender',
    'name',
    'face_location',
    'face_score',
    'second_face_score',
    'celeb_names',
    'celeb_id'
]

imdb_dict = dict(zip(keys, np.asarray(data_set)))
imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]
imdb_dict['full_path'] = [create_path(path) for path in imdb_dict['full_path']]

# Add 'age' key to the dictionary
imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']

print("Dictionary created...")


raw_path1 = imdb_dict['full_path']
raw_age1 = imdb_dict['age']
raw_gender1 = imdb_dict['gender']
raw_name1 = imdb_dict['name']
raw_face_location1 = imdb_dict['face_location']
raw_face_score1 = imdb_dict['face_score']
raw_second_face_score1 = imdb_dict['second_face_score']
raw_celeb_names1 = imdb_dict['celeb_names']
raw_celeb_id1 = imdb_dict['celeb_id']


# In[284]:


n1=[]
path1=[]
label1=[]
import cv2
for i in range(len(raw_path1)):
    if raw_face_score1[i]>0:
        if raw_age1[i] in range(1,100):
            try:
                img = len(cv2.imread(raw_path1[i], cv2.IMREAD_UNCHANGED))
                label1.append(raw_age1[i])
                path1.append(raw_path1[i])
                n1.append(i)
            except TypeError:
                pass


# In[287]:


n+=n1
path+=path1
label+=label1


# In[288]:


sns.distplot(label);
print("Age size: " + str(len(label)))


# In[551]:


group=[]
for age in label:
    age=int(age)
    if age in range(0,13):
        group.append(0)
    elif age in range(13,20):
        group.append(1)
    elif age in range(20,32):
        group.append(2)
    elif age in range(32,43):
        group.append(3)
    elif age in range(43,53):
        group.append(4)
    else:
        group.append(5)


# In[223]:


from collections import Counter
cnt=Counter(group)
cnt


# In[552]:


newpath=[]
newlabel=[]
for a in range(0,6):
    tl=[]
    tp=[]
    for i in range(len(group)):
        if group[i]==a:
            tl.append(group[i])
            tp.append(path[i])
    if len(tl)<500:
        newpath+=tp
        newlabel+=tl
    else:
        newpath+=tp[:500]
        newlabel+=tl[:500]


# In[568]:


IMG_SIZE = 192
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


# In[570]:


from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(newpath, newlabel, test_size=0.1, random_state=56)


# In[571]:


path_ds1 = tf.data.Dataset.from_tensor_slices(X_train)
image_ds1 = path_ds1.map(load_and_preprocess_image)
label_ds1 = tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.int64))
train = tf.data.Dataset.zip((image_ds1, label_ds1))


# In[572]:


path_ds2 = tf.data.Dataset.from_tensor_slices(X_validation)
image_ds2 = path_ds2.map(load_and_preprocess_image)
label_ds2 = tf.data.Dataset.from_tensor_slices(tf.cast(y_validation, tf.int64))
validation = tf.data.Dataset.zip((image_ds2, label_ds2))


# In[540]:


#read evaluation dataset
import pandas as pd
df=pd.read_csv('faces.txt',sep='\t')# read csv into df

#vals = df.loc[:,:].values
age=df["age"].values
path=df["original_image"].values

file="faces"
fullpath=[]
for filename in os.listdir(file):
    if filename!=".DS_Store":
        for img in os.listdir(os.path.join(file, filename)):
            fullpath.append(str(os.path.join(file, filename,img)))
    
testpath=[]
for item in path:
    for fpath in fullpath:
        if item in fpath:
            testpath.append(fpath)
            break
            
testgroup=[]
testpatha=[]
for a in range(len(age)):
    i=age[a]
    if i=="(25, 32)":
        testgroup.append(2)
        testpatha.append(testpath[a])
    elif i=='(38, 43)':
        testgroup.append(3)
        testpatha.append(testpath[a])
    elif i=="(4, 6)":
        testgroup.append(0)
        testpatha.append(testpath[a])
    elif i=="(60, 100)":
        testgroup.append(5)
        testpatha.append(testpath[a])
    elif i=="(15, 20)":
        testgroup.append(1)
        testpatha.append(testpath[a])
    elif i=="(48, 53)":
        testgroup.append(4)
        testpatha.append(testpath[a])
    elif i=="(8, 12)":
        testgroup.append(0)
        testpatha.append(testpath[a])
    elif i=="(0, 2)":
        testgroup.append(0)
        testpatha.append(testpath[a])
    elif i=="35":
        testgroup.append(3)
        testpatha.append(testpath[a])
    elif i=="55":
        testgroup.append(5)           
        testpatha.append(testpath[a])


# In[586]:


path_ds3 = tf.data.Dataset.from_tensor_slices(testpatha)
image_ds3 = path_ds3.map(load_and_preprocess_image)
label_ds3 = tf.data.Dataset.from_tensor_slices(tf.cast(testgroup, tf.int64))
test = tf.data.Dataset.zip((image_ds3, label_ds3))


# In[573]:


def augment_data(image, label):
  print("Augment data called!")
  image = tf.image.random_flip_left_right(image)
  
  return image, label

train = train.map(augment_data)


# In[574]:


SHUFFLE_BUFFER_SIZE = 1024
BATCH_SIZE = 32

train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation = validation.batch(BATCH_SIZE)

# (Optional) prefetch will enable the input pipeline to asynchronously fetch batches while
# your model is training.
train = train.prefetch(tf.data.experimental.AUTOTUNE)
print(train)
print(validation)


# In[587]:


test=test.batch(BATCH_SIZE)


# ## 2. Building model

# In[271]:


from tensorflow import keras
keras.__version__


# In[299]:


# Creating a simple CNN model in keras using functional API
def create_model():
    img_inputs = keras.Input(shape=IMG_SHAPE)
    conv_1 = keras.layers.Conv2D(96,(7,7),(3,3),padding='VALID', activation='relu')(img_inputs)
    maxpool_1 = keras.layers.MaxPooling2D(2,3 )(conv_1)
    norm_1 = tf.nn.local_response_normalization(maxpool_1, 5, alpha=0.0001, beta=0.75, name='norm_1')
    conv_2 = keras.layers.Conv2D(256, (5,5), (1,1), activation='relu',padding='SAME')(norm_1)
    maxpool_2 = keras.layers.MaxPooling2D((2, 2))(conv_2)
    norm_2 = tf.nn.local_response_normalization(maxpool_2, 5, alpha=0.0001, beta=0.75, name='norm_2')
    conv_3 = keras.layers.Conv2D(384, (3,3), (1,1), activation='relu',padding='SAME')(norm_2)
    maxpool_3 = keras.layers.MaxPooling2D((2, 2))(conv_3)
    flatten = keras.layers.Flatten()(maxpool_3)
    dense_1 = keras.layers.Dense(512, activation='relu')(flatten)
    drop_1 = keras.layers.Dropout(0.3)(dense_1)
    dense_2 = keras.layers.Dense(512, activation='relu')(drop_1)
    drop_2 = keras.layers.Dropout(0.3)(dense_2)
    output = keras.layers.Dense(len(set(group)), activation='softmax')(drop_2)

    model = keras.Model(inputs=img_inputs, outputs=output)
    
    return model


# In[558]:


simple_model = create_model()
simple_model.summary()


# In[287]:


keras.utils.plot_model(simple_model, 'age prediction.png', show_shapes=True)


# In[575]:


num_train=len(X_train)
num_val=len(X_validation)



steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = round(num_val)//BATCH_SIZE

print('Number of examples in the train set:', steps_per_epoch)
print('Number of examples in the validation set:', validation_steps)


# ### Compile and train the model

# In[560]:


import os

def train_model(model,sp,vs):
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=5,
        verbose=1),
      ]
    
   
    history = model.fit(train.repeat(),
                              epochs=10,
                              steps_per_epoch = sp,
                              validation_data=validation.repeat(), 
                              validation_steps=vs,
                              callbacks=callbacks,
                             
                              )
    return history


# In[561]:


history = train_model(simple_model,steps_per_epoch,validation_steps)


# In[562]:


for a in range(0,6):
    tl=[]
    tp=[]
    for i in range(len(testpatha)):
        if testgroup[i]==a:
            tl.append(testgroup[i])
            tp.append(testpatha[i])
    path_ds3 = tf.data.Dataset.from_tensor_slices(tp)
    image_ds3 = path_ds3.map(load_and_preprocess_image)
    label_ds3 = tf.data.Dataset.from_tensor_slices(tf.cast(tl, tf.int64))
    test = tf.data.Dataset.zip((image_ds3, label_ds3))
    BATCH_SIZE=32
    test=test.batch(BATCH_SIZE)
    simple_model.evaluate(test)


# Plotting the training and validation metrics returned by the `train_model()` routine. We use matplotlib to plot the graph.

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[538]:


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# ## 3. Using pre-trained model

# ##inception-v3

# In[17]:


from tensorflow import keras

# Create the base model from the pre-trained model MobileNet V2
base_model = keras.applications.InceptionV3(input_shape=IMG_SHAPE,
# We cannot use the top classification layer of the pre-trained model as it contains 1000 classes.
# It also restricts our input dimensions to that which this model is trained on (default: 299x299)
                                               include_top=False, 
                                               weights='imagenet')


# In[549]:


keras.utils.plot_model(base_model, 'inception_model_with_shape_info.png', show_shapes=True)


# In[22]:


base_model.trainable = False
# Let's take a look at the base model architecture
base_model.summary()


# In[23]:


def build_model():
    
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(len(set(group)), 
                           activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

v3 = build_model()
v3.summary()


# In[19]:


##mobile_net
basic = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
basic.trainable=False


# In[71]:


def change_range(image,label):
  return 2*image-1, label

train_mobile = train.map(change_range)
validation_mobile=validation.map(change_range)


# In[576]:


def build_model():
    
    model = keras.Sequential([
        basic,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(len(set(group)), 
                           activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

mobile_net = build_model()
mobile_net.summary()


# In[577]:


log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir)

# Creating Keras callbacks 
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
os.makedirs('training_checkpoints/', exist_ok=True)
early_stopping_checkpoint = keras.callbacks.EarlyStopping(patience=5)



history = mobile_net.fit(train_mobile.repeat(),
                              epochs=2,
                              steps_per_epoch = steps_per_epoch,
                              validation_data=validation_mobile.repeat(), 
                              validation_steps=validation_steps,
                              callbacks=[tensorboard_callback,
                              model_checkpoint_callback,
                              early_stopping_checkpoint])


# After training the model for 5 epochs we were able to get ~70% accuracy. We plot the learning curves of the training and validation accuracy / loss.

# In[35]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(np.arange(1,3),acc, label='Training Accuracy')
plt.plot(np.arange(1,3),val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(np.arange(1,3),loss, label='Training Loss')
plt.plot(np.arange(1,3),val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[56]:


mobile_net.save('mobile_net.h5')


# In[55]:


# Save keras model
tf.saved_model.save(mobile_net, 'model/mobilenet1')


# In[593]:


new_model = keras.models.load_model('mobile_net.h5')
new_model.summary()


# In[73]:


# Un-freeze the top layers of the model
basic.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(basic.layers))


# In[74]:


# Fine tune from this layer onwards
fine_tune_at = 145

# Freeze all the layers before the `fine_tune_at` layer
for layer in basic.layers[:fine_tune_at]:
    layer.trainable =  False
    
# Compile the model using a much-lower training rate.
mobile_net.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
mobile_net.summary()


# In[565]:


tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
os.makedirs('training_checkpoints/', exist_ok=True)
early_stopping_checkpoint = keras.callbacks.EarlyStopping(patience=5)
history_fine =mobile_net.fit(train_mobile.repeat(), 
                                  steps_per_epoch = steps_per_epoch,
                                  epochs=8, 
                                  initial_epoch = 3,
                                  validation_data=validation_mobile.repeat(), 
                                  validation_steps=validation_steps,
                                  callbacks=[tensorboard_callback,
                                  model_checkpoint_callback,
                                  early_stopping_checkpoint])


# In[594]:


for a in range(0,6):
    tl=[]
    tp=[]
    for i in range(len(testpatha)):
        if testgroup[i]==a:
            tl.append(testgroup[i])
            tp.append(testpatha[i])
    path_ds3 = tf.data.Dataset.from_tensor_slices(tp)
    image_ds3 = path_ds3.map(load_and_preprocess_image)
    label_ds3 = tf.data.Dataset.from_tensor_slices(tf.cast(tl, tf.int64))
    test = tf.data.Dataset.zip((image_ds3, label_ds3))
    BATCH_SIZE=32
    test=test.batch(BATCH_SIZE)
    mobile_net.evaluate(test)


# In[ ]:


mobile_net.save('mobile1_net.h5')


# In[ ]:


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


# In[81]:


initial_epochs=2.05

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(np.arange(1,7),acc, label='Training Accuracy')
plt.plot(np.arange(1,7),val_acc, label='Validation Accuracy')
plt.plot([initial_epochs,initial_epochs], 
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(np.arange(1,7),loss, label='Training Loss')
plt.plot(np.arange(1,7),val_loss, label='Validation Loss')
plt.plot([initial_epochs,initial_epochs], 
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

