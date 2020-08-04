# -*- coding: utf-8 -*-
"""
CNN Model to predict cervical cancer labels - Multimodal and  Multitask-learning

The present model predict the risk (low or high) of a patient had cervical cancer based on colposcopy images and clinical data.
This model uses two different data sets (NCI and Kaggle) in order to get more data to feed the convolutional network.

Convolutional block: ResNet50

Machine Learning Project
"""
#imports
from keras import models, layers, backend, utils, losses, callbacks 
from data_set_preprocess import load_nci_bin, load_kaggle
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from keras.applications import ResNet50
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from math import ceil
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

################# Define Loss function #################################

def categorical_crossentropy_weights(w):
    def internal_function(y_true, y_pred):
        return w*losses.categorical_crossentropy(y_true, y_pred)
    return internal_function


######################### Data Augmentation #############################
BATCH_SIZE=16
IMAGE_HEIGHT=224
IMAGE_WIDTH=224
IMAGE_CHANNEL=3
SEED = 1

def random_saturation(image):
    saturation_factor = np.random.uniform(0.5, 2)
    image = image*saturation_factor
    ix = image > 1
    image[ix] = 1
    return image

datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.3,
            preprocessing_function = random_saturation)

def apply_augmentation(image):
    params = datagen.get_random_transform(image.shape)
    image_tr = datagen.apply_transform(datagen.standardize(image), params)
    return image_tr

 # it is necessary to create a oversampling_generator for multiinputs when using keras          
def oversampling_generator(images, clinical, labels):
    while True:
        
        # Find the indexes of each class
        index_0 = np.argwhere(labels[:,0]==1) 
        index_1 = np.argwhere(labels[:,1]==1)
        
        # Randomize the indices of the bigger class to make an array
        index_0_permuted = np.random.choice(index_0.flatten(), len(index_0), False)
        
        for batch in range(0, len(index_0), int(BATCH_SIZE/2)):
            
            # Pick random samples, ensuring a balanced batch
            ix_0 = index_0_permuted[batch:(batch + int(BATCH_SIZE/2))] 
            ix_1 = np.random.choice(index_1.flatten(), int(BATCH_SIZE/2), False)
            ix = np.concatenate((ix_0, ix_1))
            
            # initializing the arrays, x_train and y_train
            x_train = np.empty(
                [0, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)
            y_train = np.empty([0, 2], dtype=np.int32)
            c_train = np.empty([0, 4], dtype=np.int32)

            for i in ix:
                # get an image and its corresponding label and clinical data
                image = images[i]
                label = labels[i]
                cdata = clinical[i]
                
                image = apply_augmentation(image)

                # Appending them to existing batch
                x_train = np.append(x_train, [image], axis=0)
                y_train = np.append(y_train, [label], axis=0)
                c_train = np.append(c_train, [cdata], axis=0)

            yield ([x_train, c_train], y_train)
            

for subset in range(0,5):
    
    print('Load data...')
    Xn, Yn, Bn, Fn, Cn = load_nci_bin(subset)
    Xk, Yk = load_kaggle()
    
    alpha = 0.7
    
    ### Data augmentation for nci dataset ###
    # Find the indexes of each class
    index_0 = np.argwhere(Yn[0][:,0]==1) 
    index_1 = np.argwhere(Yn[0][:,1]==1)
    
    nci_gen = oversampling_generator(Xn[0], Cn[0], Yn[0])
    
    ########################### Define Shared Layers ###############################
    #IMG Dimensions
    img_rows = 224
    img_cols = 224
    img_channel = 3
    x = img_input = layers.Input([img_rows, img_cols, img_channel])
    clinical_input = layers.Input(shape=(4,))
    base_ResNet50=ResNet50(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
    
    output_shared = base_ResNet50(x)
    output_shared = layers.Flatten()(output_shared)
    
    ########################### Define Specific Layers ###############################
    # KAGGLE LAYERS
    x = layers.Dense(10, activation='relu')(output_shared)
    x = BatchNormalization()(x)
    x = layers.Dense(3, activation='softmax')(x)
    
    output_kg = x
    
    # NCI LAYERS
    x = layers.Dense(10, activation='relu',activity_regularizer=l1(1))(output_shared)
    x = BatchNormalization()(x)
    x = layers.Concatenate(axis=-1)([x, clinical_input])
    x = layers.Dense(2, activation='softmax')(x)
    
    output_nci = x
    
    ########################## Define Model #################################
    
    model1 = models.Model([img_input, clinical_input], output_nci)
    model1.compile('adam', loss=categorical_crossentropy_weights(w=alpha), metrics=['accuracy'])
    
    model2 = models.Model(img_input, output_kg)
    model2.compile('adam', loss=categorical_crossentropy_weights(w=1-alpha), metrics=['accuracy'])
    
    model_all = models.Model([img_input, clinical_input], [output_nci, output_kg])
    model_all.summary()
    
    
    
    ############################# TRAIN #####################################
    
    #calculate weights for balance classes 
    weights = len(Yn[0])/(np.bincount(Yn[0].argmax(1)))
    print('Train Model')
    
    #create arrays to store the loss values
    loss=[]
    loss_val=[]
    
    for epoch in range(150):  
        print('* Epoch %d/%d' % (epoch+1, 150))
        
        history=model1.fit_generator(
                nci_gen, steps_per_epoch=ceil(len(index_0)*2/BATCH_SIZE),
                epochs=epoch+1, verbose=1, initial_epoch=epoch,
                class_weight=weights, validation_data=([Xn[1], Cn[1]], Yn[1]))
        
        #Save loss for plot the results
        loss = loss + history.history['loss']
        loss_val = loss_val + history.history['val_loss']
        
        model2.fit_generator(
                datagen.flow(Xk[0], Yk[0], batch_size=BATCH_SIZE),
                steps_per_epoch=int( np.ceil(len(Xn[0])/ BATCH_SIZE)),
                epochs=epoch+1, verbose=1, initial_epoch=epoch,
                validation_data=(Xk[1], Yk[1]))
    
    model_all.save('MTL_ResNet50_clinical_%d.h5' % subset)
    
    
    # summarize history for accuracy
    name = 'simple_mtl_k%d' % (subset)
    plt.plot(loss, label='train loss')
    plt.plot(loss_val, label='val loss')
    plt.legend()
    plt.title(name)
    plt.savefig('history-%s.png' % name)
    
  
    
    ############################# TEST #####################################
    
    prediction, _ = model_all.predict([Xn[2], Cn[2]])
    score_pred = prediction[:,1]
    
    print('NCI:',
        '\n Accuracy:', metrics.accuracy_score(Yn[2].argmax(1), prediction.argmax(1)),
        '\n Balanced acc:', metrics.balanced_accuracy_score(Yn[2].argmax(1), prediction.argmax(1)),
        '\n Confusion Matrix:', metrics.confusion_matrix(Yn[2].argmax(1), prediction.argmax(1)),
        '\n AUC:', metrics.roc_auc_score(Yn[2].argmax(1), score_pred)
        )
    
    f = open('results_MTL_ResNet50_clinical.txt', 'a+')
    f.write('\n\nNCI_%d:' % subset +
        ' Confusion Matrix:'+ str(metrics.confusion_matrix(Yn[2].argmax(1), prediction.argmax(1)))+
        ' AUC:'+ str(metrics.roc_auc_score(Yn[2].argmax(1), score_pred))
        )
    f.close()