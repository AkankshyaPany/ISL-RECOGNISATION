# Importing the Keras libraries and packages
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense , Dropout,GlobalAveragePooling2D
import matplotlib.pyplot as plt
import os

from tensorflow.keras.applications import VGG19#,preprocess_input    #from keras.applications.vgg19 import VGG19,preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   #@@@ 1 tha
sz = 200     # @@@@@@@ 128 tha    What is 128 here? ie our img size is 128*128 pixels


print("Enter the Algorithm number to run")
print("1.CNN (1 conv layers) 2.CNN (2 conv layers) 3.CNN (3 conv layers) 4.ANN 5.VGG19 6.InceptionV3 7.Resnet50")
temp='3'
print("algorithm number selected is {}".format(temp))



train_path='E:\\Anaconda\\Spyder\\Sign-Language-to-Text-master\\Sign-Language-to-Text-master\\data2\\output\\train'
test_path='E:\\Anaconda\\Spyder\\Sign-Language-to-Text-master\\Sign-Language-to-Text-master\\data2\\output\\test'
val_path='E:\\Anaconda\\Spyder\\Sign-Language-to-Text-master\\Sign-Language-to-Text-master\\data2\\output\\val'



#   Batch size * Steps per epoch   ---- is total images in train/test set


if temp=='1':       # CNN with 1 convolution layer
    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(sz,sz,1)))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))   
    classifier.add(Dropout(0.25))
    
    classifier.add(Flatten())   # Dense layer k liye flatten

    classifier.add(Dense(200, activation='relu'))
    Dropout(0.2),
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dense(29, activation='softmax'))

    classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],)


if temp=='2':    # CNN with 2 convolution layers
    # Step 1 - Building the CNN
    
    # Initializing the CNN
    classifier = Sequential()
    
    # First convolution layer and pooling
    classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    # Second convolution layer and pooling
    classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    # input_shape is going to be the pooled feature maps from the previous convolution layer
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    #classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    # input_shape is going to be the pooled feature maps from the previous convolution layer
    #classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening the layers       ..................  FLATTENING HERE
    classifier.add(Flatten())
    
    # Adding a fully connected layer
    classifier.add(Dense(units=200, activation='relu')) #128 tha
    classifier.add(Dropout(0.2))                                                             # WE have 3 dense layers  )
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(units=29, activation='softmax')) # softmax for more than 2        # 27 tha
    
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2

if temp == '3':   # 3 conv layer
    sz=200
    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(sz,sz,1)))
    classifier.add(MaxPooling2D((2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Convolution2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Convolution2D(200, (3, 3), activation='relu'))  #128 tha
    classifier.add(Dropout(0.2))

    classifier.add(Flatten())

    classifier.add(Dense(200, activation='relu'))     # WE have 3 Dense ,layerz 128 tha
    classifier.add(Dropout(0.2))
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dense(29, activation='softmax'))

    classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
if temp=='1' or temp=='2' or temp=='3':
    # Step 2 - Preparing the train/test data and training the model
    
    sz=200
    
    classifier.summary()
    
    
    # Code copied from - https://keras.io/preprocessing/image/
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    
    
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
    
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(sz, sz),
                                                     batch_size=3,               #10
                                                     color_mode='grayscale',
                                                     class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(test_path,                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=4,        #10
                                                color_mode='grayscale',
                                                class_mode='categorical')
    
    
    
    start=time.time()
    
    history=classifier.fit(                                         # added history= lateron
            training_set,
            steps_per_epoch=4000, # No of images in training set  #12841 tha ()   #680
            epochs=6,           #5 tha   #10
            validation_data=test_set,
            validation_steps=500)# No of images in test set #4268 tha  #140
    
    end=time.time()
    print("Time elasped is : ",end-start,"seconds")
    print((end-start)/60,"minutes")
    
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,7)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
    acc_train = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1,7)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



if temp=='4':           #Building the ANN

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,      # max value can be 255 so scaling everything bw 0 to 1 
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(sz, sz),
                                                     batch_size=3,   #16        #10
                                                     color_mode='grayscale',
                                                     class_mode='categorical',)
    
    test_set = test_datagen.flow_from_directory(test_path,                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=4,        #10
                                                color_mode='grayscale',
                                                class_mode='categorical') 
    
    
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
    
    ann = Sequential([
        Flatten(input_shape=(sz,sz,1)),
        Dense(200, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(29, activation='softmax')    
    ])
    
    ann.summary()

    ann.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    start =time.time()

    history=ann.fit(training_set,
                    steps_per_epoch=4000, # No of images in training set  #12841 tha ()   #680
                    epochs=6,           #5 tha   #10
                    validation_data=test_set,
                    validation_steps=500)
    
    end=time.time()
    print("Time elasped is : ",end-start,"seconds")
    print((end-start)/60,"minutes")
    
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,7)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss (ANN)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    acc_train = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1,7)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy (ANN)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    '''   
    ann.evaluate(test_x,test_y,batch_size=32)
    
    #predict
    y_pred=ann.predict(test_x)
    y_pred=np.argmax(y_pred,axis=1)
    #get classification report
    print(classification_report(y_pred,test_y))
    #get confusion matrix
    result=confusion_matrix(y_pred,test_y)
   '''
if temp=='5':
    # create the base pre-trained model
    sz=200
    vgg = VGG19(weights='imagenet', include_top=False) # @@@@@@@@@
    
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
    
    #do not train the pre-trained layers of VGG-19
    for layer in vgg.layers:
        layer.trainable = False

    # add a global spatial average pooling layer
    x = vgg.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(512, activation='relu')(x)  #512 isko glti se 128 kar diya tha
    x = Dropout(0.3)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(29, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=vgg.input, outputs=predictions)
    # train the model on the new data for a few epochs
    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])    #'categorical_accuracy'])
    
    model.summary()
    
    # Code copied from - https://keras.io/preprocessing/image/
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(sz, sz),
                                                     batch_size=3,               #10
                                                     class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(test_path,                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=4,         #6  rakha tha out of memory ho gaya
                                                class_mode='categorical') 
    start=time.time()
    
    history=model.fit(                                        
            training_set,
            steps_per_epoch=4000, # No of images in training set  #12841 tha ()   #680
            epochs=6,           #5 tha   #10
            validation_data=test_set,
            validation_steps=500)# No of images in test set #4268 tha  #140
    
    end=time.time()
    print("Time elasped is : ",end-start,"seconds")
    print((end-start)/60,"minutes")
    
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,7)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss VGG19')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    acc_train = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1,7)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy VGG19')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    

if temp=='6':
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
    
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(29, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # Code copied from - https://keras.io/preprocessing/image/
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(sz, sz),
                                                     batch_size=3,               #16
                                                     class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(test_path,                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=4,        #16
                                                class_mode='categorical') 
    
    # train the model on the new data for a few epochs
    
    start=time.time()
    
    history=model.fit(                                        
            training_set,
            steps_per_epoch=4000, # No of images in training set  #12841 tha ()   #680
            epochs=6,           #5 tha   #10
            validation_data=test_set,
            validation_steps=100)# No of images in test set #4268 tha  #140
    
    end=time.time()
    print("Time elasped is : ",end-start,"seconds")
    print((end-start)/60,"minutes")
    
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,7)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss (InceptionV3)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    acc_train = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1,7)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy (InceptionV3)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
if temp=='8':
    
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 29 classes
    predictions = Dense(29, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
        
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(sz, sz),
                                                     batch_size=3,               #10
                                                     class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(test_path,                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=4,        #10
                                                class_mode='categorical') 
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # train the model on the new data for a few epochs
    start=time.time()
    
    model.fit(
        training_set,
        steps_per_epoch=4000, # No of images in training set  #12841 tha ()   #680
        epochs=2,           #5 tha   #10
        validation_data=test_set,
        validation_steps=500)# No of images in test set #4268 tha  #140
    
    end=time.time()
    print("Time elasped is : ",end-start,"seconds")
    print((end-start)/60,"minutes")    
    
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.
    
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)
    
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True
    
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from tensorflow.keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    # yhn   par loss acc sab likhna h
    
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    
    start=time.time()
    
    model.fit(
        training_set,
        steps_per_epoch=4000, # No of images in training set  #12841 tha ()   #680
        epochs=6,           #5 tha   #10
        validation_data=test_set,
        validation_steps=500)# No of images in test set #4268 tha  #140
    
    end=time.time()
    print("Time elasped is : ",end-start,"seconds")
    print((end-start)/60,"minutes")




if temp=='7':
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
   

    model = ResNet50(weights='imagenet')

    # add a global spatial average pooling layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(200, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(29, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # Code copied from - https://keras.io/preprocessing/image/
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(sz, sz),
                                                     batch_size=3,               #10
                                                     class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(test_path,                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=4,        #10
                                                class_mode='categorical') 
    
    # train the model on the new data for a few epochs
    
    start=time.time()
    
    history=model.fit(                                        
            training_set,
            steps_per_epoch=4000, # No of images in training set  #12841 tha ()   #680
            epochs=6,           #5 tha   #10
            validation_data=test_set,
            validation_steps=500)# No of images in test set #4268 tha  #140
    
    end=time.time()
    print("Time elasped is : ",end-start,"seconds")
    print((end-start)/60,"minutes")
    
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,7)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss (ResNet50)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    acc_train = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1,7)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy (ResNet50)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

   

    
    
    
# Saving the model
   
if temp=='1':   # CNN model to be saved ANN just for display
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\CNN1(2)")
    print(os.getcwd())
    model_json = classifier.to_json()
    with open("model-bw1.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    classifier.save_weights('model-bw1.h5')
    print('Weights saved')
    
if temp=='2':   # CNN model to be saved ANN just for display
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\CNN2(2)")
    print(os.getcwd())
    model_json = classifier.to_json()
    with open("model-bw2.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    classifier.save_weights('model-bw2.h5')
    print('Weights saved')
    
if temp=='3':   # CNN Conv 3
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\CNN3(2)")
    print(os.getcwd())
    model_json = classifier.to_json()
    with open("model-bw32.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    classifier.save_weights('model-bw32.h5')
    print('Weights saved')
    
if temp=='4':   # ANN
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\ANN(2)")
    print(os.getcwd())
    model_json = ann.to_json()
    with open("model-bwA.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    ann.save_weights('model-bwA.h5')
    print('Weights saved')
    
if temp=='5':   
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\VGG19_2")
    print(os.getcwd())
    model_json = model.to_json()
    with open("model-bwVGG192.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    model.save_weights('model-bwVGG192.h5')
    print('Weights saved')

if temp=='6':   
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\InceptionV3_2")
    print(os.getcwd())
    model_json = model.to_json()
    with open("model-bwInceptionV32.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    model.save_weights('model-bwInceptionV32.h5')
    print('Weights saved')
    
if temp=='7' or temp=='8':   
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\Resnet502")
    print(os.getcwd())
    model_json = model.to_json()
    with open("model-bwResnet.json", "w") as json_file:   #INception v3 ka 8 wala hai isme.
        json_file.write(model_json)
    print('Model Saved')
    model.save_weights('model-bwResnet.h5')
    print('Weights saved')
