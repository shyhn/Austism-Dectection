
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np 
from keras.preprocessing import image




classifier = Sequential()
classifier.add(Conv2D(32, 3, 3, input_shape = (224, 224, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(32, 3, 3, activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2))) 
classifier.add(Flatten())
classifier.add(Dense(224, activation="relu"))
classifier.add(Dense(1, activation="sigmoid"))
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics= ["accuracy"])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(224,224), 
        batch_size=32,
        class_mode='binary'
        ) 

test_set = test_datagen.flow_from_directory('test',
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='binary'
                                                ) 


cnn = classifier.fit(
            training_set,
            steps_per_epoch= int(2536/32), 
            epochs=100, 
            validation_data=test_set,
            validation_steps= int(300/32), 
            workers = 4) 


accuracy = cnn.history['accuracy']
accuracy = sum(accuracy)/len(accuracy)
accuracy



val_accuracy = cnn.history['val_accuracy']
val_accuracy = sum(val_accuracy)/len(val_accuracy)
print(val_accuracy)

val_loss = cnn.history['val_loss']
val_loss = sum(val_loss)/len(val_loss)
print(val_loss)


loss = cnn.history['loss']
loss = sum(loss)/len(loss)
print(loss)



accuracy = cnn.history['accuracy']
val_accuracy = cnn.history['val_accuracy']
loss = cnn.history['loss']
val_loss = cnn.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()




test_image = image.load_img('valid/non_autistic/22.jpg' , target_size = (224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)  #Car une image doit faire partie d'un batch (ce qui donne un élément à 4 dimensions)
result = classifier.predict(test_image)


training_set.class_indices


print(result)                   #     0:autistic,1:non_autistic



