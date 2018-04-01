from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import tensorflow as tf



batch_size = 64

nb_train_samples = 3644

nb_validation_samples = 1214

nb_test_samples = 6
datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(64, 64),
    batch_size=64,
    class_mode='categorical')

val_generator = datagen.flow_from_directory(
    'data/validation',
    target_size=(64, 64),
    batch_size=1,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    'data/test',
    target_size=(64, 64),
    batch_size=1,
    class_mode='categorical')

model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
    
model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#model.load_weights("weights1.h5")
#

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

unit_to_multiplier = {
    0: 'armor',
    1: 'Bijouterie',
    2: 'Food',
    3: 'Mobs',
    4: 'Potions',
    5: 'Weaponsmelee',
    6: 'Weaponsrange'
}

scores = model.evaluate_generator(val_generator, nb_validation_samples // 1)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

#j=0;
#s=val_generator.index_array
#for z in s:
#    print(val_generator.filenames[z]+' '+unit_to_multiplier[np.argmax(model.predict_generator(val_generator)[j])])
#    j = j + 1

#model.save_weights("weights1.h5")