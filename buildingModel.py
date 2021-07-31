# build a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(150, 150, 3)))

# 1st conv block
model.add(Conv2D(64, (3,3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
# 2nd conv block
model.add(Conv2D(128, (3,3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
# 3rd conv block
model.add(Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
# 4th conv block
model.add(Conv2D(512, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(Conv2D(512, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
# ANN block
model.add(Flatten())
model.add(Dense(units=5000, activation='relu'))
model.add(Dense(units=3000, activation='relu'))
# output layer
model.add(Dense(units=1264, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
# fit on data for 50 epochs
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))