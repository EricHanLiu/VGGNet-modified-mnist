


class VGG:
    def __init__(self, input_shape, num_classes, optimizer):
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))


model = VGG(128*128, 10, Adam())

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()
lr_scheduler = LearningRateScheduler(lr_schedule)
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=4,min_lr=0.5e-9)
lr_reducer = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5)
early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
callbacks = [lr_reducer, lr_scheduler, early_stop]

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                    validation_data=(x_val, y_val))
