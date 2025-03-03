import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Definizione di un blocco residuo
def residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.3):
    shortcut = x
    # Primo strato convoluzionale
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # Secondo strato convoluzionale
    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    # Se la dimensione del canale cambia o se lo stride non è 1, adatta il shortcut
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    # Somma e attivazione finale
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    return x

# Costruzione del modello con Functional API
input_shape = (48, 48, 1)
inputs = Input(shape=input_shape)

# Primo blocco: Convoluzione base
x = Conv2D(32, (3,3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.3)(x)

# Blocco residuo con 64 filtri
x = residual_block(x, 64, stride=1, dropout_rate=0.3)
x = MaxPooling2D(pool_size=(2,2))(x)

# Blocco residuo con 128 filtri
x = residual_block(x, 128, stride=1, dropout_rate=0.3)
x = MaxPooling2D(pool_size=(2,2))(x)

# Flatten e fully connected
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(7, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# Configurazione dei Data Generators
train_dir = os.path.join('..', 'data', 'train')
test_dir = os.path.join('..', 'data', 'test')
batch_size = 64
img_size = (48, 48)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Callback: checkpoint, ReduceLROnPlateau e EarlyStopping
model_save_path = os.path.join('..', 'models', 'emotion_model_improved.h5')
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)

steps_per_epoch = max(1, train_generator.samples // batch_size)
validation_steps = max(1, validation_generator.samples // batch_size)
epochs = 40

print("Inizio addestramento...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, reduce_lr, early_stopping],
    verbose=1
)
print("Addestramento completato!")
print("Modello salvato in:", model_save_path)

# Grafico dell'andamento del training
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoca')
plt.xlabel('Epoche')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss per Epoca')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
