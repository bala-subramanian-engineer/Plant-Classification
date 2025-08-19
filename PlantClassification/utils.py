import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(128, 128)):
   
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    
    class_names = sorted(os.listdir(train_dir))
    num_classes = len(class_names)
    
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(train_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            X_train.append(img)
            y_train.append(class_idx)
    
    # Load validation data
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(val_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            X_val.append(img)
            y_val.append(class_idx)
    
    # Load test data
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            X_test.append(img)
            y_test.append(class_idx)
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val, batch_size=batch_size, shuffle=False
    )
    
    return train_generator, val_generator