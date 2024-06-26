import os
import tensorflow as tf
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

def to_complex(image):
    rows, cols = image.shape
    complex_image = np.zeros((rows, cols), dtype=complex)
    for x in range(rows):
        for y in range(cols):
            complex_image[x, y] = image[x, y] * (x + 1j * y)
    return complex_image

complex_train_images = np.array([to_complex(img) for img in train_images])
complex_test_images = np.array([to_complex(img) for img in test_images])

def to_polar(image):
    rows, cols = image.shape
    polar_image = np.zeros((rows, cols, 2))
    for x in range(rows):
        for y in range(cols):
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            polar_image[x, y] = [r, theta]
    return polar_image

polar_train_images = np.array([to_polar(img) for img in train_images])
polar_test_images = np.array([to_polar(img) for img in test_images])

save_dir = 'C:/Users/Administrator/Desktop/ComplexNumber_and_Polar_CNN/datasets'

os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, 'complex_train_images.npy'), complex_train_images)
np.save(os.path.join(save_dir, 'complex_test_images.npy'), complex_test_images)

np.save(os.path.join(save_dir, 'polar_train_images.npy'), polar_train_images)
np.save(os.path.join(save_dir, 'polar_test_images.npy'), polar_test_images)

np.save(os.path.join(save_dir, 'base_train_images.npy'), train_images)
np.save(os.path.join(save_dir, 'base_test_images.npy'), test_images)

np.save(os.path.join(save_dir, 'train_labels.npy'), train_labels)
np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)
