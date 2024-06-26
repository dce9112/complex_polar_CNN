from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

base_train_images = np.load("C:/Users/Administrator/Desktop/ComplexNumber_and_Polar_CNN/datasets/base_train_images.npy")
base_test_images = np.load("C:/Users/Administrator/Desktop/ComplexNumber_and_Polar_CNN/datasets/base_test_images.npy")

complex_train_images = np.load("C:/Users/Administrator/Desktop/ComplexNumber_and_Polar_CNN/datasets/complex_train_images.npy")
complex_test_images = np.load("C:/Users/Administrator/Desktop/ComplexNumber_and_Polar_CNN/datasets/complex_test_images.npy")

polar_train_images = np.load("C:/Users/Administrator/Desktop/ComplexNumber_and_Polar_CNN/datasets/polar_train_images.npy")
polar_test_images = np.load("C:/Users/Administrator/Desktop/ComplexNumber_and_Polar_CNN/datasets/polar_test_images.npy")

train_labels = np.load("C:/Users/Administrator/Desktop/ComplexNumber_and_Polar_CNN/datasets/train_labels.npy")
test_labels = np.load("C:/Users/Administrator/Desktop/ComplexNumber_and_Polar_CNN/datasets/test_labels.npy")

def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model_base = create_model((28, 28, 1)) #기본 데이터로 학습
model_complex = create_model((28, 28, 1)) #복소수 데이터로 학습
model_polar = create_model((28, 28, 3)) #극좌표 데이터로 학습

#기본 데이터 학습
base_train_images_expanded = np.expand_dims(base_train_images, axis=-1)
base_test_images_expanded = np.expand_dims(base_test_images, axis=-1)

model_base.fit(base_train_images_expanded, train_labels, epochs=10, validation_split=0.2)

score_base = model_base.evaluate(base_test_images_expanded, test_labels)

#복소수 데이터 학습
complex_train_images_expanded = np.expand_dims(complex_train_images, axis=-1)
complex_test_images_expanded = np.expand_dims(complex_test_images, axis=-1)

model_complex.fit(complex_train_images_expanded, train_labels, epochs=10, validation_split=0.2)

score_complex = model_complex.evaluate(complex_test_images_expanded, test_labels)

#극좌표 데이터 학습
def convert_polar_to_cartesian(polar_images): #각도를 sin과 cos값으로 변경
    r = polar_images[..., 0]
    theta = polar_images[..., 1]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.stack([r, sin_theta, cos_theta], axis=-1)

polar_train_images_cartesian = convert_polar_to_cartesian(polar_train_images)
polar_test_images_cartesian = convert_polar_to_cartesian(polar_test_images)

model_polar.fit(polar_train_images_cartesian, train_labels, epochs=10, validation_split=0.2)

score_polar = model_polar.evaluate(polar_test_images_cartesian, test_labels)

print(f'기본 데이터 테스트 정확도 : {score_base[1]}\n')
print(f'복소수 데이터 테스트 정확도 : {score_complex[1]}\n')
print(f'극좌표 데이터 테스트 정확도 : {score_polar[1]}')

# 모델들의 정확도를 리스트에 저장
accuracies = [score_base[1], score_complex[1], score_polar[1]]
models = ['Base Model', 'Complex Model', 'Polar Model']

# 정확도 시각화
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Test Accuracy of Different Models')
plt.ylim([0, 1.0])  # 정확도 범위 지정
plt.grid(True)
plt.show()