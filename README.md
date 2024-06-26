# 복소수와 극좌표를 이용하여 CNN 훈련하기

MNIST 손글씨 숫자 데이터를 복소수, 극좌표 형태로 바꾸어 훈련데이터로 사용했습니다.

복소수의 경우 a+bi 형태로 a에는 데이터의 x(픽셀정보), b에는 y(위치정보)로 사용하여 전처리 했습니다.

극좌표는 반경 r과 각도 세타로 표현되기 때문에 r = sqrt(x^2+y^2), 세타 = arctan(y/x)로 설정하여 전처리 했습니다.

이를 .npy로 저장하여 데이터셋을 구성했습니다. (용량이 2GB 가량되어 이를 삭제한 후 업로드 했습니다.)

극좌표의 경우 세타의 주기성이 있어 이를 CNN이 알아차리도록 하기 위해 입력 정보를  sin, cos으로 입력 했습니다.

# Result

![result](https://github.com/dce9112/complex_polar_CNN/assets/172959778/2955ceec-bb6c-4aa6-a45f-b3c1c674f53b)
