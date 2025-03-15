# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯을 위해 필요

# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y + 1e-7)) / batch_size

# # 예제 데이터
# y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
# t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
# z_value = cross_entropy_error(y, t)

# # y, t에 해당하는 각 점에 대해 z값을 동일하게 할당
# z = np.full_like(y, z_value)

# # 3D 플롯 생성
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 산점도 그리기
# ax.scatter(y, t, z, color='b', marker='o')

# # 축 레이블 설정
# ax.set_xlabel('y')
# ax.set_ylabel('t')
# ax.set_zlabel('Cross Entropy Error')

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯 기능 활성화

# def binary_cross_entropy_error(p, t):
#     epsilon = 1e-7  # log(0) 방지를 위한 작은 값
#     return - (t * np.log(p + epsilon) + (1 - t) * np.log(1 - p + epsilon))

# # 예측 확률 p와 타깃 t의 범위를 설정합니다.
# p = np.linspace(0.001, 0.999, 100)  # 0과 1에 가까운 값은 log에서 문제를 피하기 위해 약간의 여유를 둡니다.
# t = np.linspace(0, 1, 100)
# P, T = np.meshgrid(p, t)

# # 각 (p, t) 조합에 대한 binary cross entropy error 값을 계산합니다.
# Z = binary_cross_entropy_error(P, T)

# # 3D 서피스 플롯 생성
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(P, T, Z, cmap='viridis', edgecolor='none')

# # 축 레이블 및 제목 설정
# ax.set_xlabel('Predicted Probability (p)')
# ax.set_ylabel('Target (t)')
# ax.set_zlabel('Binary Cross Entropy Error')
# ax.set_title('Surface Plot of Binary Cross Entropy Error')

# # 컬러바 추가
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# x는 예측값 (0에서 1 사이)
x = np.linspace(0, 1, 1000)

# 임계값 0.5를 기준으로 한 스텝 함수(정확도 함수)
accuracy = np.where(x > 0.5, 1, 0)

plt.plot(x, accuracy, label="Accuracy")
plt.xlabel("예측값 (y)")
plt.ylabel("정확도")
plt.title("이진 분류에서의 정확도 함수 (스텝 함수)")
plt.legend()
plt.grid(True)
plt.show()
