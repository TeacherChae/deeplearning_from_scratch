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

# import numpy as np
# import matplotlib.pyplot as plt

# # x는 예측값 (0에서 1 사이)
# x = np.linspace(0, 1, 1000)

# # 임계값 0.5를 기준으로 한 스텝 함수(정확도 함수)
# accuracy = np.where(x > 0.5, 1, 0)

# plt.plot(x, accuracy, label="Accuracy")
# plt.xlabel("예측값 (y)")
# plt.ylabel("정확도")
# plt.title("이진 분류에서의 정확도 함수 (스텝 함수)")
# plt.legend()
# plt.grid(True)
# plt.show()

# 1. Heatmap of Matrix Values

# import numpy as np
# import matplotlib.pyplot as plt

# # 예제 데이터: 10x10 랜덤 행렬
# matrix = np.random.rand(10, 10)

# plt.imshow(matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title('Heatmap of Matrix Values')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')
# plt.show()

# 2. Contour Plot of Matrix Values

# import numpy as np
# import matplotlib.pyplot as plt

# # 예제: 두 매개변수에 대한 단순한 손실 함수 (예: f(x, y) = (x-1)^2 + (y-2)^2)
# x = np.linspace(-3, 3, 100)
# y = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x, y)
# Z = (X - 1)**2 + (Y - 2)**2

# plt.contour(X, Y, Z, levels=20)
# plt.colorbar()
# plt.title('Contour Plot of Sample Loss Function')
# plt.xlabel('Parameter 1')
# plt.ylabel('Parameter 2')
# plt.show()

# 3. Animation of Gradient Descent
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # 예제 손실 함수: f(x, y) = (x-1)^2 + (y-2)^2
# def loss(x, y):
#     return (x - 1)**2 + (y - 2)**2

# # 경사하강법 시뮬레이션
# x, y = -3, -3  # 초기값
# lr = 0.05      # 학습률
# trajectory = [(x, y, loss(x, y))]

# for i in range(50):
#     dx = 2 * (x - 1)
#     dy = 2 * (y - 2)
#     x = x - lr * dx
#     y = y - lr * dy
#     trajectory.append((x, y, loss(x, y)))
# trajectory = np.array(trajectory)

# # 컨투어 플롯 배경 그리기
# x_vals = np.linspace(-3, 3, 100)
# y_vals = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x_vals, y_vals)
# Z = loss(X, Y)

# fig, ax = plt.subplots()
# ax.contour(X, Y, Z, levels=20)
# line, = ax.plot([], [], 'ro-', lw=2)

# def init():
#     line.set_data([], [])
#     return line,

# def update(frame):
#     data = trajectory[:frame+1]
#     line.set_data(data[:, 0], data[:, 1])
#     return line,

# ani = FuncAnimation(fig, update, frames=len(trajectory), init_func=init,
#                     blit=True, repeat=False)
# plt.title('Gradient Descent Animation')
# plt.xlabel('Parameter 1')
# plt.ylabel('Parameter 2')
# plt.show()

# 4. Quiver Plot of Gradient Vectors

# import numpy as np
# import matplotlib.pyplot as plt

# # 예제 손실 함수 및 기울기: f(x, y) = (x-1)^2 + (y-2)^2
# def loss(x, y):
#     return (x - 1)**2 + (y - 2)**2

# def grad(x, y):
#     return 2 * (x - 1), 2 * (y - 2)

# # 격자 생성
# x = np.linspace(-3, 3, 20)
# y = np.linspace(-3, 3, 20)
# X, Y = np.meshgrid(x, y)
# U, V = grad(X, Y)

# plt.figure()
# # 음의 기울기 방향(최소값으로 향하는 방향)을 표시
# plt.quiver(X, Y, -U, -V)
# plt.title('Quiver Plot of Gradient Vectors')
# plt.xlabel('Parameter 1')
# plt.ylabel('Parameter 2')
# plt.show()

# 5. Step-by-step operation visualization

# import numpy as np
# import matplotlib.pyplot as plt

# # 예제 행렬
# A = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])
# B = np.array([[9, 8, 7],
#               [6, 5, 4],
#               [3, 2, 1]])
# C = np.zeros((3, 3))

# fig, ax = plt.subplots()
# heatmap = ax.imshow(C, cmap='viridis', vmin=np.min(A @ B), vmax=np.max(A @ B))
# plt.colorbar(heatmap)
# plt.title('Step-by-step Matrix Multiplication')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')
# plt.ion()  # 인터랙티브 모드 ON
# plt.show()

# # 행렬 곱셈의 각 단계별로 누적합을 시각화
# for i in range(3):
#     for j in range(3):
#         s = 0
#         for k in range(3):
#             s += A[i, k] * B[k, j]
#             C[i, j] = s
#             heatmap.set_data(C)
#             ax.set_title(f'Calculating C[{i}, {j}], k = {k}')
#             plt.pause(0.5)  # 잠깐 멈추며 업데이트
# plt.ioff()  # 인터랙티브 모드 OFF
# plt.show()

