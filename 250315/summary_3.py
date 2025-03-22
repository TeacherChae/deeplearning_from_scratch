# import numpy as np
# import matplotlib.pyplot as plt

# def f(x, y):
#     return x**2 + x*y + y**2
    
# x = np.linspace(-3, 3, 100)
# y = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # 파라미터 a와 접선을 그릴 x위치 지정
# x0 = -3  # 접선을 그릴 위치

# # 함수 정의: f(x) = x^2 + ax + a^2
# def f(x):
#     return x**2

# # f'(x) = 2x + a (미분)
# def df(x):
#     return 2*x

# # 접선 정의: y = f(x0) + f'(x0)(x - x0)
# def tangent_line(x):
#     return f(x0) + df(x0) * (x - x0)

# # x 범위 설정
# x = np.linspace(-5, 5, 300)
# y = f(x)
# y_tangent = tangent_line(x)

# # 그래프 그리기
# plt.figure(figsize=(8, 5))
# plt.plot(x, y, label='f(x)', color='blue')
# plt.plot(x, y_tangent, '--', label='Tangent at x={}'.format(x0), color='orange')
# plt.scatter([x0], [f(x0)], color='red', zorder=5, label='Point of tangency')
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.xlim(-5, 5)     # x축을 -5 ~ 5로 고정
# plt.ylim(-5, 25)    # y축을 -5 ~ 25로 고정
# plt.legend()
# plt.title('Quadratic Function and Tangent Line')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 함수 및 도함수 정의
def f(x):
    return x**2

def df(x):
    return 2*x

# 경사하강법 수식
def gradient_step(x, eta):
    return x * (1 - 2 * eta)

# 파라미터
eta = 1.1       # 발산을 유도할 큰 학습률
x0 = 1.0
steps = 8

# 반복 수행
x_vals = [x0]
f_vals = [f(x0)]

for _ in range(steps):
    x_new = gradient_step(x_vals[-1], eta)
    x_vals.append(x_new)
    f_vals.append(f(x_new))

# 전체 곡선 범위 준비
x_margin = max(np.abs(x_vals)) * 1.4
x_plot = np.linspace(-x_margin, x_margin, 500)
y_plot = f(x_plot)

# 시각화 시작
plt.figure(figsize=(12, 7))

# 함수 그래프
plt.plot(x_plot, y_plot, label=r'$f(x) = x^2$', color='lightblue', linewidth=2)

# 점 및 접선 그리기
for i, x_n in enumerate(x_vals):
    y_n = f(x_n)
    slope = df(x_n)

    # 접선 그리기
    x_tangent = np.linspace(x_n - 1.5, x_n + 1.5, 50)
    y_tangent = y_n + slope * (x_tangent - x_n)
    plt.plot(x_tangent, y_tangent, color='orange', linestyle='--', alpha=0.6)

    # 점 찍기
    plt.scatter(x_n, y_n, color='red', zorder=5)

# 점선 경로와 화살표 추가
for i in range(len(x_vals) - 1):
    x_start, y_start = x_vals[i], f_vals[i]
    x_end, y_end = x_vals[i+1], f_vals[i+1]

    # 점선
    plt.plot([x_start, x_end], [y_start, y_end], color='red', linestyle='--', alpha=0.5)

    # 화살표
    plt.annotate('', 
                 xy=(x_end, y_end), 
                 xytext=(x_start, y_start),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# 스타일링
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# ✅ y축을 넓게 설정해 발산을 극적으로 보여줌
plt.ylim(0, max(f_vals) * 1.5)

plt.title(r'Gradient Descent Diverging with $\eta = 1.1$: Function Value Exploding', fontsize=15, weight='bold', color='darkred')
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # 함수 및 도함수 정의
# def f(x):
#     return x**2

# def df(x):
#     return 2*x

# # 경사하강법 수식
# def gradient_step(x, eta):
#     return x * (1 - 2 * eta)

# # 파라미터
# eta = 0.01       # 매우 작은 학습률
# x0 = 1.0         # 시작점
# steps = 15       # 반복 횟수

# # 반복 수행
# x_vals = [x0]
# f_vals = [f(x0)]

# for _ in range(steps):
#     x_new = gradient_step(x_vals[-1], eta)
#     x_vals.append(x_new)
#     f_vals.append(f(x_new))

# # 전체 곡선 범위 준비
# x_min = -0.2
# x_max = 1.2
# x_plot = np.linspace(x_min, x_max, 500)
# y_plot = f(x_plot)

# # 시각화 시작
# plt.figure(figsize=(10, 6))

# # 함수 곡선
# plt.plot(x_plot, y_plot, label=r'$f(x) = x^2$', color='lightblue')

# # 각 점, 접선, 이동 경로 표시
# for i in range(len(x_vals) - 1):
#     x_n = x_vals[i]
#     y_n = f(x_n)
#     slope = df(x_n)

#     # 접선 그리기
#     x_tangent = np.linspace(x_n - 0.2, x_n + 0.2, 50)
#     y_tangent = y_n + slope * (x_tangent - x_n)
#     plt.plot(x_tangent, y_tangent, color='orange', linestyle='--', alpha=0.5)

#     # 현재 점
#     plt.scatter(x_n, y_n, color='red', zorder=5)

#     # 다음 점
#     x_next = x_vals[i + 1]
#     y_next = f(x_next)
#     plt.scatter(x_next, y_next, color='red', zorder=5)

#     # 점선 & 화살표 (이동)
#     plt.plot([x_n, x_next], [y_n, y_next], color='red', linestyle='--', alpha=0.4)
#     plt.annotate('',
#                  xy=(x_next, y_next),
#                  xytext=(x_n, y_n),
#                  arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

# # 마지막 점
# plt.scatter(x_vals[-1], f_vals[-1], color='red', zorder=5)

# # 스타일
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.title(r'Slow Convergence: Tiny $\eta = 0.01$ Cannot Reach Minimum in Limited Steps')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid(True)
# plt.legend(['$f(x)$', 'Tangent line', 'Descent steps'], loc='upper left')
# plt.show()

