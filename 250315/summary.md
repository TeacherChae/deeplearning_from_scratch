# Chapter4. 신경망 학습

일반적인 머신 러닝(Machine Learning)과 딥 러닝(Deep Learning)이 구분되는 가장 핵심적인 부분이 바로 '스스로' 학습한다는 점이다. 이번 장은 기계가 스스로 학습한다는 것이 무엇인지, 어떤 수학적 원리가 이를 가능케 하는지 알아보도록 한다.

## 손실 함수

결국 기계학습에 있어서 성패는 편향과 가중치, 즉 매개변수를 얼만큼 적절하게 조절하냐에 달려있다. 매개변수의 최적값을 찾아나가는 도구가 바로 '손실 함수'이다.

### 오차제곱합(sum of squares for error, SSE)

$$
SEE = \sum_{i=1}^{n} (y_i - \hat{y_i})^2
$$

$y_i$는 신경망의 출력값(소프트 맥스 함수의 출력값 = 확률), $\hat{y_i}$는 정답 레이블을 의미한다.

직관적으로 이해하면 **실제값과 예측값의 차이** 정도 된다. 다만 음수가 나오는 것을 막기 위해 제곱을 하는 정도?

비교하는 데이터셋이 많아지면 이를 다음과 같이 일반화 시킬 수 있다.
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
$$


0 < $y_i - \hat{y_i}$ < 1인 경우엔 제곱이 원래 값보다 작아지기 때문에 이를 방지하기 위해 제곱이 아닌 절댓값을 씌우는 MAE 방법도 있다고 하더라.

### 교차 엔트로피 오파(cross entropy error, CEE)

$$
CEE = -\sum_{i=1}^{n} t_i \log y_i
$$

변수의 의미는 동일하나 SSE와는 달리 **tk = 1일 때**의 값만 취급한다는 차이점이 있다.

아래는 밑이 e인 로그함수의 그래프인데,
![](https://velog.velcdn.com/images/brianc1213/post/fc21244c-92c0-4982-ae96-db067d549a5c/image.png)
0 < x < 1에서 음수값을 가지므로 부호를 바꿔주면, **정답인 tk에 해당하는 yk값이 작을 때**(=확률이 낮을 때) 오차가 커지는 모습을 확인할 수 있다.

데이터가 여러 셋일 때 수식을 일반화해보면 다음과 같다.
$$
E = -\frac{1}{N} \sum_{n} \sum_{k} t_{nk} \log y_{nk}
$$

이에 따라 데이터의 양에 상관없이 일반화된 수식을 적용할 수 있으나, 실제로 방대한 양의 연산을 컴퓨터에 적용할 때는 보다 효율적인 방법이 요구된다. 바로 전체 표본에서 무작위로 몇개를 뽑아 일부만 학습하는, 즉 '근사치'를 구하는 것이다.

### 미니배치 학습

데이터의 수가 일정 수준 이상으로 많을 때, 전체 데이터를 대상으로 손실 함수를 계산하는 것은 (가능은 하지만) 컴퓨터 연산 측면에서 매우 비효율적이다. 양자 컴퓨팅도 나오는 시대에 비효율이 어딨냐 라고 할 수도 있지만 제프리 힌튼 아저씨가 역전파 알고리즘을 제안한게 무려 40년 전이라는 점을 기억하자. 다른 이유들도 많겠지만 딥러닝 모델의 발전이 더뎠던 이유는 그만큼 막대한 연산을 구현할 수 있는 하드웨어가 그동안 부재했다는 것이다. 실제로 오늘 스터디하면서 친구가 수치 미분으로 시뮬레이션을 돌리니 예상 소요 시간이 24시간이 나왔다고 했다(...)

그래서 데이터의 일부만을 추려 전체의 '근사치'를 구하는 방법이 바로 미니배치 학습이다. 이를 통해 전체 데이터의 손실 함수를 계산하는 것이 아니라, 미니배치의 손실 함수를 계산하고 이를 통해 전체 데이터의 손실 함수를 근사치로 추정할 수 있다.
```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load _mnist
(x _train, t _train), (x _test, t _test) = load _mnist(normalize = True, one _hot _label = True)
print(x _train.shape) # (60000, 784)
print(t _train.shape) # (60000, 10)
```
```python
train_size = x _train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train _size, batch _size)
x _batch = x _train[batch _mask]
t _batch = t _train[batch _mask]
```
```python
np.random.choice(60000, 10)
```
참 이... numpy는 뭐든 다 해주는구나. 이렇게 하면 60000개 중에서 10개를 뽑아낼 수 있다.

### 배치용 교차 엔트로피 오차 구현하기

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```
```python
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
z = cross_entropy_error(y, t)
```
matplotlib을 이용해 그래프로 나타내면 다음과 같다.
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯을 위해 필요

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# 예제 데이터
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
z_value = cross_entropy_error(y, t)

# y, t에 해당하는 각 점에 대해 z값을 동일하게 할당
z = np.full_like(y, z_value)

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 산점도 그리기
ax.scatter(y, t, z, color='b', marker='o')

# 축 레이블 설정
ax.set_xlabel('y')
ax.set_ylabel('t')
ax.set_zlabel('Cross Entropy Error')

plt.show()
```
![](https://velog.velcdn.com/images/brianc1213/post/657245f1-3223-46c7-beae-a924399ca9a5/image.png)

근데 어쩐지 좀 심심하다.
gpt한테 물어보니, 이진 교차 엔트로피 오차(binary cross entropy error) 수식
$$
CE(p,t)=−(tlog(p)+(1−t)log(1−p))
$$
를 쓰면 더 재밌을 것 같다고 한다. 그래서 다음과 같이 코드를 수정했다.
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯 기능 활성화

def binary_cross_entropy_error(p, t):
    epsilon = 1e-7  # log(0) 방지를 위한 작은 값
    return - (t * np.log(p + epsilon) + (1 - t) * np.log(1 - p + epsilon))

# 예측 확률 p와 타깃 t의 범위 설정
p = np.linspace(0.001, 0.999, 100)  # 0과 1에 가까운 값은 log에서 문제를 피하기 위해 약간의 여유를 둔다.
t = np.linspace(0, 1, 100)
P, T = np.meshgrid(p, t)

# 각 (p, t) 조합에 대한 binary cross entropy error 값을 계산.
Z = binary_cross_entropy_error(P, T)

# 3D 서피스 플롯 생성
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P, T, Z, cmap='viridis', edgecolor='none')

# 축 레이블 및 제목 설정
ax.set_xlabel('Predicted Probability (p)')
ax.set_ylabel('Target (t)')
ax.set_zlabel('Binary Cross Entropy Error')
ax.set_title('Surface Plot of Binary Cross Entropy Error')

# 컬러바 추가
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
```
어지럽다. 넘어가도록 하자.

정답 레이블이 원-핫 인코딩이 아니라 '2'나 '7' 등의 숫자 레이블로 주어졌을 때의 교차 엔트로피 오차는 다음과 같이 구현할 수 있다.
```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```
여기서 차이점은 t가 원-핫 인코딩일 때랑 달리 숫자 레이블일 때는 정답의 위치를 인덱스로 지정해준다는 것이다. 뭐... 다시 생각해보면 원-핫 인코딩도 결국 같은 말이지만.

중요한 것은 t가 원-핫인지, 숫자 레이블인지는 기계가 판단하는 것이 아닌 **사람이 판단해야한다**는 점이다. 좀 더 정확히 표현하면, t가 원-핫인지, 숫자 레이블인지에 따라 위의 두 연산법을 *사람이 구분해서 지정해줘야한다*는 것이다. 그러니깐, t가 원-핫인 데이터를 가지고 백날 숫자 레이블용 연산을 하거나, 반대로 원-핫인 척하는 숫자 레이블에 원-핫 인코딩용 연산을 해봤자 알고리즘이 이를 알아서 구분할 수 없기 때문에 올바른 결과를 얻을 수 없다는 것이다. 이 설명이 책에는 없어서 한참을 싸매고 있었다.

## 왜 손실 함수를 설정하는가?

손실 함수를 설정하는 이유는 **매개변수의 값을 조정해가며 손실 함수의 값을 가능한 한 작게 만드는 매개변수를 찾기 위함**이다. 이를 위해 미분을 사용하는데, 미분이란 결국 **변화율**을 나타내는 것으로 이를 통해 손실 함수의 값을 줄이는 방향으로 매개변수를 조정할 수 있다.

근데 이해가 잘 안 가는 부분
>신경망을 학습할 때 정확도를 지표로 삼아서는 안 된다. 정확도를 지표로 하면 매개변수
의 미분이 대부분의 장소에서 0이 되기 때문이다.

이건 무슨 말인가?

정확도는 보통 다음과 같이 정의된다고 한다.
$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{\hat{y}_i = y_i\}
$$

여기서 $\mathbf{1}\{\cdot\}$은 참이면 1, 거짓이면 0을 반환하는 함수이다. 이는 **이산적**이기 때문에 함수가 전 구간에서 연속적이지 않은 경우가 대부분이고, 따라서 미분이 불가능하거나 가능하다하더라도 미분 계수는 늘 0이다. 이는 손실 함수로서 적합하지 않다는 것을 의미한다.

수치 미분 파트부터는 다음 편에서 계속하도록 하자.