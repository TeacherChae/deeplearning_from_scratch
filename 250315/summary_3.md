# Chapter4. 추가 정리

4장은 이쯤하고 넘어갈랬는데, 5장을 보고 오니 4장의 구현 방법에 대해 제대로 정리를 안해두면 앞으로 힘들겠다 싶어서 추가로 정리를 하려고 한다.

우선 수치 미분 파트 부터.

## Numerical Differentiate
```python
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
```

원래 미분 가능한 함수라면 좌극한에서의 미분계수와 우극한에서의 미분계수가 동일해야 맞겠지만, 수치미분은 부동 소수점 계산 오차 때문에 **적당히 작은** 값으로 극한값을 대신한다. 따라서 좌미분계수와 우미분계수가 다를 위험이 존재하므로, 둘의 평균을 구해 오차를 줄여보고자 **중앙 차분**으로 계산한다.

## 편미분

부끄럽지만, 고등학교 때 편미분 배울 때 대충 넘어갔던지라 적당히 개념만 정리하고자 한다.

> 편미분(偏微分, 영어: partial derivative)은 다변수 함수의 특정 변수를 제외한 나머지 변수를 상수로 간주하여 미분하는 것이다.(위키백과)

예를 들어
$$
z = f(x, y) = x^2 + x*y + y^2
$$
인 함수가 있다고 하자. 이 함수를 matplotlib를 통해 좌표 공간에서 그려보면

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + x*y + y**2
    
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()
```
![](https://velog.velcdn.com/images/brianc1213/post/be668190-3d7b-4cfc-b19b-f7f317185b11/image.png)

이렇게 나온다.


여기서 y를 상수 a로 간주하면 함수식은 다음과 같다.
$$
f_{(a)}(x) = x^2 + ax + a^2
$$

이 함수를 x에 대해 미분하면
$$
\frac{\partial f_{(a)}(x)}{\partial x} = 2x + a
$$
와 같고, 이는 모든 y=a에 대해 적용 가능하므로 이를 일반화하면
$$
\frac{\partial f(x, y)}{\partial x} = 2x + y
$$
을 얻을 수 있다. 이를 변수 $x$에 대한 함수 $f$의 **편미분**이라 부른다.

## Numerical Gradient
$(\frac{\partial f}{\partial x_{0}},\frac{\partial f}{\partial y{0}})$ 처럼 모든 변수의 편미분을 벡터로 정리한 것을 **기울기(gradient)**라고 한다. 마치 좌표쌍 같이 생겨서 위치 벡터처럼 보이나 위치 벡터랑은 다른 개념이라고 하더라.

```python
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 모든 원소가 0이면서 x와 형상이 같은 배열 생성

    for idx in range(x.size): # x.size는 배열 x에 있는 원소의 총 개수. x가 2차원 이상이라도 1차원 인덱스로 각 원소에 접근한다.
        tmp_val = x[idx] # 값 저장

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h) # 중앙 차분
        x[idx] = tmp_val # 값 복원
    
    return grad
```

배열 x의 모든 원소를 순회하면서 각 원소의 미분계수(기울기)를 수치미분으로 구하는 구현법이다. 경사하강법은 바로 이 기울기를 이용해서 함숫값(오차)을 줄이는 방향으로 매개변수를 갱신해 나가는 것인데, 별거 아니지만 수식을 써보면 다음과 같다.
$$
x_0 = x_0 - \eta \frac{\partial f}{\partial x_0} \\
x_1 = x_1 - \eta \frac{\partial f}{\partial x_1}
$$

수식이 갖는 의미는 간단한 이차함수 그래프를 그려보면 쉽게 이해할 수 있다. 물론 실제로 손실함수의 그래프는 이차함수 그래프와는 다르지만, 국소적으론 유사하다고 보자는 것.

![](https://velog.velcdn.com/images/brianc1213/post/3a22f1e8-b9d8-402a-bb4b-1fcd87e04fee/image.png)

기울기가 양수일 동안은 x값이 감소해야, 기울기가 음수일 동안은 x값이 증가해야 함숫값이 감소한다. 간단하다.

좀 더 눈여겨보면, $\eta$ 값에 따라 함수가 발산할 수도, 수렴할 수도, 그닥 변화가 없을 수도 있다.
$f(x) = x^2$ 일 때, 경사법의 수식은 다음과 같다.
$$
x_0 = x_0 - \eta \frac{\partial f}{\partial x_0} = x_0 - 2x_0 \eta = x_0(1 - 2\eta)
$$
여기서 $x_0(1 - 2\eta) < -x_0$ 가 되어버린다면? 대참사다. 함숫값이 끝도 없이 증가해버릴 거다.

![](https://velog.velcdn.com/images/brianc1213/post/07c2c801-3051-45e8-83b7-a92f79e99f20/image.png)

그래서 우선 $\eta < 1$와 같은 식 $\eta$ 값의 범위가 좁혀진다. 그리고 또 하나, 우리는 연산을 무한정 반복할 수 없다. 정해진 연산 횟수 내에 최적의 매개변수 값을 찾아야하는데, 이 매개변수가 갱신되는 정도가 너무 작다면? 이 역시 문제가 된다.

![](https://velog.velcdn.com/images/brianc1213/post/cc8eaafa-cc39-483b-b467-5f94b02da51e/image.png)

정해진 연산 횟수 내에 손실함수의 최솟값에 도달하지 못하기 때문.

그래서 이 $\eta$ 를 적절히 설정해주는 것이 경사 하강법에선 매우 중요한 문제가 된다. 이 $\eta$ 를 **학습률**이라 부르고 사람이 직접 설정해야하는 매개변수이기에 **하이퍼파라미터**라고 한다.
~~(근데 왠지 $\eta$도 학습이 가능할 것 같은데?)~~

## 신경망에서의 경사하강법

경사 하강법을 이용한 신경망 학습의 절차는 다음과 같다.

1. 미니배치
    - 훈련 데이터 중 일부를 무작위로 가져 온다.
2. 기울기 산출
    - 각 가중치 매개변수의 기울기를 구한다.
3. 매개변수 갱신
    - 미니배치의 손실함수 값을 줄이기 위해 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.
4. 반복
    - 1~3의 반복

이때, 데이터를 무작위로 선정해 미니배치를 구성하기 때문에 **확률적 경사 하강법**, 줄여서 **SGD** 라고 부른다.

파이썬으로 2층 신경망을 구현해보자.
```python
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # shape이 (input_size, hidden_size)고, 정규 분포를 이루는 무작위 값을 가진 가중치 생성. 1층의 가중치
        self.params['b1'] = np.zeros(hidden_size) # 편향은 0
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 2층의 가중치
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1 # 1층 연산
        z1 = sigmoid(a1) # 1층 활성화 함수
        a2 = np.dot(z1, W2) + b2 # 2층 연산
        y = softmax(a2) # 출력(분류)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x) # 출력값

        return cross_entropy_error(y, t) # 손실함수 값(즉, 오차)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) # 가중치 W에 따른 손실함수

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 앞서 구현한 numerical_gradient() 함수를 통해 구한 가중치 W1의 손실함수 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
```

근데 다른 분들은 numerical_gradient에서 loss_W가 마치 W에 종속적이지 않은 것처럼 보이는 것이 불편하다고 하셨다. 코드가 스파게티 코드가 될 우려가 있다나...나는 이해가 잘 안간다. 무슨 맥락일까?

어쨌든, 이렇게 구현된 클래스를 가지고 이제 학습 과정을 구현해보자.

```python
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True) # mnist 데이터 셋에서 훈련 데이터, 시험 데이터를 가져온다.

train_loss_list = [] # 각 훈련의 결과인 오차들을 기록하는 리스트

# 하이퍼파라미터
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100 # 미니배치 크기
learning_rate = 0.1

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10) # 입력층은 784(28 x 28)개, 은닉층은 50개(은닉층 노드 수는 임의로 결정한 듯?), 출력층은 10개(0~9까지의 숫자)의 노드를 갖는다.

for i in range(iters_num):
    # 미니배치 목록
    batch_mask = np.random.choice(train_size, batch_size) # 전체 훈련 데이터에서 100개를 무작위로 뽑아낸다. 왜 이름이 마스크인지는 모르겠다.
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch) # 매 반복마다 해당 매개변수의 기울기 계산

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key] # 경사 하강법을 이용해서 기울기를 갱신

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
```

자, 코드 실행 버튼을 누르고 ctrl + shift + esc를 누르면 cpu이용률이 50%를 상회하고 있다.
씻고 온다.
커피를 내린다.
유튜브를 본다.
그래도 끝나질 않는다.
이것이 수치미분법의 큰 단점이다. 매 iteration마다 매개변수들의 기울기를 수치미분법으로 일일이 계산하기 때문에 연산량이 무식하게 많다. 그래서 책의 이미지로 대신 확인하는게 빠르다.

이 뒤는 오버피팅 여부를 확인하기 위해 1epoch당 훈련 데이터와 시험 데이터의 정확도를 비교하는 과정이므로 생략.

---

아까 loss_W가 W에 종속적이지 않은 것처럼 보이는 것이 위험한 이유를 이해해보자면, 코드가 기능적으로는 문제 없이 작동하더라도, 람다 함수가 인자 W를 명시적으로 사용하지 않는 구조 때문에 코드의 의존 관계가 불투명해지고, 나중에 코드 수정이나 확장 시 예상치 못한 버그나 혼란을 야기할 수 있다는 우려가 아닐까 싶다. 이는 결국 코드의 가독성과 유지보수성을 저해할 수 있으므로, 보다 명시적으로 인자를 사용하여 의존 관계를 드러내는 방식으로 개선하는 것이 바람직하다는 논지로 이해된다.

이제 4장의 내용이 어느 정도 감이 온다. 내일은 5장을 정리해봐야지.