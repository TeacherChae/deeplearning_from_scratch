## 수치 미분

수치 미분은 수학에서 말하는 **엄밀한 의미**의 미분과는 달리 소수점 자릿수가 너무 많을 경우 실제 값과의 오차가 발생하는 컴퓨터의 특성에서 기인한다. [참고 링크](https://www.youtube.com/watch?v=-GsrYvZoAdA)

기본적인 개념은 사실 다르지 않다. 다만 변량을 극한값이 아닌 적당히 작은 값(예를 들어 $10^-4$)으로 바꿔서 미분계수를 구한다는 것.

그래서 미분 관련 내용은 이정도로 넘어가도록 한다.

#### 파이썬 구현 ####

```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
```
굳이 좌우극한의 평균을 내는 이유는 오차를 줄이기 위함이라고 한다.

## 기울기

기울기는 **모든 변수의 편미분을 벡터로 정리한 것**이다. 예를 들어 $f(x_0, x_1) = x_0^2 + x_1^2$라는 함수가 있다면, 이를 편미분하면 다음과 같다.

$$
\frac{\partial f}{\partial x_0} = 2x_0, \quad \frac{\partial f}{\partial x_1} = 2x_1
$$

이를 벡터로 정리하면 다음과 같다.

$$
(\frac{\partial f}{\partial x_0}, \frac{\partial f}{\partial x_1}) = (2x_0, 2x_1)
$$

이처럼 모든 변수의 편미분을 벡터로 정리한 것을 **기울기**라고 한다.

#### 파이썬 구현 ####

```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad
```

예를 들어 $f(x_0, x_1) = x_0^2 + x_1^2$일 때, 그래프를 시각화하면,

![](https://velog.velcdn.com/images/brianc1213/post/437349c5-298f-47d2-a1bb-05fbd1d19037/image.png)


앞서 말한 기울기를 시각화해보면 다음과 같다.

![](https://velog.velcdn.com/images/brianc1213/post/7b7a6cab-d008-4638-93c5-10e1320ea243/image.png)


결국 기울기가 가리키는 방향은 **각 지점에서 함수의 출력 값을 가장 크게 줄이는 방향**이다. 이 개념은 이어서 **경사 하강법**으로 이어진다.

## 경사 하강법

경사 하강법은 **기울기를 이용해 함수의 최솟값을 찾는 방법**이다. 경사 하강법은 다음과 같은 식으로 나타낼 수 있다.

$$
x_0 = x_0 - \eta \frac{\partial f}{\partial x_0}
$$

이때 $\eta$는 **학습률**이라고 하며, 이는 **하이퍼파라미터**라고 한다. 이는 사용자가 직접 설정해야 하는 값이다.

#### 파이썬 구현 ####

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x
```
step_num을 다 도는 동안 $f(x)$가 최소가 되는 $x$값을 구하고, 만약 step_num을 다 돌기 전에 grad = 0에 도달한다면 더이상의 학습에선 변화가 없다는 의미다.

학습률은 너무 크면 함숫값이 발산하고 너무 작으면 학습이 느려진다. $x^2$의 부호가 양수인 단순한 이차함수를 떠올려보면 쉽게 알수 있다. 이 학습률이 *하이퍼 파라미터*라는 말은 다시 말해 너무 크거나 너무 작은 정도를 기계가 알아서 찾을 수 없고 사람이 지정해줘야한다는 말과 같다.


질문. 근데 return값은 변수 쌍으로 구성된 벡터인데, 기울기의 대소관계는 대수적으로 어떻게 비교할 수 있는가?
