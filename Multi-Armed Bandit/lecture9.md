# Lecture 9: Linear Bandits and Thompson Sampling

## 1. Introduction

### 1.1. Linear Bandit Background

Linear Bandit은 agent가 각 round별로 arm을 선택하고 그에 대응되는 확률적인 보상을 받는 문제이다. arm별로 주어지는 확률적 보상의 기대값은 알 수 없는 linear function의 형태를 띈다. agent는 n번의 round동안 누적보상을 최대로 하도록 탐색을 한다. 각 round 별로 agent가 선택한 arm은 $e_i$의 형태로 선택된 arm을 제외한 모든 값을 0으로 표현한다. 결과적으로 각각의 arm은 독립적이고, 보상의 확률을 추정하는 계수는 그 arm에 대해서만 의존적임을 알 수 있다.

Bandit을 OFU(the optimisim in the face of uncertainty) principle을 다루는 문제로 규정하고, 이는 얻은 reward를 바탕으로 결정되는 arm의 보상에 대한 선형 함수의 계수들을 특정 구간안에 유지시킴으로써 exploration-exploitation trade-off를 풀어내는 것이다. 각 Round 별로, 신뢰 구간내에서 선형 함수의 계수를 추정하고 예상 reward를 최대화 하는 방향으로 action을 취한다. 이는 과거 round에서 관측했던 action-reward값을 기반으로 한 선형 함수의 계수를 가지고 신뢰 구간을 구성하면서 risk를 줄여나간다. 

그런데, 해당 문제의 중요점은 미래의 arm이 선택되는 것은 과거 선택에 의해 결정되기 떄문에 arm들이 독립적이지 않다는 것이다. 

### 1.2. Learning Model

* 각 round t 마다, arm $X_t$를 Decision set $D=D_t\subseteq{R^d}$에서 선택하고, arm에 해당하는 reward를 $Y_t=<X_t,\Theta_{*}>+\eta_t$ 만큼 얻는다($\Theta_{*}$는 agent에게 알려지지 않은 parameter이고, $\eta_t$는 random noise이다).

* Agent는 n round를 거치는 동안 누적 보상의 기대값을 최대화($\sum^n_{t=1}<X_t,\Theta_*>$)하는 Action들을 찾아야 한다. 

* 최적의 전략은 round t에 기대되는 즉각적인 보상을 최대화하는 arm을 고르는 것이다. $x^*_t=argmax_{x\in{D_t}<x,\Theta_*>}$

* 최적의 전략으로 얻은 누적 보상의 기대값은 $\sum^n_{t=1}<x_t,\Theta_*>$이다.

Regret은 Bandit의 전략과 최적의 전략의 성능을 비교하기 위해 사용된다.
* expected regret: reward에 대해 모두 다 알고 있는 경우 최적 action에 대한 기대값
* pseudo regret: Bandit의 최적 Reward와 수행한 Action으로 얻은 실제 Reward간의 차이로 얻는 기대값

pseudo regret => regret

그리고, reget의 상한선을 추정하기 위한 몇가지 가정들이 필요
1. round 1부터 inf까지의 Decision set Dt는 한정되어 있어야 한다
2. Error $\eta_t$는 conditionally R이 0이상의 상수로 고정된 R-sub-Gaussian를 따라야 한다. 