# A Survey on Contextual Multi-armed Bandits

## 1. Introduction

<div align="center">
<img src="imgs/Survey_of_Multi-armed_Bandit_uncertainty_senarios.jpg" />
</div>

 의사결정과정에서, agent들은 관측지를 바탕으로 결정을 한다. 위의 테이블은 불확실한 상황의 네가지 케이스에서 유형별로 어떤 형태를 활용하여 의사결정하는지를 보여준다. multi-armed bandit(MAB)의 경우에는 모델의 결과는 알 수 없지만 결과는 확률적이거나 적대적이다. 게다가 agent의 행동은 주변의 상태를 변화시키지 않는다.

 해당 paper는 MAB만을 중점적으로 다룬다. 이런 의사결과정에서 Agent는 1,2,3,...,T에 이어지는 결정을 해야하고 각 시점 t의 agent는 K개의 arm들이 있다. arm을 당기고 난 후에, 해당 arm에 대한 보상이 주어지고, 다른 것들의 보상은 모르는 상태이다. 특정 arm의 stochastic setting은 임의의 모르는 분포를 기준으로 얻으며, adversarial setting된 arm의 경우 적대적으로 선택되어지고 어느 분포에서 샘플링될 필요는 없다. 해당 paper에서 관심을 가지고 있는 것은 t 시점에서 얻을 수 있는 side information이다. 이를 'side information the context'라고 부른다. 가장 높은 보상을 줄것으로 기대되는 arm은 different context를 줄 수 있을 것이다. 이러한 multi-armed bandit을 "contextual bandits"이라고 한다.

 contextual bandit문제에서 정책들의 set과 각각의 정책은 arm의 context와 매핑된다. bandit을 통한 분류문제를 해결하는데 있어서 특정 수의 정책들이 주어진다. 그리고, 어느 정책에서도 가장 많은 누적 보상의 기대값을 주는것과 에이전트가 실제로 얻을 수 있는 보상의 차이를 "regret"으로 정의한다. agent의 목표는 regret을 최소화 하는 것이다. C-MAB는 많은 문제에 적용 가능하다. 

e.g) 뉴스 추천 - news 기사: arm; article과 user의 feature - context; Agent는 각각의 유저에 대해 CTR 또는 dwell time을 최대화 하기위한 방식으로 선택을 하게 될 것이다.

Bandit Algorithm의 종류는 다양하다.
* K-armed bandit
* contextual bandit: 기존에 정의된 expert(pre-defined policy)와 가장 높은 기대 보상을 얻을 수 있도록 경쟁하여 학습하는 방식

<div align="center">
<img src="imgs/Survey_of_Multi-armed_Bandit_contextual-bandits.jpg"/>
</div>

* C: distinct context의 수
* N: policy의 수
* K: arm의 수
* d: context의 dimension
* require the knowledge of T: proposed regret을 달성하기 위해 knowledge T가 필요한지 여부

## 2. Unbiased Reward Estimator
Bandit problem은 partial feedback에 대해서만 관측 가능하다는 것이다. t라는 시점에 알고리즘이 무작위로 a_t를 p_t의 확률로 당겼다고 하자. 진짜 보상은 ![formular](https://render.githubusercontent.com/render/math?math=r_t\in{[0,1]^K})으로 나타내고, 관측된 보상은 ![formular](https://render.githubusercontent.com/render/math?math=r^{\prime}_t\in{[0,1]^K})으로 나타내자. 모든 element r^{\prime}_t에 대해 r^{\prime}_{t,a_{t}}를 제외하면 0이 되고 이는 r_{t,a_t}와 동일하다. r^{\prime}_t는 E(r^{\prime}_{t,a_t})=p_{a_t}*r_{t,a_t}\neq{r_t,{a_t}} 때문에 r_t의 unbiased estimator가 아니다. r^{\prime}_{a_t} 대신에, \hat{r}_{t,a_t}=r^{\prime}_{t,a_t}/p_{a_t} track을 사용한다. 이를 통해 true reward vector r_t의 unbiased estimator를 얻어낼 수 있다.

<img src="https://latex.codecogs.com/svg.latex?E(\hat{r}_t,a)=p_a*r_{t,a}/p_a+(1-p_a)*0" title="E(\hat{r}_t,a)=p_a*r_{t,a}/p_a+(1-p_a)*0"/>
<img src="https://latex.codecogs.com/svg.latex?=r_{t,a}" title="=r_{t,a}"/>

평균 값은 특정 시간 t에 무작위로 선택한 것을 나타내며, 이는 이후의 알고리즘들에도 많이 사용된다.

## 3. Reduce to K-Armed Bandits

모든 context들을 돌아볼 수 있다고 가정한다면, 가장 적용하기 쉬은 방법은 K-armed bandit argorithm을 각각의 context에 활용해 보는 것이다. 그러나 각각의 context가 독립적이라고 사전에 다루기 때문에 모든 컨텍스트 간의 관계를 고려할 수 없다.
C개의 distinct한 context를 set X로 나타낸다면, t 시점의 가능한 컨텍스트는 x_t\in{\{1,2,...,C\}}로 나타낼 수 있다. 또한 K개의 Arm들의 집합인 A에 대해서도 t시점에 a_t\in{\{1,2,...,K\}}로 나타낼 수 있다. 모든 가능한 Context와 Arm의 매핑에 대해 \prod = {f:X\rightarrow{A}}로 나타낼 수 있으며, agent의 regret은 아래와 같의 정의된다.

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?R_T=\underset{f\in{\prod}}{sup}E[\overset{T}{\underset{t=1}{\sum}}(r_{t,f(x_t)}-r_{t,a_t})]" title="R_T=\underset{f\in{\prod}}{sup}E[\overset{T}{\underset{t=1}{\sum}}(r_{t,f(x_t)}-r_{t,a_t})]"/>

<img src="imgs/Survey_of_Multi-armed_Bandit_EXP3.jpg" />
<p>EXP3, non-contextual multi-armed bandit algorithm for K-armed bandit</p>
</div>

이 방법의 문제 중 하나는 true가 아닌 context들에 대해서도 손회를 한다고 가정한다는 것이다. 또한 각각의 context들이 독립적이라고 다루기 때문에, context들 중에 하나를 학습하는 것은 다른 것들을 학습하는데 도움이 되지 않는다. 

만약 사전에 정의된 policy들이 있고 그것 중 최선의 policy와 경쟁하고자 한다면, 각각의 policy를 arm으로 하여 EXP3 algorithm을 적용하는 것이다. \prod는 context들에서 arm들에 대해 가능한 모든 매핑들을 활용하는 것 대신에 선언한 pre-defined policy이다. N을 policy들의 집합이라고 하면, EXP3 알고리즘의 regret은 O(\sqrt{TNlnN})이다.

이는 작은 수의 정책과 많은 수의 arm에 대해서만 가능하기 때문에, 이 경우가 아니라면 regret bound가 약하다.

## 4. Stochastic Contextual Bandits

Stochastic Contextual Bandit은 각각의 arm에 대한 보상이 알려지지 않은 확률분포를 따른다고 가정한다. 더 나아간 몇몇의 알고리즘들은 sub-Gaussian 분포를 따른다고 가정하기도 한다. 우선, linear realizability assumption을 따르는 상황에서의 stochastic contextual bandit에 대해 알아보자. 이 경우, 각각의 arm에 대한 보상의 기대값은 선형적이다. 그렇다면, 이러한 가정이 없는 임의의 policy들에 대한 algorithm을 알아보자.

### 4.1. Stochastic Contextual Bandits with Linear Realizabilty Assumption

### 4.1.1. LinUCB / SupLinUCB

LinUCB는 contextual case에 대해 확장한 UCB 알고리즘이다. 각각의 arm은 feature vector x_{t,a}\in{R^d}와 연관이 있다고 가정한다. e.g) 뉴스 기사 예측에서 x_{t,a}는 user-article간의 pairwise feature vector라고 할 수 있다. LinUCB는 하나의 arm a에 대해 기대되는 보상은 선형적이라고 가정한다. 

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?R_T=E[r_{t,a}|X_{t,a}]=x^T_{t,a}\Theta^{*}" title="R_T=E[r_{t,a}|X_{t,a}]=x^T_{t,a}\Theta^{*}"/>
</div>

\Theta^{*}는 true coefficient vector이며, 노이즈 \epsilon_{t,a}는 임의의 t시점에 대해 R-sub-Gaussian분포를 따른다고 가정한다. 그리고 ||\Theta^{*}|| \leq S를 따르고, ||x_{t,a}|| \leq L을 따른다고 가정한다. 또한 한번의 시행으로 인해 얻는 reward r_{t,a} \leq 1이다.

t시점의 최선의 arm은 a^{*}_t=argmax_{a}x^{T}_{t,a}\Theta^{\prime}이며, a_t는 t시간에 알고리즘에 의해 선택되는 arm을 의미한다.

<img src="https://latex.codecogs.com/svg.latex?R_T=E[\overset{T}{\underset{t=1}{\sum}}r_{t,a^*_t}-\overset{T}{\underset{t=1}{\sum}}r_{t,a_t}]" title="R_T=E[\overset{T}{\underset{t=1}{\sum}}r_{t,a^*_t}-\overset{T}{\underset{t=1}{\sum}}r_{t,a_t}]"/>

<img src="https://latex.codecogs.com/svg.latex?=\overset{T}{\underset{t=1}{\sum}}x^T_{t,a^*_t}\Theta^{*}-\overset{T}{\underset{t=1}{\sum}}x^T_{t,a_t}\Theta^{*}" title="\overset{T}{\underset{t=1}{\sum}}x^T_{t,a^*_t}\Theta^{*}-\overset{T}{\underset{t=1}{\sum}}x^T_{t,a_t}\Theta^{*}"/>

D_t\in{R^{t\times{d}}} 와 c_t\in{R^t}를 임의의 시간 t에 대한 historical data라고 한다면, D_t의 i번째 row는 i시점에 당겨진 arm의 feature vector를 나타내고, c_t의 i번째 row는 그에 상응하는 reward를 뜻한다. 만약 표본 (x_{t,a},r_{r,a_t})이 독립적이라면, ridge regression에 의한 \Theta^{*}의 estimator를 아래와 같이 표현 가능하다.

<img src="https://latex.codecogs.com/svg.latex?\hat{\Theta}_t=(D^T_tD_t+\lambda{I}_d)^{-1}D^T_tc_t" title="\hat{\Theta}_t=(D^T_tD_t+\lambda{I}_d)^{-1}D^T_tc_t"/>

estimator의 정확도 또한 데이터의 양에 의해 결정되며, prediction x^T_{t,a}\hat{\Theta}_t를 위한 UCB는 아래와 같이 계산된다.

<img src="https://latex.codecogs.com/svg.latex?|x^T_{t,a}\hat{\Theta}_t-x^T_{t,a}\Theta^{*}|\leq(\epsilon+1)\sqrt{x^T_{t,a}A^{-1}_tx_{t,a}}" title="|x^T_{t,a}\hat{\Theta}_t-x^T_{t,a}\Theta^{*}|\leq(\epsilon+1)\sqrt{x^T_{t,a}A^{-1}_tx_{t,a}}"/>

<img src="imgs/Survey_of_Multi-armed_Bandit_LinUCB.jpg"/>

그러나 LinUCB는 \Theta^{*}를 estimate하기 위해 과거의 round로 부터 샘플을 사용하고 현재의 round를 위해 샘플을 선정한다. 그렇기 때문에 샘플들은 독립적이지 않다. Abbasi-Yadkori의 접근 방식에서는 마팅게일 방식으로 접근하여 독립적인 무작위 변수의 선형적인 조합을 통해 구축되어야 한다는 가정 없이 predicator를 위한 concentration result를 얻을 수 있다. 

<img src="imgs/Survey_of_Multi-armed_Bandit_LinUCB_2.jpg"/>

이로 인해 적절한 LinUCB를 위한 \alpha_t를 선정할 수 있다. \alpha는 t값에 의존적이며 independent 가정을 가지고 있는 original LinUCB 알고리즘과 약간 다름을 알 수 있다.