# Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization

hyperparameter tuning by pure-exploration non-stochastic infinite-armed bandit problem(NIAB)



hyperparameter tuning 법에는

* 지정된 값에서 결정하는 Grid Search

* 지정된 범위에서 무작위로 수행하는 Randomized Search

* Bayesian Optimization - configuration selection [SMAC | TPE  | Spearmint]

  적응적인 방식으로 최적의 파라미터를 결정하는 Random Search보다 좋고 빠른 결과를 내는 대 목표.

  그러나, 모델에의 즉각적인 fitting, 고차원에서의 최적화, smoothness 성질을 가진 non-convex function, noisy evaluation에 대한 문제를 지닌다. 

  실증적으로 Random Search보다 좋은 성능을 띄지만, 고차원 환경에서는 Random Search와 유사한 성능을 낸다. 이 때문에 최근의 방식에는 저차원 환경을 가정하기도 하고, target function에 additive decomposition을 적용하기도 한다. 

  BO의 acquisition function으로 gaussian process를 반영한 GP-UCB가 사용되고 있다. 이는 exploration과 exploitation을 조절함으로써 파라미터 tunning의 필요성을 줄인다. 그런데 GP의 학습률은 매우 민감하고, poor prior를 가지는 경우 학습률은 관측값 n에 대해 logarithmic 하다. 그리고 공분산에 대한 구조적인 가정이 없는경우 posterior 에 fitting하는데 $O(n^3)$의 시간복잡도를 가진다.  

* non-stochastic infinite-armed bandit problem

  

configuration selection vs configuration evaluation

* configuration selection

  i.e random search

* configuration evaluation

  더 promising한 hyperparameter에 많은 resource를 부여하고, 나쁜 파라미터는 빨리 제거하는 방식.

  이렇게 Resource를 차별적으로 부여하는 방식의 목표는 동일하게 부여하는 방식 보다 더 많은 hyperparameter tuning 조합들을 시험해 보는 것이다. 

  **해당 방식은 Bayesian Optimization에도 적용되고 있지만, 해당 paper는 Random Search 방식의 추정속도를 높이는데 있다.** 



## 목표

hyperparameter optimization을 임의로 추출된 hyperparameter 값들을 가지고 한정된 resource를 이들에 할당시켜주는 'pure-exploration adaptive resource allocation problem' 문제로 정의한다. 

자원 할당에 있어 Bayesian Optimization Method와 같은 'black-box'절차를 따르는 방식보다 많은 임의의 파라미터 조합들을 평가할 수 있는 principled early-stopping'방식을 활용한다. 그리고 Hyperband는 prior configuration evaluation 방법과 달리 사전 가정을 최소로하는 방법론이다.  

Hyperband의 장점은 unknown convergence rate와  behavior of validation losses를 hyperparameter의 function에 적용할 수 있다는 것이고, BO보다 몇배 더 빠른 성능을 보여준다. 

### Adaptive configuration evaluation

이는 training time이 상대적으로 덜들어가고, 검증셋의 서브셋이 늘어난 상태에서 큰 검증셋에 대한 evaluation을 빠르게(성능이 나쁜 configuration에 대해서는 early-stopping을 적용하는 방법론) 하기 위해 생겨난 접근법이다. 

**<u>검증셋의 서브셋이 unbiased estimate를 가진다고 가정하기 때문에, 이를 multi-armed bandit을 활용한 stochastic best-arm identification problem이라고 정의할 수 있다.</u>** 

그런데, evaluation time이 상대적으로 inexpensive하고 목표를  전체 검증셋에 대해 부분적으로 학습한 모델을 평가하는 early-stop long-running training procedure이라고 설정하게 되면, 과거의 접근법들은 많은 제약들을 요구 했다. 

최근에는 Adaptive configuration evaluation를 접목한 여러 hybrid method들이 등장한다. 

이 중에 명시적인 convergence behavior를 요구하지 않는 **<u>'halving style bandit algorithm'</u>** 이 있는데, 이는 이로적으로도 증명이되었고 실증적인 결과도 좋았다. 그러나 이 방법에서의 문제점은 <u>**'n versus B/n problem'**</u>이 발생하는 것이다. 

이를 해결하기 위해서 paper에서는 Hyperband 알고리즘을 보이고, 이는 이론적으로도 탄탄한 'principled early-stopping algorithm'이다. 그리고 이는 parameter를 sampling하는데 있어 random sampling이 아닌 임의의 전략을 도입할 수도 있다. 단지 가정하는 것 하나는 샘플링을 통해 정의한 parameter set의 loss는 stationary distribution을 따른다는 것이다 .



## Bandit? 

Pure exploration bandit의 목표는 simple regret을 최소화 하는 것이다. stochastic setting을 반영한 bandit은 기존부터 연구가 이어져왔지만, non-stochastic setting을 위한 연구도 이뤄지고 있다.

* stochastic pure-exploration infinite-armed bandit

  각각의 arm i는 i.i.d한 특성을 가지고, 0에서 1사이의 값을 가지고 expectation vi를 가진다. vi는 누적분포함수 F 를 가지고 있는 분포를 통해 얻는 손실 값이다. 이 vi값은 unknown value이지만, arm을 i번 많이 당김으로써 이를 추정할 수 있다. 

  'anytime algorithm'에서는 upper bound의 에러가 F의 beta-parameterization하다는 것 아래에 추정할 수 있다.  그런데 이는 모델에 앞서 beta를 추정해야하므로 실효성에 문제가 있고 stochastic setting은 수렴하는 정도를 이미 알아야하기 떄문에 바람직하지 못하다.

* non-stochastic

  Hyperband는 위와 달리 non-stochastic한 설정에서 이루어지고, 임의의 파라미터에 대한 가정 없이 unknown F함수를 adaptive하게 수행한다.

## Hyperband

### Successive Halving

정의된 하이퍼 파라미터의 집합에 대해 동일한 수준의 budget을 할당하여 모든 설정값들에 대해서 평가하고, 나쁜 것에대해서는 버린다. 이를 하나의 configuration이 남을때 까지 수행한다. 

그런데 이방법은 input으로 설정값 조합의 개수를 먼저 정의해야한다. 주어진 Budget B에서  B/n의 자원들이 모든configuration들에 할당된다. 그러나 이 고정된 B값으로 인해 문제가 방생하는데,

* n의 값이 큰경우: 많은 configuration들을 생성하지만, 훈련하는데 적은 시간을 배정
* n의 값이 작은 경우: 적은 configuration을 생성하지만, 훈련하는데 긴시간을 배정

**<u>TRADE-OFF</u>**

그러나 일반적으로 우리가 추정해야 하는 loss function과 최적점은 알려져 있지 않다. 

따라서, 

* configuration하기 이전에 많은 자원을 필요로하는 것은 quality의 측면에서 달라짐을 확인할 수 있다.

  그렇다는 것은, 적은 수의 configuration으로 적절히 추론할 수 있다는 것이고

* configuration의 quality가 일반적으로 작은 수의 자원을 수행한 뒤에 나타난다면, n은 bottleneck 상태이고 n이 필요 이상으로 많은 것을 알 수 있다.  

다시 말해, 과거의 경험을 활용하는 것이 trade-off에 잘 대응하는 것으로 알 수 있고, 임의의 반복이 exploit한 결정이면 trade off에 대해 대부분의 자원을 할당한다. 

n의 값을 크게 잡으면 r의 개수가 작아져 early-stopping하는 행위를 더 자주하게 된다. 

<img src="imgs/hyperband_1.png" />

* Inputs

  R: single configuration에 할당할 최대 자원량

  eta: SUCCESSIVEHALVING에서 라운드별 제거할 configuration의 비율

Hyperband는 the most aggressive bracket(n)부터 수행하고, R resource에 적어도 1이 남을때까지 반복한다. 각 연속적인 bracket은 eta에 의해 점차 줄여나가고 s=0일때(classical random search) 모든 configuration이 R 리소스를 할당받는다. 

Hyperband는 config별로 할당한 평균 budget을 활용한 geometric search를 수행하며, fixed budget을 위해 n의 개수를 고를 필요가 없다. 

### Theory

<img src="imgs/hyperband_table_1.png" />

* $\mathcal{X}$: the space of valid hyperparameter configurations

* $k=1,2,... l_k:\mathcal{X}\rightarrow{[0,1]}$ X에 의해 정의된 paramter들의 loss function들

* $x\in\mathcal{X},l_k(x)$ x로 train된 모델의 k번째 resource(iteration)를 할당한 validation error

* **$R\in\cup\mathcal{N}\{\infty\},** l_*=lim_{k\rightarrow{R}}l_k; v_*=inf_{x\in{\mathcal{X}}}l_*(x)$

**<u>$l_k(\centerdot)$ for all $k\in\mathcal{N},l_*(\centerdot)$ and $v_*$ are all unknown</u>**

해당 문제에서는 어떻게 $l_k(x)$함수가 고정된 k값에서 파라미터 x에 의해 달라지는지와 $l_k(x)\rightarrow{l_*(x)}$ 고정된 파라미터 x를 가지고 얼마나 빨리 최적의 loss로 접근하는지가 불확실하다.



**<u>hyperparameter configuration는 임의로 정의한 확률분포  p(x)를 통해 추출한다</u>**. 이를 추출하는 분포는 uniform distribution이며, 다른 임의의 sampling method를 사용해도 된다. 만약 $X\in\mathcal{X}$가 p(x)확률분포를 따르는 가정하에 샘플링되었다면, $l_*(X)$ 또한 $l_*(\centerdot)$를 모르기 때문에 분포를 알 수 없는 확률변수 일 것이다. 다시말해 $l_k(x)$가 어떻게 x또는 k에 의해 달라지는지 모르기 때문에 $l_j(y);(j\in{N},y\in{X})$를 통해 $l_k(x)$를 추정할 수 없게 된다. 결과적으로 parameter 추정은 더욱 더 간단한 문제가 되고 파라미터를 어떻게 추출하는지 무시할 수 있게 된다.

단지, $x\in\mathcal{X}$가 loss function sequence $l_k(x);k=1,2,...$와 상호작용하는 문제로 정의할 수 있다. 이로 인해, 특정 parameterset $x\in\mathcal{X}$는 loss sequence를 구분하는 index값의 의미만 가지게 된다.



Hyperband의 목표는,

얼마나 빨리 $l_k(\centerdot)\rightarrow{l_*(\centerdot)}$로 수렴하는지 또는 어떻게$l_*(X)$가 분포로 나타나는지에 대한 정보 없이,

가능한 적은 자원을 가지고, 임의의 parameter configuration을 시도함으로써 $l_*(x)-v_*$를 최소화하는  $x\in\mathcal{X}$ parameter configuration을 얻는 것이다.



* Assumption1.
* Assumption2.