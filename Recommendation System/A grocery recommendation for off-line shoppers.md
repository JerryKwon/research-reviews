# A grocery recommendation for off-line shoppers

## Abstract
고객의 다음 장바구니의 물건을 예측하기 위해 'Marcov chain model'을 활용한 기법을 사용. 이는 구매한 물건과 구매의 흐름을 동시에 고려한 방법.

이를 CF(Collaborate Filitering), bestseller-based system 과 비교하여 그 성능에 대해 알아본다.

## Introduction
Grocery store를 위한 recommendation은 일반 recommendation system이 지향하는 점과 다른 점이 있다. 이는 반복적인 소비를 recommendation에 추가하는지의 차이이다. 

예를 들어 음악 추천 시스템의 경우에는 특정 사용자가 들은 음악을 기반으로 '다른 유사한 음악'을 추천하는 것이다. 그러나 Grocery store의 예시에서는 한국인의 경우 '마늘'이 대부분의 음식에 사용되기 때문에 이를 반복적으로 소비하는 경향이 많다. 

그래서 위와 같은 두 가지 모두에 대해서 예측하기 위해 '고객이 구매한 제품'와 '고객이 다음으로 구매할 제품'을 동시에 고려하는 Marcov chain 기반의 방식을 활용한다.

## Related Work
### 2.2. Markov chains
이산적인 타임라인에 있어서 특정 랜덤한 변수($X_1, X_2, ..., X_n$)로 이동하는 순서를 나타내는 확률적인 절차이다. 이는 이산적인 시간 순서에 있어 확률적인 절차를 나타내므로 미래의 특정 시간에 있어서 상태를 예측하는데 매우 유용하다.

<div align="center" style="margin-top:15px">$P(X_{t+1}=j | X_1=i_1, X_2=i_2,...,X_t=i) = P(X_{t+1}=j|X_t=i)=P_{ij}$</div> 

위의 $P_{ij}를 전이확률 이라고 일컫으며, 아래의 수식을 만족한다면 순서 m의 Markov chain이라고 부른다.

<div align="center" style="margin-top:15px">$P(X_{t+1}=J | X_t=i, X_{t-1}=i_{t-1},...,X_{t-m}=x_{t-m}, where t>m$</div> 


## Methodology

![overall_config](/img/grocery_recommendation_overall_config.jpg)

Markov chain을 활용한 Grocery Recommendation의 도식은 위와 같다. 앞서 이에 특징적인 Recommendation 수행을 위해 'Weighted gerneral preference profile'과 'Weighted dynamic preference profile'을 만들고, 각 고객들간의 Cosine Simliarity를 계산한 후 이를 결합한다. 마지막으로, 이웃간의 장바구니의 전이확률 을 계산하고 이를 고객의 이전 장바구니 정보를 활용하여 상품을 추천한다.

$U={u_1,...,u_m}; P={p_1,...p_n}$

$B^{ui}={B_1^{ui},...,B_t^{ui}}$ 

$B_l^{ui}={p_1,...,p_n}, n\ge2$

장바구니를 구성하는데 있어서 무조건 2개 이상의 상품을 가진 고객들을 대상으로 하였다. 

![example](/img/grocery_recommendation_example.jpg)

### 3.2.1. Weighted general preference profile
$g_{i,jk} = p_{i,jk}, if p_{i,jk}> 0 ; 0, otherwise$

![wgps](/img/grocery_recommendation_weighted_general_preference_profile.jpg)

### 3.2.2. Weighted dynamic preference profile
$d_{i,jk} = f_{i,jk}, if f_{i,jk}> 0 ; 0, otherwise$

![wdps](/img/grocery_recommendation_weighted_dynamic_preference_profile.jpg)

### 3.3. Neighbor formation

$sim(a,b) = {2*wgps(a,b)*wdps(a,b)}/{wgps(a,b)+wdps(a,b)}$

고객들의 profile을 생성하고 각각 그들간의 유사도를 계산한다. 

$wgps(a,b) = {a*b}/{\Ia\I \Ib\I}$

참고. <a href="https://neo4j.com/docs/graph-algorithms/current/labs-algorithms/cosine/">Cosine Similarity<a>

이를 계산한 후, 특정 고객 a와 가장 유사도가 높은 N명의 고객을 도출한다.

### 3.4. Markov Chain을 활용한 top-N recommendation list 생성

Markcov chain에서 미래의 t+1의 Basket을 예측하기 위해 과거의 m개 시점의 데이터를 참고하게 된다. 그러나 해당 논문에서는 m=1이다. 왜냐하면 한명의 고객은 적어도 2개 이상의 상품을 구매하기 때문이다.

$Pr(B_{t+1}|B_t,...,B_{t-m})$

$B_t={B_t^{u1},...,B_t^{um}}; B_t^{ui}={p_1,...,p_n}$

PLS(a,j)={\underset{i\inN}{\sum}((pr(p_j\inB_{t+1}^{u_i}|B_t^a\subseteqB_t^{u_i},...,B_{t-m}^a\subseteqB_{t-m}^{u_i}))* sim(a,i))}/{\underset{i\inN}{\sum}sum(a,i)}

추천 대상 고객 a에 대해 j물품이 t+1 시간대의 장바구니에 있을 확룰은 위와 같이 계산한다.

이를 가지고 PLS값의 상위 N개 제품에 대해서 recommendation을 수행한다.

## Experiment

![evaluation](/img/grocery_recommendation_evaluation.jpg)

### 모수 검정, 비모수 검정
link: <a href="https://qlclinic.com/62">click here</a>

![kruskal-wallis test](/img/grocery_recommendation_kruskal-wallis test.jpg)

About <a href="https://nate9389.tistory.com/1688">Kruskal-Wallis test</a>

![Mann-Whitney test](/img/grocery_recommendation_Mann-Whitney test.jpg)

About <a href="https://nate9389.tistory.com/1689?category=1044821">Mann-Whitney test</a>
