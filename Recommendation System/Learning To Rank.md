# Learning To Rank

https://www.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/



implicit feedback matrix factorization을 전통적인 ALS나 SGD를 활용하지 않고 학습하는 다른 방식이 'Learning To Rank'이다. 



## BPR

이에 대한 가장 기초적인 접근은 <a href="https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf">BPR: Bayesian Personalized Ranking from Implicit Feedback</a>에서 찾아볼 수 있다. 해당 접근 법은 positive, negative 아이템들에 대해 sampling을 수행하고, pairwise하게 비교를 수행하는 것이다. 예제 데이터는 특정 유저가 웹사이트에서 다양한 item들에 대해 클릭한 횟수를 기반으로 한다.



1. 임의의 사용자 u가 임의의 아이템 i를 클릭하는 경우 - positive item
2. i item보다 적게 사용자들에 의해 클릭된 j 아이템은 - negative item
3. user u가 positive item i와 상호작용을 통해 얻은 점수를 $P_{ui}$ = $X_u * y_i$

4. user u와 negative item j와의 score를 $p_{uj}$
5. positive 와 negative score와의 차이를 $x_{uij} = p_{ui}-p{uj}$
6. 5의 차이를 sigmoid를 통해 이를 SGD을 통해 모델의 파라미터를 설정하고 업데이트 한다.



## WARP

BPR과 약간의 유사한 개념이며, Weighted Approximate-Rank Pairwise loss(WARP loss)는 <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37180.pdf">WSABIE: Scaling Up To Large Vocabulary Image Annotation</a>에서 처음으로 소개되었다. WARP도 BPR과 같이 특정 사용자에 대해 positive 및 negative item을 골라내고, 예측하고, 차이를 계산한다. 그러나 WARP는 prediction이 잘못되었을 경우에만 SGD 업데이트를 실시한다. (e.g. negative item의 score를 positive item의 score보다 높게 예측하는 경우). WARP에서는 BPR이 AUC score를 loss function으로 지정한 것과 달리 precision을 optimizing하는 방식으로 최적화한다. 



WARP는 두개의 parameter가 있고,

1. SGD 업데이트를 반드시 수행해야하는 wrong prediction의 정도
2. 다은 사용자로 넘어가기 이전에 negative sample으로 얼마나 많이 틀린 예측을 수행하는지