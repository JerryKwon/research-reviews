# [Tian Li et al._May_2020]Federated Learning Challenges, Methods, and Future Directions

Federated Learning: Challenges, Methods, and Future Directions. <a href="https://ieeexplore.ieee.org/abstract/document/9084352">go article</a>



## Abstract

FL: 지역적으로 따로 관리되고 있는 데이터들에 대한 통계학적 학습과 관련된 학습법

동시적이고 큰 Network 학습을 위해서 필요한 것들은

1. large-scale ML
2. 분산 최적화
3. 보안이 보장되는 데이터 분석

해당 article에서는 FL의 고유한 특징과 도전점에 대해서 알아본다.

: 최근 접근법에 대한 관망, 추후 연구에 대한 방향성



## Introduction

모바일, 웨어러블, 자율주행에서 distributed network가 생성되고 있음.

* Edge Computing

모바일 사용자 모델링 및 개인화에서 여전히 데이터들은 그 환경에 Local하게 저장되지만, ML을 학습시키는데 있어서 특정 구역에 모아서 처리하고 있다.

: 로컬 디바이스에서도 기술의 발전으로 인해 그 데이터를 수용할 수 있을 정도로 성장했으며, 이 raw데이터를 특정 영역으로 옮기는 데 있어서 보안상의 문제가 있다.

=> 이것이 FL의 등장을 촉발한 촉매제 역할을 했다. (로컬 디바이스에서 학습을 수행하는...)

* Smartphones

  : 학습 모델을 main 서버에 두고 이를 모바일 디바이스에서 사용할 때는, 모델만 넘겨서 해당 데이터에 대해서 학습을 수행하고 다시 학습된 모델만 Main 서버에 다시 업데이트하여 뿌리는 방식?

* Organization (such as hostpital)

* IoT (such as 웨어러블, 자율주행, 스마트홈)



#### 핵심적인 문제점?

1. Expensive Communication

   : FL network의 병목현상.

    많은 양의 local 기기들과 Communication을 하게 되면 대역폭, 파워 등의 문제로 인해 연산속도가 기존의 local에서 수행하는 것 보다 느려질 수 있다는 것.

   따라서, communication-efficient한 방식으로 나누어서 training process를 진행하는 것이 중요함.

   이를 위해, 아래의 두가지가 중요함

   * communication round의 수를 줄이는 것
   * 각 round별 transmitted message를 줄이는 것



2. Systems heterogeneity (시스템 이질성?)

   : FL에서 기기의 Storage, computational, communication capabilities 는 하드웨어/네트워크/파워 의 스펙에 따라서 달라질 수 있다. 이러한 시스템 수준의 특성에 따른 문제들로 인해 FL 에서 발생하는 문제들이 심각해지고 있음.

   이를 극복하기 위해

   * partiticion의 양을 예측
   * 견고한 hardware
   * 디바이스와 연결되는 네트워크에 있어서 견고함.



3. Statistical heterogeneity

4. Privacy concern



### 관련 연구

FL에서 해결해야할 문제점

* Privacy
* Large scale ML
* 분산 최적화



1. Communication Efficiency

   * local updating method: communication round를 줄이는 방법

   * compression schemes: communication message size를 줄이는 방법

   * decentralized training (central serverd와 communication cost를 줄이기)

     : 기존의 center server와 star 형식으로 토폴로지를 가지는 것이 아니라, 주변 기기들 간에 network를 형성함으로써 낮은 대역폭 및 높은 지연율의 환경에서도 더 빠른 속도를 보여준다.

2. System heterogenity

   * Asynchronous communication

   * Active Sampling

     : FL 네트워크에서는 작은 부분의 기기들만 학습 라운드에 참여하게 되어있다. 그러나 대다수의 FL모델들이 수동적이고 어느 기기들이 학습에 참여하는지 영항을 미치도록 목표하지 않으므로

     그에 대한 대안으로 각 학습라운드에 참여할 기기들을 동적으로 선택하는 방식을 제안한다.

   * Fault tolerance

     Center에서 ML모델을 학습하다가 발생하는 tolerance 문제보다 decentral한 환경에서 발생하는 문제가 더 치명적이다. 따라서 이에 대한 문제점 해결 책 중에 실패한 것에 대해서는 단지 "무시"하는 방법이 제안됨.

3. Statistical heterogenity

   * Modeling heterogeneous data

     : 학습함에 있어서 특정 기기에 데이터의 양이나 스펙등으로 인해서 더 많은 가중치를 가지고 학습되는 경우가 있다. 이를 공평한 가중치로 하여 학습을 진행하는 것 또한 중요하다.

   * Convergence guarantees for non-i.i.d data

4. Privacy

   * Privacy in ML

     1. differential privacy to communicate noisy data sketches
     2. homomorphic encryption to operate on encrypted data
     3. secure function evaluation or multiparty computation

     differential privacy VS model accuarcy

   * Privacy in FL



## Future directions

1. Extreme Communication Schemes
2. Communication reduction and the Pareto Frontier
3. Novel models of asynchrony
4. Heterogenity diagnositics
5. Granular privacy constraints
6. Beyond Supervised Learning
7. Productionizing ferderated learning
8. Benchmarking



## Conclusion

FL: 분산된 네트워크의 끝점에서 모델을 학습시키는 패러다임

FL VS Traditional distributed data center computing & Classical privacy-preserving learning

FL을 알아봄으로써 추후에 해결해나가야할 연구적인 노력 또한 살펴보았다.(어디에서??)