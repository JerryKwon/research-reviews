# Factorization Machines

https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

## References

* concept of FM

  https://www.jefkine.com/recsys/2017/03/27/factorization-machines/

* etc references

  https://yamalab.tistory.com/107

  https://yamalab.tistory.com/128?category=747907

  https://greeksharifa.github.io/machine_learning/2019/12/21/FM/

* Packages

  https://github.com/rixwew/pytorch-fm

  https://github.com/jmhessel/fmpytorch

  https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch



 FM(Factorization Machine)은 SVM과 같은 예측모델이지만 높은 희소행렬 환경에서도 신뢰할 만한 추정이 이루어지는 모델이다. FM은 선형적인 시간복잡도로 수행이 되며, 이로 인해 직접적인 최적화 및 모델 파라미터 저장 시 SVM과 달리 그에 수반되는 데이터를 저장할 필요가 없다. 



FM의 장점

1. SVM이 실패하는 엄청 희소한 데이터에 대해서도 parameter 추정이 가능
2. 선형적인 복잡성으로 support vector들에 의존이 필요없이 최적화 수행이 가능하고, 엄청난 양이 데이터셋에도 훈련이 가능하다.
3. 어떤 실제 값을 가진 feature vector에도 예측을 수행할 수 있다.

