# MMCF: Multimodal Collaborative Filtering for Automatic Playlist Continuation

## paper link
<a href="https://dl.acm.org/doi/10.1145/3267471.3267482">https://dl.acm.org/doi/10.1145/3267471.3267482</a>

## github repo
<a href="https://github.com/hojinYang/spotify_recSys_challenge_2018">https://github.com/hojinYang/spotify_recSys_challenge_2018</a>

## 1. Abstract
Automatic playlist continuation(APC)는 음악 추천시스템의 기본적인 task이다. 사용자에게 특정 track으로 구성된 리스트를 playlist에 산재하는 특징을 반영하여 추천하는 것이다.

그러나 이러한 추천시스템은 여러 문제점을 가지고 있다(via Collaborative Filtering; CF).

1. 'popularity bias': 플레이리스트에 적게 등장하는 노래를 추천하지 못하는 문제

2. 'cold-start problem': 엄청 적은 수의 트랙만을 가지고 있는 플레이리스트를 활용하지 못하는 문제

3. 'context-aware continuation': track의 순서나 playlist title과 같은 context한 정보를 무시하는 문제

해당 논문에서는 다양한 데이터를 효과적으로 이용하기 위해 'multimodal collaborative filtering' 모델을 제안한다. 이는 크게 두개의 요소로 구성된다.

1. playlist와 categorical content를 사용하는 autoencoder

2. playlist title만들 사용한 character-level convoultional neural network

playlist와 context한 정보를 동시에 다룸으로써, 'popularity bias'와 'cold-start problem'에 대처하ㄹ수 있으며, playlist의 title값을 활용하여 더 잘 들어맞는 track을 추천할 수 있게 되었다.

## 2. Introduction

Abstract에서 언급된 자주 활용되는 CF 기반의 APC가 3가지 문제에 마주한다. 따라서 'multimodal collaborative filtering' 방식을 제안한다.

이 중 하나는 playlist와 categorical한 contents를 input으로 하는 'hide-and-seek apporach'에서 영감을 받은 autoencoder이다. 학습을 수행함에 있어 playlist나 contents 중에 하나를 무시한다. 더 나아가, playlist의 순서를 학습하기 위해 'two dropout' 전략들을 제안한다. 이를 통해서 playlist간의 복잡한 패턴 뿐만아니라 playlist에 순서에 있어서 hidden context를 학습할 수 있었다.

두 번째로는 'character-level convolutional neural network(charCNN)'이다. 이는 playlist와 playlist title간의 잠재 관계를 학습한다.

그리고 마지막으로, 두 모델의 앙상블을 위한 method를 보인다.

## 3. Preliminaries
### 3.3. Baseline Models
#### **Autoencoders**
AE(Autoencoder)는 비선형의 방식으로 데이터를 압축하여 표현하는 것과 임의의 결과값을 생성하는데 효과적인 모델이다. 최근 CF에 많이 사용되고 있으며, encoding부과 decoding부로 구성된다. 

입력부에는 은닉층으로 인해 압축되고 이를 다시 출력층으로 복원한다. 은닉층으로 인해 압축되면서 데이터를 임의의 잠재요인으로 표현한다.

* Encoder

    ![formular](https://render.githubusercontent.com/render/math?math={y=f(x)=\sigma(Wx+b),\ x\in[0,1]^n\ to\ y\in{R^d}})

* Decoder

    ![formular](https://render.githubusercontent.com/render/math?math={\hat{x}=g(y)=\sigma(W^{\prime}y+b^\prime),\ W^{\prime}\in{R^{n\times{d}}},b^{\prime}\in{R^n}})

따라서, Encoder와 Decoder를 지난 input x를 ![formular](https://render.githubusercontent.com/render/math?math={g(f(x))})로 표현할 수 있다.

##### **loss function**

![formular](https://render.githubusercontent.com/render/math?math={\underset{\theta}{argmin}\underset{p\in{P}}{\sum}\mathcal{L}(p,\hat{p}),\ \Theta=\{W,W^{\prime},b,b^{\prime}\}})


![formular](https://render.githubusercontent.com/render/math?math={\mathcal{L}(p,p^{\prime})=-\frac{1}{n}\overset{n}{\underset{i=1}{\sum}}p_i\ log\hat{p}_i+(1-p_i)log(1-\hat{p}_1)})


Autoencoder가 playlist에 대해 학습하는 것은 효과적이지만, 다른 context data를 학습하는데 한계가 있어 'Context-Aware Autoencoder'를 통해 한계를 극복하려고 한다.

#### **Character-level convolutional neural network**

CNN(convolutional neural network)는 이미지 및 영상 인식에서 많이 사용된다. CNN이 가지고 있는 텍스트의 지역성을 인지할 수 있는 능력 덕분에 텍스트 분류에도 사용되어 왔다.

CNN은 구조적으로 local feature 생성을 위한 convolution layer와 subsampling을 통해 정밀한 표현을 나타내는 pooling layer로 구성되어진다. 