<!-- Heading -->

# TaiLab_Net

<!-- 수정 사항입니다.-->

Tailab_Net은 인공지능 학습용 심장질환 심초음파 및 심전도 데이터셋을 활용한 AI 모델 Datathon인 Heart Disease AI Datathon 2021(H.D.A.I 2021)의 참여를 위해 만들어진 nnUNet 기반의 응용 model로써, 대회 주제1의 A2C, A4C View에서의 좌심실 분할 모델 개발을 위해 만들어졌습니다.

<!-- install 없애도 되고, 다른 이름 method 대제목으로 바꿔서 github 참고시키고 우리가 nnUnet 사용해서 train을 했다. 길어서 풀고 -->


## Featured Results
|A2C Evaluation|Model 1|Model 2|Model 3|Ensemble|
|:----------:|:------:|:------:|:------:|:------:|
|Dice Similiarity Coefficient|0.9189|0.9241|0.9262|0.0000|
|Jaccard Index|0.0000|0.0000|0.0000|0.0000|

|A4C Evaluation|Model 1|Model 2|Model 3|Ensemble|
|:----------:|:------:|:------:|:------:|:------:|
|Dice Similiarity Coefficient|0.9489|0.9517|0.9491|0.0000|
|Jaccard Index|0.0000|0.0000|0.0000|0.0000|

Note: Ensemble 모델의 경우, minimum, median, maximum, mean의 DSC와 JI 값을 기준으로 제일 높았던 것으로 결정했으며, A2C의 경우 Minimum, A4C의 경우 OOOO으로 설정하였습니다. 

## Requirements
테스트는 아래와 같은 환경에서 이루어졌습니다.

- Linux(Ubuntu 18.04)
- Python 3.7
- Pytorch 1.7+
- CUDA 11.3

TaiLab_Net 모델 시연에 앞서, 최소한의 요구사항은 다음과 같습니다.

- imgaug==0.4.0
- matplotlib==3.4.2
- numpy==1.21.1
- opencv-python==4.5.3.56
- SimpleITK==2.1.0
- tqdm==4.61.2

 편의를 위해 저희가 마련한 requirements.txt 파일을 설치/참고해서 모델을 run 할 수 있게 준비해 두었습니다. requirements.txt를 설치하기 앞서 TaiLab_Net을 아래와 같이 설치합니다.

```
git clone https://github.com/Impku/HDAI2021_Tail.git
cd HDAI2021_Tail
```

터미널에 아래와 같이 입력하여 requirements.txt를 설치합니다.

```
pip install -r requirements.txt
```

## Structure:
- ```data/```: 지정된 기본 Test Dataset 폴더
- ```dataset/```: 지정된 기본 dataloader 폴더
- ```exp/```: 지정된 기본 결과 출력 폴더
- ```net/```: UNet 모델과 관련 trainer 저장 폴더
- ```utils/```: 이미지 처리 및 arg parser 등 필요 라이브러리 및 함수 저장 폴더
- ```weights/```: pre-trained 모델 저장 폴더
- ```inference_A2C.py``` : A2C를 위한 inference 실행 코드
- ```inference_A4C.py``` : A4C를 위한 inference 실행 코드

## Train
TaiLab_Net은 ~~~~~~~ train 되었습니다. 


## Methods

<!-- 수정 사항입니다. 모델 2개를 사용했기 때문에 각각의 방법을 모두 설명해야함-->

TaiLab_Net은 nnU-Net 기반으로 pre-trained된 모델입니다. nnU-Net 관련 정보를 더 알고 싶으시다면, [References](#references) 섹션에 기재된 논문을 참조해주세요. 
TaiLab_Net은 inference만 시연 가능하게 만들어졌고, Train과 Validation dataset은 대회 참가에 제공된 Dataset을 이용했습니다. 

자세한 Inference 방법은 다음과 같이 실행해주세요.

0. Inference 전에 nnU-Net을 설치한다.

   - 새로운 가상 환경을 만들어주세요.
   - PyTorch를 설치해주세요.
   - 아래와 같이 nnU-Net을 설치해주세요. 
   
   ```
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   cd nnUNet
   pip install git+https://github.com/MIC-DKFZ/batchgenerators.git
   pip install -e .
   ```

1. Inference를 위해서는 몇가지 argument들을 설정 및 지정해주셔야 합니다. 
총 4개의 ```--data_root```, ```--exp```, ```--json```, ```--plot_png``` argument들을 지정해줄 수 있습니다.
   - ```--data_root```: inference를 위한 data 디렉토리 경로    
      - default directory: ```./data ```

      폴더 구조 예시)
      ```
      data
      ├───── A2C
      │      ├──  0801.png
      │      ├──  0801.npy
      │      │      ...    
      │      └──    
      │
      └───── A4C
              ├──  0801.png (번호 확인)
              ├──  0801.npy
              │      ...
              └── 
      ```
   
   - ```--exp```: .npy파일이 저장되는 output 디렉토리 경로 
      - default directory: ```./exp ```

   - ```--json```: 결과 document의 저장 여부 (boolean)
      - default value: ``` false ```
      - default directory: ```./exp/result_A2C.json```  **OR**  ```./exp/result_A4C.json```

   - ```--plot_png```: 생성된 mask의 저장 여부 (boolean)
      - default value: ``` false ```
      - default directory: **```위에 --exp에서 설정한 경로와 같은 directory에 저장  ```**


2. A2C 데이터 Inference를 위해서 터미널에 다음과 같이 입력하세요:
   ```
   python3 inference_A2C.py --data_root "path_to_data_directory" --exp "path_to_output_data_directory"
   ```
   - A4C 데이터 Inference 역시 동일한 방법으로 입력/실행해주세요. 


3. 시연하는 환경마다 차이가 있겠지만, 100개의 Validation Data 기준, inference는 약 5분정도 소요됩니다. 모든 inference가 끝나면 사용자가 위에서 지정한 output directory안에 npy파일들이 생성됩니다. 


## References

<!-- Citation 적을게 뭐가 더 있을지 알려주세요. 수정사항 입니다. format도 제안 주시면 바꿔놓겠습니다.  -->

더 많은 nnU-Net 구조와 정보를 원하시면, [여기](https://github.com/MIC-DKFZ/nnUNet)를 참조해주세요.

아래는 nnU-Net의 논문 citation 정보입니다:

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.



## Contacts

<!-- 메일 주소 넘기기 완료. 근데 공용 이메일 대신 일단 제 이메일 넣어놨어요.
    ㄴ 방금전 태윤이가 준 주소로 다시 수정해놓았습니다.  -->

에러나 버그 또는 관련된 질문사항이 있으시다면 저희 [이메일](mailto:ygj03084@gmail.com)로 연락 부탁드리겠습니다.

<!--- 연대 로고를 넣으려고 했는데,, 뒤에 흰색 배경이 나와서 일단은 넣지 않았습니다. 의견 주세요  --->

---

![tailab_logo2](https://user-images.githubusercontent.com/39204766/144746204-2d39b036-3ea0-476e-945d-25e4f695ece1.png)

TaiLab-Net은 [연세대학교(Yonsei University)](https://www.yonsei.ac.kr/en_sc/index.jsp)의 [의료인공지능연구실 (tAILab)](https://sites.google.com/view/tailab/home?authuser=0) 에 의해 응용, 유지 되고 있습니다.
