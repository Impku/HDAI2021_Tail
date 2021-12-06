<!-- Heading -->

# TaiLab_Net

<!-- 수정 사항입니다.-->

Tailab_Net은 인공지능 학습용 심장질환 심초음파 및 심전도 데이터셋을 활용한 AI 모델 Datathon인 Heart Disease AI Datathon 2021(H.D.A.I 2021)의 참여를 위해 만들어진 nnUNet 기반의 응용 model로써, 대회 주제1의 A2C, A4C View에서의 좌심실 분할 모델 개발을 위해 만들어졌습니다.

## Table of Contents

- [Requirements](#requirements)
- [Methods](#methods)
  - how to predict / inference method
- [References](#references)
- [Contacts](#contacts)

<!-- install 없애도 되고, 다른 이름 method 대제목으로 바꿔서 github 참고시키고 우리가 nnUnet 사용해서 train을 했다. 길어서 풀고 -->

## Requirements

TaiLab_Net 모델 시연에 앞서, 최소한의 요구사항은 다음과 같습니다. 편의를 위해 저희가 마련한 requirements.txt 파일을 설치/참고해서 모델을 run 할 수 있게 준비해 두었습니다.

This is the **MINIMUM** requirement you have to satisfy before you play with this _model_.
For your convenience, we have uploaded our requirements.txt file in out repository.

<!-- pytorch, 또는 tensorflow 등 제일 필요한 부분만 골라서 쓰면 좋을듯 싶으나, 거르는데 시간이 오래 걸린다면, 그냥
 terminal 창에

 pip freeze > requirements.txt

치시면 설치된 library들 version들이 적힌 requirements.txt가 나옵니다 그걸 이곳에 복사 / 붙여넣기 하면 될 것 같습니다.

-->

```
   pip install -r requirements.txt
```

## Methods

<!-- 수정 사항입니다.-->

TaiLab_Net은 nnUNet 기반으로 pre-trained된 모델입니다. nnUNet 관련 정보를 더 알고 싶으시다면, References 섹션에 기재된 논문을 참조해주세요. TaiLab_Net은 inference만 시연 가능하게 만들어졌고, Train과 Validation dataset은 대회 참가에 제공되었던 Dataset을 이용했습니다. 다음의 방법은 inference 방법
paragraph paragraph
paragraph paragraph

1. ~~ 을 설치합니다.
2. ~~ 터미널을 열어서 다음과 같이 type 합니다.

   i. asdfasdfjkla;sjdfl;asjdf

   ```
   pip install TaiLab-Net
   ```

   ii. asdfaksdpfajsdpfj

   ```
   git clone https://github.com/hjung35
   cd TaiLabNet
   pip install -e
   ```

3. 시연하는 환경마다 차이가 있겠지만, 약 1시간 이후 내외로 모든 inference가 끝나면 터미널이 열린 directory에 폴더가 새로 만들어집니다.
4. 비교 연산을 위해 DSC와 JI가 제공되며 이를 활용해 모델의 정확도를 측정할 수 있습니다.
   `pip install --upgrade git+https://github.com/hjung35`
   paragraph paragraph
   paragraph paragraph

## References

<!-- Citation 적을게 뭐가 더 있을지 알려주세요. 수정사항 입니다. format도 제안 주시면 바꿔놓겠습니다.  -->

For more information about nnU-Net, the base model of ours, please read the following paper:

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.

[nnUNet](https://github.com/MIC-DKFZ/nnUNet)

## Contacts

<!-- 메일 주소 넘기기 완료. 근데 공용 이메일 대신 일단 제 이메일 넣어놨어요.
    ㄴ 방금전 태윤이가 준 주소로 다시 수정해놓았습니다.  -->

에러나 버그 또는 관련된 질문사항이 있으시다면 저희 [이메일](mailto:ygj03084@gmail.com)로 연락 부탁드리겠습니다.

<!--- 연대 로고를 넣으려고 했는데,, 뒤에 흰색 배경이 나와서 일단은 넣지 않았습니다. 의견 주세요  --->

---

![tailab_logo2](https://user-images.githubusercontent.com/39204766/144746204-2d39b036-3ea0-476e-945d-25e4f695ece1.png)

TaiLab-Net은 [연세대학교(Yonsei University)](https://www.yonsei.ac.kr/en_sc/index.jsp)의 [의료인공지능연구실 (tAILab)](https://sites.google.com/view/tailab/home?authuser=0) 에 의해 응용, 유지 되고 있습니다.
