<div align="center">   
    <img src = "https://github.com/SKT-FLY-AI-Challenger-3-Ambition-5/.github/assets/77189999/0f2bda5d-b2d0-4c31-a891-63a333191404" width = "750px">
    <h1>에이단ㅡ스</h1>
    <p> 유행하는 숏폼 챌린지, 나도 한 번 따라해보고 싶었다면?<br />
    내 댄스 실력을 다른 사람과 겨루며 성장하고 싶었다면?<br /><br />
    원하는 안무 영상의 유튜브 링크와 핸드폰 하나만으로 춤을 연습할 수 있는 <br />편리한 앱 서비스, 에이단ㅡ스를 만나보세요!</p>
    <h3>나만의 AI 안무 코치, 에이단ㅡ스</h3>
    <a href="https://skttechacademy.com/nonmember/flyAi/flyAiProjectReviewList"/>에이단ㅡ스 프로젝트 자세히 알아보기</a>
    <h3>🛠 Skill Keywords 🛠</h3>
    <div align="center">
         <img src="https://img.shields.io/badge/Flutter-02569B?style=flat-square&logo=Flutter&logoColor=white"/>&nbsp
         <img src="https://img.shields.io/badge/Android-3DDC84?style=flat-square&logo=Android&logoColor=white"/>&nbsp  
        <br>
        <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white">&nbsp
        <img src="https://img.shields.io/badge/MariaDB-003545?style=flat-square&logo=MariaDB&logoColor=white">&nbsp
        <img src="https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=MongoDB&logoColor=white">&nbsp
        <br>
        <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white"/>&nbsp
        <img src="https://img.shields.io/badge/K8S-326CE5?style=flat-square&logo=Kubernetes&logoColor=white"/>&nbsp
        <img src="https://img.shields.io/badge/Vultr-007BFC?style=flat-square&logo=Vultr&logoColor=white"/>&nbsp
        <br>
        <img src="https://img.shields.io/badge/tensorflow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white">&nbsp
        <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">&nbsp
        <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white">&nbsp
    </div>
</div>

<h2>에이단ㅡ스 아키텍쳐</h2>

<div align="center">
    <img width="1920" alt="에이단스 아키텍쳐" src="https://github.com/SKT-FLY-AI-Challenger-3-Ambition-5/.github/assets/77189999/3473c3d0-4c80-4bf9-bec7-4fbac0e14223">
</div>


<h3>1. 원하는 안무 영상을 무엇이든 선택해보세요!</h3>

- 원하는 영상을 포즈 추정 딥러닝 모델을 통해 어떤 움직임을 가지는지 추정합니다.

<div align="center">
    <img src="https://github.com/SKT-FLY-AI-Challenger-3-Ambition-5/.github/assets/77189999/943dcbdf-eca3-4b79-a2f8-fc9b4c68d593" width = "600px" />
</div>

<br/>

<h3>2. 댄스 실력을 발휘해보세요!</h3>

- 원하는 영상에 대한 움직임을 스켈레톤 형태의 가이드로 제공받고, 이를 통해 연습 영상을 촬영할 수 있습니다.

<div align="center">
    <img src="https://github.com/SKT-FLY-AI-Challenger-3-Ambition-5/.github/assets/77189999/13ae8fa2-e88f-4341-ab4b-0f3e2014326e" width = "600px" />
</div>
<br/>

<h3>3. 당신의 댄스 실력을 확인해보세요!</h3>

- 촬영한 연습 영상에 대해 원본 영상과의 유사도 측정을 통해 정확도 점수를 제공합니다. 
- 시간별 당신의 정확도 점수를 제공하므로, 부족한 부분을 쉽게 확인할 수 있습니다. 
- 특히 보완이 필요한 부분이 있다면 해당 부분에 대한 개선점을 제공합니다. 

<div align="center">
    <img src="https://github.com/SKT-FLY-AI-Challenger-3-Ambition-5/.github/assets/77189999/146d9426-dcb3-4b97-9f4d-7533888f8767" width = "600px" />
</div>

<br/>

<h3>4. 글로벌 사용자들과 순위 경쟁을 통해 성장해보세요!</h3>

- 서비스 내 사용자들과 경쟁을 위한 순위 리더보드가 제공되어, 더욱 더 즐겁게 안무 연습을 진행할 수 있습니다. 

<div align="center">
    <img src="https://github.com/SKT-FLY-AI-Challenger-3-Ambition-5/.github/assets/77189999/4690d553-360e-4ebe-a489-8a21b4c5d3b6" width = "600px" />
</div>

<br/>

<h2>유사도 측정 방법</h2>

- 키포인트 벡터에 대해 거리 기반의 메트릭인 PCK를 사용하여 유사도 측정을 진행하였습니다.  
- PCK란, Percentage of Correct Keypoints의 준말로, Human Pose Estimation에서 모델의 성능지표로 많이 사용되는 효과적인 메트릭이며, 모델의 예측값과 GT에 대한 정확도를 측정하여 포즈 추정 모델의 성능을 측정합니다.
<br/>
- 저희 프로젝트에서는 사용자의 영상에서 예측된 키포인트와 원본 영상에서 추출된 키포인트가 지정된 임계값 이내에 있는지를 측정하여 프레임별 정확도 점수를 구현하였습니다. 

<br/>


<h2>🔗 링크</h2>   

- 시연 영상 : https://youtu.be/PP-_aJwnB5w
- Backend 레포 : https://github.com/BanApp/A.Dance-Backend
- Frontend 레포 : https://github.com/BanApp/A.Dance_Front

<br/>

<h2>💁 팀원 소개</h2>

<table align=center>
    <thead>
        <tr >
            <th style="text-align:center;" >정민준</th>
            <th style="text-align:center;" >신민준</th>
            <th style="text-align:center;" >김다희</th>
            <th style="text-align:center;" >방지수</th>
            <th style="text-align:center;" >최재혁</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://github.com/BanApp"><img width="200" src="https://avatars.githubusercontent.com/u/93313445?v=4" /> </td>
            <td><a href="https://github.com/pulugia"><img width="200" src="https://avatars.githubusercontent.com/u/139101911?v=4"/></a></td>
            <td><a href="https://github.com/dahuikim"><img width="200" src="https://avatars.githubusercontent.com/u/123606183?v=4" /></a></td>
            <td><a href="https://github.com/usijgnap"><img width="200" src="https://avatars.githubusercontent.com/u/139188253?v=4" /></a></td>
            <td><a href="https://github.com/jjaegii"><img width="200" src="https://avatars.githubusercontent.com/u/77189999?v=4" /></a></td>
        </tr>
    </tbody>
</table>
