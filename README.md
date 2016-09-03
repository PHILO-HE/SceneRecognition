# SceneRecognition
   Scene recognition by machine learning. 

1. code/ 文件夹下为实现项目基本要求的代码。 code-Gist/ 文件夹下是增加 Gist 特征的代码；

2. 实现了三个场景识别算法
   (1)小图像+K-NN  (2)视觉词汇包+K-NN (3)视觉词汇包+Linear SVM

3. 针对不同的参数，进行了多次的实验。html/ 下有每次实验代码生成的详细Web数据，也可在项目报告书 html/index.html 里点击

   混淆矩阵图片(超链接)查看这些Web数据；

4. 将给定的data文件夹放在本目录下后，需要下载VLfeat工具包 http://www.vlfeat.org/

   并将文件夹命名为vlfeat放在 code/ 和 code-Gist/ 目录下。代码直接运行，生成的是 视觉词汇包+KNN 的结果。
