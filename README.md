# SceneRecognition
   Scene recognition for images by machine learning algorithms. 

1. code/: 实现项目基本要求。 code-Gist/: 增加了 Gist 特征；

2. 实现了三个场景识别算法
   (1)小图像+K-NN  (2)视觉词汇包+K-NN (3)视觉词汇包+Linear SVM

3. 针对不同的参数，进行了多次的实验。html/ 下有每次实验代码生成的详细的实验结果，也可在项目报告书 html/index.html里点击混淆矩阵中的图片查看实验结果；

4. 将给定的data/放在本目录下后，需要下载VLfeat工具包 http://www.vlfeat.org/

   并将文件夹命名为vlfeat放在 code/ 和 code-Gist/ 目录下。直接运行，生成的是 视觉词汇包+KNN 的结果。
