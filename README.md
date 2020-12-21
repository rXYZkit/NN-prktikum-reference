# NN-prktikum-reference
在这里记录看到的有用的参考信息

## 参考1
MTCNN+Face_recognition实时人脸识别（一）基础环境的搭建/算法结合 https://blog.csdn.net/weixin_38106878/article/details/101286761
MTCNN+Face_recognition实时人脸识别(二)训练自己的数据/多进程实时视频人脸识别 https://blog.csdn.net/weixin_38106878/article/details/102294999   


目录   
第一篇博客：基础环境的搭建以及mtcnn算法与人脸识别算法的结合
1. 人脸识别思路以及project代码思路的简要介绍；
2. 算法环境的搭建以及相关依赖包的安装；
3. MTCNN算法与face-recognition算法结合的测试code；   



第二篇博客：训练自己的人脸识别模型   
1. 数据集结构的介绍
2. 利用KNN训练生成自己的人脸特征底库
3. 多进程实时人脸识别
4. Project总结

人脸识别思路   
人脸识别是当今非常热门的技术之一，很多大公司已经把精度做到了99%以上，厉害～
人脸识别技术简单的实现流程是**人脸检测（对齐）+人脸特征提取+人脸比对+输出识别结果**；这个简单实现思路中，涉及到了两部分比较重要的算法，一部分是人脸检测算法，还有一部分就是人脸特征提取算法；   
1. 本博客就是按照这个简单的人脸识别思路进行人脸识别；
2. 人脸检测部分算法采用的是**mtcnn**(其实face-recognition算法里面也有相关的人脸检测算法，dlib和cnn的)、
3. 人脸特征提取部分算法采用的是**face-recognition模块**中的face_encoding模块、
4. 特征提取以及生成底库特征采用的是KNN算法、
5. 人脸比对采用的是通过计算待识别人脸128特征与底库人脸特征的欧式距离；   


> 这里mtcnn是一个需要长时间训练的网络在[MTCNN + Deep_Sort实现多目标人脸跟踪之MTCNN人脸检测部分(一)](https://blog.csdn.net/weixin_38106878/article/details/98958406)训练了两天   
Face Alignment：This section is based on the work of MTCNN.
> 关于face-recognition模块：这是一个python的module 下载后使用  [Face Recognition 人脸识别模块方法学习](https://blog.csdn.net/u014695788/article/details/89352503)  [人工智能之Python人脸识别技术--face_recognition模块](https://blog.csdn.net/qq_31673689/article/details/79370412?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-3&spm=1001.2101.3001.4242)   

> MTCNN使用的一个教程 https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/



---
## 参考二
从[深度学习之人脸识别模型--FaceNet](https://www.buildworld.cn/2020/04/17/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9E%8B-FaceNet/#5%E3%80%81GPU%E5%86%85%E5%AD%98%E6%BA%A2%E5%87%BA%E9%97%AE%E9%A2%98%EF%BC%8C%E5%B7%B2%E7%BB%8F%E8%A7%A3%E5%86%B3) 以及它参考的文章 [FaceNet pre-trained模型以及FaceNet源码使用方法和讲解](https://blog.csdn.net/MrCharles/article/details/80360461)中摘取的关键：
> 因为程序中神经网络使用的是谷歌的“inception resnet v1”网络模型，这个模型的输入时160*160的图像，而我们下载的LFW数据集是250*250限像素的图像，所以需要进行图片的预处理。   

> 2）、基于mtcnn与facenet的人脸识别（输入单张图片判断这人是谁）
代码：facenet/contributed/predict.py
主要功能：   
① 使用mtcnn进行人脸检测并对齐与裁剪   
② 对裁剪的人脸使用facenet进行embedding  
③ 执行predict.py进行人脸识别（需要训练好的svm模型）   
① 使用mtcnn进行人脸检测并对齐与裁剪  
---

## 参考三
谷歌 face recognition github pytorch 第一个搜索结果 [Face Recognition Using Pytorch](https://github.com/timesler/facenet-pytorch)   



## 参考四
谷歌 face recognition github pytorch 第四个搜索结果 https://github.com/ZhaoJ9014/face.evoLVe.PyTorch，  还发布在微信公众号上[face.evoLVe：高性能人脸识别开源库，内附高能模型](https://mp.weixin.qq.com/s/V8VoyMqVvjblH358ozcWEg)   

face.evoLVe介绍
face.evoLVe 为人脸相关分析和应用提供了全面的人脸识别库，包括：   
1. 人脸对齐（人脸检测，特征点定位，仿射变换等）；   
2. 数据处理（例如，数据增广，数据平衡，归一化等）；   
3. 各种骨干网（例如，ResNet，IR，IR-SE，ResNeXt，SE-ResNeXt，DenseNet，LightCNN，MobileNet，ShuffleNet，DPN等）；   
4. 各种损失函数（例如，Softmax，Focal，Center，SphereFace，CosFace，AmSoftmax，ArcFace，Triplet等等）；   
5. 提高性能的技巧包（例如，训练改进，模型调整，知识蒸馏等）。    


## 参考五
一篇很基础的博文：[在Amazon Sagemaker上使用Pytorch进行人脸识别（包括Jupyter笔记本代码）](https://medium.com/vaibhav-malpanis-blog/face-recognition-using-pytorch-on-amazon-sagemaker-c4f9f34c45f5)  
+ 您可以看到我们正在使用预训练的模型（alexnet）。Torchvision提供了许多预先训练的模型


## 参考六 
在GitHub上看到的 [runiRC/resnet-face-pytorch](https://github.com/AruniRC/resnet-face-pytorch)  内容是   
Contents   
ResNet-50 on UMD-Faces
  + Dataset preparation
  + Training
  + Evaluation demo   
ResNet-50 on VGGFace2   
  + Dataset preparation
  + Training
  + Evaluation LFW   
  
就是使用 UMD-Faces/VGGFace2 两个数据集训练 ResNet-50网络

## 参考七
srwpf / ResNet50-Pytorch-Face-Recognition(https://gitee.com/srwpf/ResNet50-Pytorch-Face-Recognition)
一个非常简洁的完整的小项目组织形式，可以学习。内容是训练了resnet和vgg来用作面部识别

## 参考八 
AlfredXiangWu/LightCNN (https://github.com/AlfredXiangWu/LightCNN)
light_cnn出自2016 cvpr吴翔A Light CNN for Deep Face Representation with Noisy Labels，
优势在于一个很小的模型和一个非常不错的识别率。主要原因在于，
（1）作者使用maxout作为激活函数，实现了对噪声的过滤和对有用信号的保留，从而产生更好的特征图MFM(Max-Feature-Map)。这个思想非常不错，本人将此思想用在center_loss中，实现了大概0.5%的性能提升，同时，这个maxout也就是所谓的slice+eltwise，这2个层的好处就是，一，不会产生训练的参数，二，基本很少耗时，给人的感觉就是不做白不做，性能还有提升。

---

明天看一下这个https://github.com/ageitgey/face_recognition
## 参考九
看了这个，里面是使用已有的库(face-recognotopm)进行了从图片里找到人脸，识别图片中的人是谁等功能，但是没有用到深度学习啊
-->
但是顺藤摸瓜找到一个流程 链接：https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
![image.png](https://i.loli.net/2020/12/21/inQLCmJMd5sOVNy.png)   

![image.png](https://i.loli.net/2020/12/21/3GOUNDVxQrt5RE2.png)   


下午看下这个【Face-Recognition-from-scratch】https://github.com/vishvanath45/Face-Recognition-from-scratch
## 参考10
同上，这份项目指出，这是一个分类任务，可分为如下几个部分  
1. 收集图像类，即Narendra_modi，Arvind_kejriwal，人民
2. 从不同类别的图像中提取人脸
3. 将数据分为培训，测试和交叉验证三类
4. 在训练集中的每个班级的图像上训练分类器。
5. 在Test_Set上测试
读了这个项目其实内容无参考价值，上面的这个步骤可以参考
