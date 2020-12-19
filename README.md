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
