# NN-praktikum-reference
在这里记录看到的有用的参考信息

lfw图像对的读取  https://blog.csdn.net/oYeZhou/article/details/88942598


论文写作参考 https://www.researchgate.net/publication/339064753_Development_of_Deep_Learning-Based_Facial_Recognition_System 


## 参考1
> [在本教程中，我们还将使用多任务级联卷积神经网络（MTCNN）进行面部检测，例如从照片中查找和提取面部。](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)   
> [使用OpenCV进行人脸检测+ 深度学习的人脸检测MTCNN原理](https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/)

但是上面提供的都是tensorflow上的实现过程，因此需要基于pytorch的实现过程...下面这个有

---
## 参考二
谷歌 face recognition github pytorch 第一个搜索结果 [Face Recognition Using Pytorch](https://github.com/timesler/facenet-pytorch)   


---
## 参考三 关于facenet的其它参考
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



## 参考四
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch  还发布在微信公众号上[face.evoLVe：高性能人脸识别开源库，内附高能模型](https://mp.weixin.qq.com/s/V8VoyMqVvjblH358ozcWEg)   

face.evoLVe介绍
face.evoLVe 为人脸相关分析和应用提供了全面的人脸识别库，包括：   
1. **人脸对齐（人脸检测，特征点定位，仿射变换等）；** -- 使用了这一部分的代码
2. 数据处理（例如，数据增广，数据平衡，归一化等）；   
3. 各种骨干网（例如，ResNet，IR，IR-SE，ResNeXt，SE-ResNeXt，DenseNet，LightCNN，MobileNet，ShuffleNet，DPN等）；   
4. 各种损失函数（例如，Softmax，Focal，Center，SphereFace，CosFace，AmSoftmax，ArcFace，Triplet等等）；   
5. 提高性能的技巧包（例如，训练改进，模型调整，知识蒸馏等）。    

能够得到什么：
+ 这是一个库，可以直接调用人脸检测模块
+ 可以直接调用各种骨干网？-- “下图是作者开源的一个人脸识别预训练模型，骨干网用IR-50，网络Head ArcFace，Loss 函数Focal，在MS-Celeb-1M_Align_112x112数据上训练” -- “在该库的最新状态中，作者称正在MS-Celeb-1M_Align_112x112数据集上训练ResNet-50、IR-SE-50、 IR-SE-152、IR-152 ，并将很快提供下载。”








## 参考五 -- 实施参考(印度)
一篇很基础的博文：[在Amazon Sagemaker上使用Pytorch进行人脸识别（包括Jupyter笔记本代码）](https://medium.com/vaibhav-malpanis-blog/face-recognition-using-pytorch-on-amazon-sagemaker-c4f9f34c45f5)  
+ 您可以看到我们正在使用预训练的模型（alexnet）。Torchvision提供了许多预先训练的模型


## _参考六 _
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
  
就是使用 UMD-Faces/VGGFace2 两个数据集**训练 ResNet-50网络** __如果我们想要使用ResNet50作为基backbone，那么或许载入这个或者其它的而不是 torchvision中使用Imgaenet训练的ResNet网络更好？？__

## 参考七  -- 项目组织
srwpf / ResNet50-Pytorch-Face-Recognition(https://gitee.com/srwpf/ResNet50-Pytorch-Face-Recognition)
一个非常简洁的完整的小项目组织形式，可以学习。内容是训练了resnet和vgg来用作面部识别

## 参考八  -- 可用来做数据增强
AlfredXiangWu/LightCNN (https://github.com/AlfredXiangWu/LightCNN)
light_cnn出自2016 cvpr吴翔A Light CNN for Deep Face Representation with Noisy Labels，
优势在于一个很小的模型和一个非常不错的识别率。主要原因在于，
（1）作者使用maxout作为激活函数，实现了对噪声的过滤和对有用信号的保留，从而产生更好的特征图MFM(Max-Feature-Map)。这个思想非常不错，本人将此思想用在center_loss中，实现了大概0.5%的性能提升，同时，这个maxout也就是所谓的slice+eltwise，这2个层的好处就是，一，不会产生训练的参数，二，基本很少耗时，给人的感觉就是不做白不做，性能还有提升。

---



## 参考 11 
微信文章：[轻松学Pytorch-使用ResNet50实现图像分类](https://mp.weixin.qq.com/mp/appmsgalbum?action=getalbum&__biz=MzA4MDExMDEyMw==&scene=1&album_id=1345187450108411905&count=3&uin=&key=&devicetype=Windows+10&version=620603c8&lang=zh_CN&ascene=1&winzoom=1)
这篇文章做的事情很简单 可以和前面的联系在一起看 就是使用torchvision自带的 Resnet50 ((pretrained=True) 来执行图像的分类任务，应该说是应用，因为就是加载已有的标签，然后加载一个图片来显示图片的类别
> 这里首先需要加载ImageNet的分类标签，目的是最后显示分类的文本标签时候使用。然后对输入图像完成预处理，使用ResNet50模型实现分类预测，对预测结果解析之后，显示标签文本

## 参考12
同一个微信公众号的文章：[轻松学Pytorch-全卷积神经网络实现表情识别](https://mp.weixin.qq.com/s?__biz=MzA4MDExMDEyMw==&mid=2247488958&idx=1&sn=172fff12a2b0486ca3eacdcb7f5bf562&chksm=9fa862faa8dfebecec80ed4555e295896789a098574d013e2f63517193293e1dd8cb24404e3e&scene=178&cur_album_id=1345187450108411905#rd)   
这篇讲的内容多一些，虽然不是人脸识别，但是是人脸表情识别，输出的结果是：
![image.png](https://i.loli.net/2020/12/22/1kwUOIzyg9cXA6N.png)   

和我们想要的结果是类似的。此文的基础网络也是残差网络，不过是自定义block的简单残差网络EmotionsResNet，有可以参考的地方。

类似的：[用Pytorch做人脸识别](https://www.jianshu.com/p/bd855481eda7)  使用了ImageFolder导入数据集，使用了2层的卷积网络，自带的损失函数训练很简单，有参考价值的是训练获得了model，应用model的过程，"因为抓取摄像头图片使用的Opencv, Pytorch识别图像使用PIL格式，所以需要做个转换：" 他是调用摄像头的，我们不需要，直接读取新的图片即可， ‘加载我们现有的模型进行预测：’ 可以学习这一部分


## 参考13 -- 应用图像增强
[一种图像增广（Image Augmentation）方式 Sample pairing image augmentation]https://blog.csdn.net/xiaoxifei/article/details/90408243   
[Sample pairing image augmentation代码参考]https://www.reddit.com/r/MachineLearning/comments/c9kmo6/p_pytorch_implementation_of_samplepairing_testing/
> 扩展知识   
+ [训练数据太少？过拟合？一文带你领略“数据增长魔法”(上)](https://blog.csdn.net/weixin_43593330/article/details/107363707)
+ [训练数据太少？过拟合？一文带你领略“数据增长魔法”(下)](https://blog.csdn.net/weixin_43593330/article/details/107364714)
自定义 transform的方法
+ https://blog.csdn.net/qq_40467656/article/details/107979726
+ https://blog.csdn.net/dragongiri/article/details/107533668?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-2&spm=1001.2101.3001.4242
+ https://blog.csdn.net/pengchengliu/article/details/108683509?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-2&spm=1001.2101.3001.4242
+ http://spytensor.com/index.php/archives/38/?yczwva=cktlx3
