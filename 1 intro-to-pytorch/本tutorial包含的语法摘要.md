





## Part 1 - Tensors in PyTorch (Exercises)

+ [pytorch 语法1 -- 利用随机函数生成一些数据](#Pytorch 语法1  -- 利用随机函数生成一些数据)
+ [pytorch 语法2 -- torch 矩阵乘法](#pytorch 语法2 -- torch 矩阵乘法)
+ [pytorch 语法3 -- 查看/改变 tensor 的形状](#pytorch 语法3 -- 查看/改变 tensor 的形状)
+ [pytorch 语法4 -- numpy数据array和pytorch数据tensor之间的转换](#pytorch 语法4 -- numpy数据array和pytorch数据tensor之间的转换 )



## Part 2 - Neural Networks in PyTorch (Exercises)

+ [【Pytorch 语法1】 -- torchvision 包及 `torch.utils.data.DataLoader()`](#[Pytorch 语法1] -- torchvision 包及 `torch.utils.data.DataLoader())

+ [【数学】softmax函数工作原理*](#[数学]softmax工作原理)
+ [【Pytorch 语法2】 -- `torch.sum()` & `Tensor.view(-1,1)` & pytorch中的矩阵维度问题](#[Pytorch 语法2] -- `torch.sum()` & `Tensor.view(-1,1)` & pytorch中的矩阵维度问题)
+ [【PyTorch 语法3】 -- 使用`nn`模块构建神经网络1(类创建)](#[PyTorch 语法3] -- 使用`nn`模块构建神经网络1(类创建))
+ [【Pytorch 语法4】 -- 使用`nn`模块构建神经网络2 调用激活函数](#[Pytorch 语法4] -- 使用`nn`模块构建神经网络2 调用激活函数)
+ [【PyTorch 语法5】 -- 自定义权重初始化](#[PyTorch 语法5] -- 自定义权重初始化)
+ [【Pytorch 语法7】 -- Using 使用`nn`模块构建神经网络3 (`nn.Sequential`法创建)](#[Pytorch 语法7] -- Using 使用`nn`模块构建神经网络3 (`nn.Sequential`法创建))
+ [【Pytorch 语法8】 -- 使用`nn.Sequential(OrderedDict)`为每层指定名称](#[Pytorch 语法8] -- 使用`nn.Sequential(OrderedDict)`为每层指定名称)



## Part 3 - Training Neural Networks (Exercises)

+ [【理论】 **error function**，与 **loss function** 的区别与联系](https://gaolei786.github.io/statistics/error.html) 
+ [【Pytorch 语法1】 -- 损失函数，对数概率分布，负对数似然损失](#[Pytorch 语法1] -- 损失函数，对数概率分布，负对数似然损失)
+ [【Pytorch 语法2】 -- pytorch的Autograd与打开/关闭梯度计算](#[Pytorch 语法2] -- pytorch的Autograd与打开/关闭梯度计算)
+ [【Pytorch 语法3】 -- 二次运行 `z.backward()` 时出现的 RuntimeError](#[Pytorch 语法3] -- 二次运行 `z.backward()` 时出现的 RuntimeError)
+ [【Pytorch 语法4】 -- 权重参数更新与实现梯度下降的 `optim` 库](#[Pytorch 语法4] -- 权重参数更新与实现梯度下降的 `optim` 库)
+ [【Pytorch 语法5】 -- python的item和pytorch的item](#[Pytorch 语法5] -- python的item和pytorch的item)



## Part 5 - Inference and Validation (Exercises)

+ [Pytorch 语法1 -- 精度计算1](#Pytorch 语法1 -- 精度计算1)
+ [Pytorch 语法2 -- 精度计算2 tensor类型的强制转换](#Pytorch 语法2 -- 精度计算2 tensor类型的强制转换)
+ [Pytorch 语法3 -- `nn.dropout()`   ](#Pytorch 语法3 -- `nn.dropout()`   )



## Part 6 - Saving and Loading Models

+ [Pytorch 语法1 -- 保存神经网络参数1](#Pytorch 语法1 -- 保存神经网络参数1)
+ [Pytorch 语法2 -- 加载神经网络参数1](#Pytorch 语法2 -- 加载神经网络参数1)
+ [Pytorch 语法3 -- 保存神经网络参数2](#Pytorch 语法3 -- 保存神经网络参数2)
+ [Pytorch 语法4 -- 加载神经网络参数2](#Pytorch 语法4 -- 加载神经网络参数2)



## Part 7 - Loading Image Data(ImageFolder) (Solution)

+ [Pytorch 语法1 -- 使用来自 torchvision 的 `ImageFolder` 加载图片](#Pytorch 语法1 -- 使用来自 torchvision 的 `ImageFolder` 加载图片)
+ [Pytorch 语法2 -- 数据变换1 `transforms.` ](#Pytorch 语法2 -- 数据变换1 `transforms.)
+ [Pytorch 语法3 -- 创建生成器 trainloader](#Pytorch 语法3 -- 创建生成器 trainloader)
+ [Pytorch 语法4 数据变换2 transforms -- Data Augmentation 数据增强](#Pytorch 语法4 数据变换2 transforms -- Data Augmentation 数据增强)





## Part 8 - Transfer Learning (Exercises)

+ [Pytorch 语法1 冻结模型参数 `.requires_grad=` + 自定义预训练网络的classifier](#Pytorch 语法1 冻结模型参数 `.requires_grad=` + 自定义预训练网络的classifier)
+ [Pytorch 语法2 python 容器：有序字典 – 与part2部分的补充](#Pytorch 语法2 python 容器：有序字典 – 与part2部分的补充])
+ [Pytorch 语法3 使用cuda训练](#Pytorch 语法3 使用cuda训练)





---

### Pytorch 语法1  -- 利用随机函数生成一些数据
上面我生成的数据可以用来得到我们的简单网络的输出。现在这些都是随机的，我们将开始使用常规数据。

+ `torch.manual_seed()`: 设置随机数种子   

+ `features = torch.randn((1, 5))`：创建 features 张量，格式是1行5列(通过一个元组类型参数指定) - feature指代输入数据；`randn()` 是指返回值按照正态分布随机分布，均值为零，标准差为1。   

+ `weights = torch.randn_like(features)`：我们为权重创建另一个matrix，值也是取自随机正态分布，使用函数 `torch.randn_like(<>)` 输入参数是另一个 tensor，方法做的是读取这个tensor的shape然后创建一个相同shape的tensor   

+ `bias = torch.randn((1, 1))`：创建bias，同样使用 torch随机正态分布方法去得到一个值shape给(1,1)

+ function `torch.sum()`：举例 `torch.sum(features * weights) + bias`   

  method on tensors `a .sum()`: 举例 `(features * weights).sum() + bias`

  *就像Numpy数组一样，PyTorch张量可以相加，相乘，相减等。 通常，使用PyTorch张量的方式与使用Numpy数组的方式几乎相同。* 





### pytorch 语法2 -- torch 矩阵乘法

> 【注意】一开始没有意识到[错误]的原因，其实 feature * weights 已经实现了tensor的矩阵乘法-然后使用使用sum方法得到乘积的元素和
> **而若成功执行了 torch.mm(features, weights) 就不用再求和 !!!!!**

[`torch.mm()`](https://pytorch.org/docs/stable/torch.html#torch.mm)

[`torch.matmul()`](https://pytorch.org/docs/stable/torch.html#torch.matmul)

```python
>> torch.mm(features, weights)

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-13-15d592eb5279> in <module>()
----> 1 torch.mm(features, weights)

RuntimeError: size mismatch, m1: [1 x 5], m2: [1 x 5] at /Users/soumith/minicondabuild3/conda-bld/pytorch_1524590658547/work/aten/src/TH/generic/THTensorMath.c:2033
```

当你在任何框架中构建神经网络时，你会经常看到这种错误, 这里所发生的是张量并不是进行矩阵乘法的正确形状。**记住，对于矩阵乘法，第一个张量的列数必须等于第二列的行数。**

`features` and `weights` 都具有相同的形状`(1、5)`。 这意味着我们需要改变 `weights` 的形状以使矩阵乘法起作用。



### pytorch 语法3 -- 查看/改变 tensor 的形状

+ `tensor.shape`: 查看名称为 `tensor` 的张量的形状   

+ `weights.reshape()`: 或以 size(a,b) 返回一个数据和 `weights` 相同的**新tensor** ；或返回克隆，比如它把数据复制到内存的另一部分。-- 存在问题是，如果进行了clone，当数据很大时浪费了很多内存是我们需要避免的

+ `weights.resize_()`: 注意以不同的shape返回**同一个tensor** -- 但是，如果新形状导致的元素数量少于原始张量，则某些元素将从张量中删除/removed（但不会从内存中删除）。如果新形状导致的元素数量多于原始张量，则新元素将在内存中未初始化/uninitialized in memory。   
这里我应该注意，方法末尾的**下划线**表示该方法是is performed **in-place**。[read more about in-place operations](https://discuss.pytorch.org/t/what-is-in-place-operation/16244) in PyTorch.   
that basically means that you are just not touching the data at all,and all you ding is changing the tensor that is sitting on top of that addressed data in memory.
+ `weights.view()`: 将 以size(a,b) 返回和 `weights` 相同数据的**新的** tensor. -- 有点不会对memory进行更改 -- 注意如果new size的元素数量和原来不一样会报错

我通常使用 `.view()`，但这三种方法中的任何一种都是适用。现在我们可以reshape `weights` ，让它有五行一列:   
```python
weights.view(5,1)
```

### pytorch 语法4 -- numpy数据array和pytorch数据tensor之间的转换  
+ 在大多数时候都要使用Numpy来preparing data and to do some preprocessing
+ 然后需要把数据输入到 网络中，因此需要将 Numpy arrays 的data数据转换成 网络使用的 Torch tensor类型的数据
+ `a = torch.from_numpy(a)`: 函数可以实现这个功能
+ `b.numpy()`: 函数可以把 tensor b 转换成 array类型数据



---

### 【Pytorch 语法1】 -- torchvision 包及 `torch.utils.data.DataLoader()`   

上面的代码块，我们把训练数据集加载到变量名`trainloader`中了，我们使用 `iter(trainloader)` 创建一个iterator -- 稍后，我们将使用它遍历数据集进行训练(区别于一次加载所有数据)，例如
```python
for images, labels in trainloader:
    ## do things with images and labels
```

我们在创建 `trainloader` 时：with a `batch` size of 64, and `shuffle=True`
```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```
+ 当我们调用一次 `iter(trainloader)`时 批处理大小(`batch_size=64`) 代表(`.next()` 代表提取第一个batch) 是我们从 data loader 读取并传入网络的 images 数量，often called a batch   
+ `shuffle=True` 告诉它在每次开始再次go through data loader 时对数据集进行洗牌(shuffle the dataset)



### 【数学】softmax工作原理

为了计算正确的**概率分布**，我们经常使用 **softmax 函数激活输出层**：
$$
\Large \sigma(x_i) = \cfrac{e^{x_i}}{\sum_k^K{e^{x_k}}}
$$

这样做是将每个输入 $x_i$ 压缩在0到1之间，并对值进行归一化，从而为提供概率总计为1的合适的概率分布。

**Exercise:** 使用 `softmax` 对一个batch(64个images)中 each example 返回概率分布：每个image返回一组10个概率分布；
> + 请注意，在执行此操作时需要注意shape: 假设有一个形状为(64, 10)的张量`a`和一个形状为(64,）的张量b，那么执行`a/b`将会报错，因为PyTorch会尝试对各列进行除法(称为**broadcasting**) -- a有10列，而b有64列，size不匹配
> + 实际上对于64个示例中的每个示例(10个输出)，只需要除以同一个值 -- 分母上的和denominator -- size为(64,1)
> + 这样，PyTorch 会将 `a` 的每一行中的10个值除以b的每一行中的一个值（b有64行，每行1个值）
> + 还要注意分母如何取和。 您需要在`torch.sum`中定义dim关键字。 **设置`dim = 0`时，将在行中求和，而`dim = 1`时，将在列中求和**。



另：与本文无关，若分母可能存在0，处理的办法是让分母加上一个非常小的数

### 【Pytorch 语法2】 -- `torch.sum()` & `Tensor.view(-1,1)` & pytorch中的矩阵维度问题

+ `torch.sum(torch.exp(x), dim=1)` ：`dim=0` takes the sum **across the rows** while `dim=1` takes the sum **across the columns** 在1行中的每一列的元素，give us a tensor--a vector of 64 elements

 + 分子是(64,10)；而分母这样是 **a** 64-long vector: it is going to divide every element in this tensor by all 64 of these values:
 + 我们想要的是分子的each row 对应去除分母的each row   


+ `torch.view(-1)`: 则原张量会变成一维的结构
+ `torch.view(参数a，-1)`: 则表示在参数b未知，参数a已知的情况下自动补齐列向量长度   

+ `tensor([9.8609e+07, 1.3485e+06, 6.7135e+09])`: 理解这个张量的维度：1个中括号代表维度为1，看这个tensor的shape可以得到 `torch.Size([3])` 代表3个元素   
+ `torch.tensor([[1,2],[3,4]])`: 这时候就有了两个中括号在这里，可以看见pytorch中的维度认为是垂直方向堆叠，**同一个array里的处于同一个维度，而不同array里的处于不同维度** -- 12/34属于同一个维度，13/24是另一个维度 [知乎](https://zhuanlan.zhihu.com/p/63802393 )   
+ 那我们再看一个例子
```python
tensor([[[0.1205, 0.1218],
         [0.1326, 0.1112],
         [0.1276, 0.1477],
         [0.1228, 0.1192],
         [0.1221, 0.1173],
         [0.1243, 0.1268],
         [0.1252, 0.1277],
         [0.1250, 0.1283]]], grad_fn=<SoftmaxBackward>)
```
我们看到，这里是有明显的三个中括号，表明的我们的**数据是3维**而不是二维，我们可以一层一层的拨开这个矩阵:最里层有八个size是2的tensor被第二个中括号包住，说明了第二个中括号内维度是8维；而我们把第二个括号里的都看成一个tensor，则它是被第三个括号包裹的，那么对于第三个括号来说，这只是1维的。而根据pytorch的size输出为：`torch.Size([1, 8, 2])`

   正好验证了这个剥开的过程，每个tensor是size为2的张量，而一共有8个这样的张量，而这8个被一个中括号括起来后，成了一个[8,2]的tensor，随后又被一个中括号括起来，相当于成了第三个维度的第一个数据。

   **可以从3维空间构成来想象**

+ 联合 CNNtutorial 里面讲维度的再理解



### 【PyTorch 语法3】 -- 使用`nn`模块构建神经网络1(类创建)

PyTorch 的 `nn` 模块使得建立神经网络更简单，使用相同的framework将能够建立更大的神经网络。在这里，将展示如何使用784个输入，256个隐藏单元，10个输出单元和softmax输出来构建与上述相同的网络。   
+ 创建一个新的类： subclass it from `nn.Module`
+ 继承 `nn.Module` 的构造器函数(`__init()__`)：因为这样PyTorch便会知道将要放入该网络的所有不同层和操作进行注册，如果不执行此部分，则它将无法跟踪要添加到的网络的内容。
+ 创建隐藏层：使用 `nn.Linear(<>,<>)` 创建一个对象：that itself has created parameters for the weights and parameters for the bias, 参数是隐藏层的输入大小和输出大小
+ 创造两个激活函数：sigmoid, softmax
+ 创建前向计算函数(`forward()`)，实现：
 + 向网络输入数据/tensor(x)
 + 隐藏层计算
 + sigmoid 激活
 + 预测输出
 + softmax激活



### 【Pytorch 语法4】 -- 使用`nn`模块构建神经网络2 调用激活函数

您可以使用 `torch.nn.functional` module 来定义sigmoid和softmax等，这可能使网络的定义更加简洁明了。实际上这是最常见定义网络的方式 as many operations are simple **element-wise functions** 因为许多 operations 都是简单的element-wise的函数。   
我们通常将此模块导入为名 `F` -- `import torch.nn.functional as F`
+ 我们可以这么做的原因是，当我们 creat these linear transformations(`self.hidden(x)->self.hidden = nn.Linear(784, 256)`)时，它会自动地 creat weights and bias metrices
+ 而我们理解 sigmoid 和 softmax 函数，它们只是 **element wise** 操作--对tensor的所有元素做相同的操作，并没有creat任何额外的参数，

```python
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
    # 1. 不需要在 init 中定义 sigmoid和softmax 激活函数
    # 2. 使用 F module 的定义函数方法
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x
```



### 【PyTorch 语法5】 -- 自定义权重初始化

使用 `nn.Linear()` 实例化层时会自动初始化权重和偏置

+ 权重等 会自动地被初始化，但是仍然可以自定义它们的初始化方式。 
+ 我们知道，在定义(实例化)各层地时候，该层的weights and biases也同时自动创建了，例如，通过 `model.fc1.weight` 可以得到第一个全连接层的权重矩阵：   
+ [pytorch中Linear类中weight的形状问题源码探讨](https://blog.csdn.net/dss_dssssd/article/details/83537765)



对于自定义初始化，我们想 **in place** 修改这些张量。 这些实际上是 **autograd变量**(autograd Variables)，因此我们需要使用 `model.fc1.weight.data` 来获取实际的张量。 一旦有了张量，就可以用 0(for biases) 或 random normal values fill它们   
+ `model.fc1.bias.data.fill_(0)`
+ `model.fc1.weight.data.normal_(std=0.01)` 



### 【Pytorch 语法7】 -- Using 使用`nn`模块构建神经网络3 (`nn.Sequential`法创建)
PyTorch提供了一种方便的方式来构建这样的网络，在这种网络中，a tensor is passed **sequentially** through operations --  `nn.Sequential` ([documentation](https://pytorch.org/docs/master/nn.html#torch.nn.Sequential))      

Using this to build the equivalent network:

+ 我们得到的模型与以前相同：784个输入单元，一个具有128个单元的隐藏层，ReLU激活，64个单元的隐藏层，另一个ReLU，然后是具有10个单元的输出层，以及softmax输出。

```python
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
```

### 【Pytorch 语法8】 -- 使用`nn.Sequential(OrderedDict)`为每层指定名称   

您也可以向 `nn.Sequential()` pass in **OrderedDict** 来命名 individual layers and operations， instead of using **incremental integers**
+ e.g. 使用 `model[0]`指代第一层, `model[0].weight` 指第一层的权重
+ `model[1]` 输出 `ReLU()` 见下
+ 请注意，字典键必须唯一(dictionary keys)，因此每个操作都必须具有不同的名称(_each operation must have a different name_)

```python
from torch import nn
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
```

这样做的好处是不仅可以通过整数也可以名称访问layer

```python
print(model[0])
print(model[1])
print(model.fc1)
print(model.relu1)
```





---

### 【Pytorch 语法1】 -- 损失函数，对数概率分布，负对数似然损失 

+ `criterion = nn.CrossEntropyLoss()`
+ `nn.LogSoftmax`或`F.log_softmax`
+ `nn.NLLLoss`   

让我们先来看看如何用PyTorch计算损失。PyTorch通过`nn`模块提供**交叉熵损失**(cross-entropy loss, `nn.crossentropyloss`)等损失。

- loss 通常被 assigned to 变量 `criterion`，因此如果我们想要使用交叉熵损失只需要写 `creterion = nn.CrossEntropyLoss()`
- 如上一部分所述，对于诸如MNIST(多)之类的分类问题，我们使用softmax函数来预测各类概率。对于softmax输出，一般使用交叉熵作为损失
- 要实际计算 loss，首先定义`criterion`，然后传入 output of your network and the correct labels -- 向`nn.CrossEntropyLoss(<>,<>)`?

进一步阅读交叉熵损失函数的文档： Looking at [the documentation for `nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss)。Something really important to note here.

> This criterion combines `nn.LogSoftmax()` and `nn.NLLLoss()` in one single class.
>
> The input is expected to contain **scores** for each class.

**这意味着我们需要将网络的原始输出(raw output-`logits` or `scores`)传递到loss函数中(如果使用交叉熵作为损失函数)，而不是softmax函数的输出。**

- This raw output is usually called the `logits` or `scores`.
- 我们使用 **logits(分数对数)** 是因为<u>softmax给出概率通常非常接近零或一</u>，但是仅使用浮点数不能准确地表示接近零或接近零的值 ([read more here](https://docs.python.org/3/tutorial/floatingpoint.html))
- 如果我们想输出概率，但又最好避免用概率来做计算，通常我们用log-概率
  + `logits` or `scores` – > `softmax()`输出的概率 –> `log_softmax()` 输出的对数概率

以我的经验，使用 `nn.LogSoftmax`或`F.log_softmax`即以**log-softmax**输出类概率分布构建模型更方便 ([documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.LogSoftmax)) 

+ 因为随后可以通过取指数 `torch.exp(output)`来获得实际概率
+ 对于log-softmax输出，就可以使用**负对数似然损失** `nn.NLLLoss`([documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss)).   

**Exercise:**
+ 建立一个返回log-softmax作为输出的模型(网络)

 + 请注意，对于`nn.LogSoftmax`和`F.log_softmax`，要适当地设置**dim关键字参数**, 考虑你想要的输出是什么，并选择适当的dim

   + `dim=0` calculates softmax across the rows, so each column sums to 1,
    + `dim=1` calculates across the columns so each row sums to 1.

+ 使用**负对数似然损失(negative log likelihood loss)**计算损失

  `nn.NLLLoss`





### 【Pytorch 语法2】 -- pytorch的Autograd与打开/关闭梯度计算

我们已经知道怎么求loss，我们怎么用loss来perform **backpropagation**?   
 + Torch提供了一个`autograd` module，用于**自动计算张量的梯度**

我们可以使用这个模块计算损失函数loss相对于所有参数的梯度：
 + Autograd通过track在张量上执行的操作(operations)来工作
 + 然后向后(backwards)进行这些操作(operations) —— 计算整个过程中的梯度
 + 当我们使用 PyTorch 创建一个网络时，所有的parameters都被初始化为`reauires_grad = True`. 即对这些参数的operations都会被记录

为了 **确保PyTorch track 张量上的操作(operations)** 并计算梯度，需要：
 +  set `requires_grad = True` on a tensor -- 告诉Pytorch你需要对一个tensor使用 `autograd`
  + 可以在创建任意一个tensor指定`requires_grad=True`关键词
  + or at any time with `x.requires_grad_(True)`



但在一些场景中(验证，测试等不需要执行反向传播的过程)我们不需要track这些对tensor的操作，如何关闭呢？

+ 例：先创建一个张量(scalar)，指定使用autograd追踪这个张量的操作
+ 使用 `with torch.no_grad()` 块代码使得在这个块内的操作的梯度记录关闭

```python
x = torch.zeros(1, requires_grad=True)  
>>> with torch.no_grad():               
...     y = x * 2
>>> y.requires_grad
False
```

+ 从全局入手打开梯度计算或者关闭梯度计算 with `torch.set_grad_enabled(True|False)`. 

+ 计算梯度：`z.backward()` **does a backward pass through the operations created `z`** -- 计算`z`相对最初变量的梯度



### 【Pytorch 语法3】 -- 二次运行 `z.backward()` 时出现的 RuntimeError
为了减少内存使用，在`.backward()`调用过程中，所有不再需要的中间结果都将被自动删除。因此，如果尝试再次调用`.backward()`，由于中间结果不存在，并且无法执行向后传递就会报错：
```python
RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.
```

如果希望可以再次调用：（结果一样吗？）
则可以通过`.backward(retain_graph=True)`进行反向传播，不会删除中间结果，随后仍可以再次`.backward()`。为了避免这个问题除最后一个向后调用外，其它所有`.backward()`都应该设置`retain_graph=True`。
+ 结果不一样？
```python
Before backward pass: 
 tensor([[-0.0003, -0.0003, -0.0003,  ..., -0.0003, -0.0003, -0.0003],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0015,  0.0015,  0.0015,  ...,  0.0015,  0.0015,  0.0015],
        ...,
        [-0.0029, -0.0029, -0.0029,  ..., -0.0029, -0.0029, -0.0029],
        [ 0.0041,  0.0041,  0.0041,  ...,  0.0041,  0.0041,  0.0041],
        [ 0.0003,  0.0003,  0.0003,  ...,  0.0003,  0.0003,  0.0003]])
After backward pass: 
 tensor([[-0.0007, -0.0007, -0.0007,  ..., -0.0007, -0.0007, -0.0007],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0030,  0.0030,  0.0030,  ...,  0.0030,  0.0030,  0.0030],
        ...,
        [-0.0057, -0.0057, -0.0057,  ..., -0.0057, -0.0057, -0.0057],
        [ 0.0082,  0.0082,  0.0082,  ...,  0.0082,  0.0082,  0.0082],
        [ 0.0005,  0.0005,  0.0005,  ...,  0.0005,  0.0005,  0.0005]])
```



### 【Pytorch 语法4】 -- 权重参数更新与实现梯度下降的 `optim` 库

训练神经网络就是在计算处loss以及loss相对于参数的梯度之后，对参数进行梯度下降。在对神经网络进行训练之前还有最后一件事，我们需要知道实际上如何使用这些梯度来更新梯度：

- 使用一个**优化器optimizer**，我们将使用它来更新梯度的权重 - an optimizer that we'll use to update the weights with the gradients
- We get these from PyTorch's [`optim` package](https://pytorch.org/docs/stable/optim.html)

- from `torch` import `optim` 库

- 使用 `optim.SGD()` 创建一个优化器 -- 需要给定参数 `model.parameters()` 表示需要使用这个优化器去更新的参数； 以及学习率`lr=`

- `optimizer.zero_grad()`

- 更新权重：`optimizer.step()`

  

```python
from torch import optim

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

在实际训练过程中需要 `optimizer.zero_grad()`：当使用相同的参数进行多次向后遍历时(multiple backwards passes with the same parameters)，梯度值将会积累(the gradients are accumulated)。 这意味需要在each training pass时将梯度归零，否则您将保留先前**训练批次(training batches)**中的梯度。

+ 注意梯度与权重参数概念的区分



### 【Pytorch 语法5】 -- python的item和pytorch的item
python的 `.item()` 用于将字典中每对key和value组成一个元组，并把这些元组放在列表中返回
例如
```python
person={‘name’:‘lizhong’,‘age’:‘26’,‘city’:‘BeiJing’,‘blog’:‘www.jb51.net’}

for key,value in person.items():
	print('key=', key, 'value=', value)
```
而pytorch中的`.item()`用于将一个**零维张量**转换成浮点数计算，比如
```python
loss = (y_pred - y).pow(2).sum()
print(loss.item())
```



---

### Pytorch 语法1 -- 精度计算1

validation 的目的是根据不属于训练集的数据衡量 模型的性能(measure the model's performance)。 此处的性能取决于开发人员。

通常这性能指标是(准确度accuracy),—— 即网络预测正确的类所占的百分比(the percentage of classes the network predicted correctly)。当然还有其它测试模型性能的指标包括( [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)) and top-5 error rate) 这里我们先只使用 accuracy指标。

> https://pytorch.org/docs/stable/generated/torch.topk.html   



```python
else:
    test_loss = 0
    accuracy = 0 # ！因为这是一个batch的统计，所以需要用+=累计出整个epoch的统计。当然，在epoch开始之前需要清零
    # 计算每个epoch的精度
    # 关闭梯度，节省内存和计算
    with torch.no_grad():
        for images, labels in testloader:
            # 在测试集的输出
            log_ps = model(images)
            test_loss = criterion(log_ps, labels)
            ps = torch.exp(log_ps) # 理论上shape是64，10
            
            top_p, top_class = ps.topk(1, dim=1)
            # 比较
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)) # 1个epoch的求出
```



```python
torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None) -> (Tensor, LongTensor)
```

> 返回**沿给定维度**的给定**输入张量**的 k 个最大元素。如果没有给出`dim=`参数，则默认选择input的最后一个维度

+ 返回一个 namedtuple (values, indices) of 2 tensors, 理解这个返回的是什么。例如返回的size是(64，1)：

  + `top_p`: 一个batch_size的64幅图的64个最大概率值分类的概率 — 1列是取一张图预测的10个输出概率值中最大的那个，

  + `top_class`: 返回的索引也是对应概率最大的索引–这个索引可以得到图像预测的lable   

+ `input` (Tensor) – the input tensor

+ `k` (int) – the k in “top-k”

+ `dim=` (int, optional) – the dimension to sort along

+ 如果`largest=False`，则返回 k 个最小元素

+ The boolean option `sorted` if True, 会确保返回的k个元素按它们本身排序
  

有了这些概率，我们可以使用 `ps.topk` 方法获得最可能的类(the most likely class)。 这将返回 `k` 个最大值。 由于我们只想要最可能的类，因此可以使用 `ps.topk(1)`  这将返回top-k值和top-k索引的元组。 如果最高值是第五个元素，我们将取回4作为索引。

+ 注意上面官方文档使用的函数是`input`作为函数输入，这里使用的是 `input.topk(dim)` input 作为句柄来调用

```python
# 返回输出概率最大的值和-所有
top_p, top_class = ps.topk(1, dim=1)

print(top_class.shape)  # torch.Size([64, 1]) 1列里的值是对应概率最大的索引
# Look at the most likely classes for the first 10 examples
print(top_class[:10,:])
```

```
torch.Size([64, 1])
tensor([[1],
        [1],
        [7],
        [1],
        [1],
        [7],
        [7],
        [1],
        [1],
        [7]])
```

求出64幅图像对应的类的索引(tensor(torch.Size([64, 1])), 现在我们可以检查预测的类的结果是否与标签匹配。将`top_class`和`label`等同起来很容易做到，但是我们必须注意 tensor 的 shape

- 如上 `top_class` is a 2D tensor with shape (64, 1) 而 labels is 1D with shape (64) 64个elements
- 为了进行equality比较，`top_class`和 `label` 必须具有相同的形状

if we do

```python
equals = top_class == labels
```

- `top_class`的shape是(64,1), `labels`的shape是(64) 
- equals 将被赋一个shape为(64, 64)的值 **广播机制**
  - 原因：语句会将`top_class`的每一行中的那1个元素与`labels`中的每个元素(共64个)进行比较，然后为每一行返回64个 True/False布尔值

正确的做法是对`labels`进行整型 `labels.view()`

```python
equals = top_class == labels.view(*top_class.shape)
print(equals.shape) #torch.Size([64, 1])
```

```
torch.Size([64, 1])
```



### Pytorch 语法2 -- 精度计算2 tensor类型的强制转换
由 `torch.ByteTensor` 转换为 `torch.FloatTensor`   

```python
  equals.type(torch.FloatTensor)
```

按照 accuracy 的定义：the percentage of classes the network predicted correctly. 我们现在需要计算正确预测的概率。   
我们知道变量 `equals` 是 binary values，即0或者1；这意味着，**只要我们对所有值求和，然后除以值的数目，便可以得出正确预测的百分比**
+ 这个操作与取这组tensor值的平均值的操作相同，因此我们只要调用`torch.mean`就可以求 accuracy了
  + 不过注意，如果我们简单使用 `torch.mean(equals)` 求 torch tensor的平均值会报错
  
  
  ```python
    RuntimeError: mean is not implemented for type torch.ByteTensor
  ```
  
  
  + 这是因为 `equals` 的类型是 `torch.ByteTensor` 而函数`torch.mean`不能对这个类型的tensor使用。我们需要将 `equals` 的类型转换为float类型tensor：
  
  ```python
  equals.type(torch.FloatTensor)
  ```


+ 另外注意：当我们使用`torch.mean`时，它返回一个标量张量(scalar tensor)，要获取实际值作为浮点数，我们需要执行 `accuracy.item()`
 + 而`pytorch中的.item()`用于将一个零维张量转换成浮点数计算

```
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')
```

+ 注意不是二分类不能这样计算精度！！

### Pytorch 语法3 -- `nn.dropout()`   

减少过拟合最常用的方法(除了“early-stopping”外)是 **dropout**，即随机丢弃一些输入单元。

- 这迫使网络在权重之间共享信息(share information between weights)，增加了泛化新数据的能力
- 在PyTorch中使用`nn.Dropout`模块可以很容易地添加dropout

```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
        # 在构造器中创建dropout，只要创建这个神将网络就会使用这个避免过拟合的方法
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # 确保输入x被flatted
        x = x.view(x.shape[0], -1)
        
        # 在网络中使用 dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        # 输出层不需要 dropout
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
```

在训练过程中，我们希望使用dropout来防止过拟合，但在inference过程中，我们希望使用完整的网络(the entire network.)。因此，我们需要在验证、测试(validation, testing,)和使用网络进行预测时**关闭dropout(turn off dropout)**!!

- 为此，可以使用

  ```
  model.eval()
  ```

  这将网络模型(model)设置为评估模式(evaluation mode) -- where the dropout probability is 0

- 注：如果想要再次开启网络的 dropout 功能使用命令`model.train()`将网络设为训练模式(train mode) 



---

### Pytorch 语法1 -- 保存神经网络参数1

可以想象，如果每次都在需要使用网络时才去训练网络都是不切实际的，我们可以保存经过训练的网络，然后再加载它们以进行更多训练或将其用于预测。

- save it to a file：我们保存模型的方式是实际上保存一个字典`state_dict`
  - 我们的模型的所有参数存储在模型的  `state_dict` 中，这是一个字典文件
  - 所有参数是指：模型**每一层的权重和偏差(键)**的矩阵

我们先来看看刚刚训练过的**模型的结构**，以及模型的**状态字典(`state_dict`)**的键

```python
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())
```

```
Our model: 

 Network(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): Linear(in_features=512, out_features=256, bias=True)
    (2): Linear(in_features=256, out_features=128, bias=True)
  )
  (output): Linear(in_features=128, out_features=10, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
) 

The state dict keys: 

 odict_keys(['hidden_layers.0.weight', 'hidden_layers.0.bias', 'hidden_layers.1.weight', 'hidden_layers.1.bias', 'hidden_layers.2.weight', 'hidden_layers.2.bias', 'output.weight', 'output.bias'])
```

通过这个状态字典的键我们可以了解当保存模型时实际上是保存了什么，这也和 `nn.Linear()` 函数有关，当我们使用`nn.Linear()`创建隐藏层的时候，同时它会字典创建权重矩阵和bias，而这些就是被保存的参数。



而保存神经网络参数最简单的方式是使用 `torch.save(<>, <>)` 保存模型的参数。例如，我们可以将它保存到文件 `checkpoint.pth` 中

+ 执行这个语句之后，会在当前工作目录下生成一个 `checkpoint.pth` 文件

```
torch.save(model.state_dict(), 'checkpoint.pth')
```



### Pytorch 语法2 -- 加载神经网络参数1

+ 读取模型参数字典 `torch.load(<>)`
+ 向模型导入参数 `model.load_state_dict(state_dict)`
  + 若成功会返回 `<All keys matched successfully>`
  + 注意，正确加载网络参数需要`model`的结构与参数数量相匹配！

```python
state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())
```

```
odict_keys(['hidden_layers.0.weight', 'hidden_layers.0.bias', 'hidden_layers.1.weight', 'hidden_layers.1.bias', 'hidden_layers.2.weight', 'hidden_layers.2.bias', 'output.weight', 'output.bias'])
```

```python
model.load_state_dict(state_dict)
```

```
<All keys matched successfully>
```

### Pytorch 语法3 -- 保存神经网络参数2

保存和加载神经网络参数

似乎很简单，但通常还是复杂的，因为只有当 **model architecture** 与 **checkpoint architecture** 完全相同时，才能正确加载 `state_dict`, 如果创建的model具有不同的 architecture，则会报错：

- 但我们在刚刚save网络的状态字典时并没有save网络的结构信息，因此若**没有网络结构只有网络字典**我们还是不能 rebuild 这个网络

```python
# Try this
# [] 表示隐藏层的数量，这里随便给出的与训练的模型是不一致的
model = fc_model.Network(784, 10, [400, 200, 100])
# This will throw an error because the tensor sizes are wrong!
model.load_state_dict(state_dict)
```

```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-12-d859c59ebec0> in <module>
      2 model = fc_model.Network(784, 10, [400, 200, 100])
      3 # This will throw an error because the tensor sizes are wrong!
----> 4 model.load_state_dict(state_dict)

E:\Anaconda3\lib\site-packages\torch\nn\modules\module.py in load_state_dict(self, state_dict, strict)
    828         if len(error_msgs) > 0:
    829             raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
--> 830                                self.__class__.__name__, "\n\t".join(error_msgs)))
    831         return _IncompatibleKeys(missing_keys, unexpected_keys)
    832 

RuntimeError: Error(s) in loading state_dict for Network:
	size mismatch for hidden_layers.0.weight: copying a param with shape torch.Size([512, 784]) from checkpoint, the shape in current model is torch.Size([400, 784]).
	size mismatch for hidden_layers.0.bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([400]).
	size mismatch for hidden_layers.1.weight: copying a param with shape torch.Size([256, 512]) from checkpoint, the shape in current model is torch.Size([200, 400]).
	size mismatch for hidden_layers.1.bias: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([200]).
	size mismatch for hidden_layers.2.weight: copying a param with shape torch.Size([128, 256]) from checkpoint, the shape in current model is torch.Size([100, 200]).
	size mismatch for hidden_layers.2.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([100]).
	size mismatch for output.weight: copying a param with shape torch.Size([10, 128]) from checkpoint, the shape in current model is torch.Size([10, 100]).
```

因此，实际上为了完全按照训练时的模型重建模型，有关 model architecture 的信息需要与 state dict 一起保存在checkpoint文件中。 为此，您需要构建一个字典，其中包含您需要完全重建模型的所有信息。

记得保存模型参数的语句是 `torch.save(<>, <>)`：

- 第一个参数是一个字典文件，若为`model.state_dict()`就是模型参数字典，我们需要重新构建一个同时包含model结构信息和参数信息的字典，例如命名为`checkpoint`
  - 网络结构包含三个部分：input_size， output_size， hidden_layers(size) ——是一个列表文件
  - 另外加上模型参数字典： `model.state_dict()`
- 第二个参数是保存目录，例如当前工作目录

```python
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
```

####  `each.out_features`
首先 each 代表的每一个隐藏层对象，即 `nn.Linear()` 所创建的，`nn.Linear()` 是用于设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]，不同于卷积层要求输入输出是四维张量。让我们来看看PyTorch中的`nn.Linaear()`

```python
torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
```

 + `in_features` – size of each input sample

 + `out_features` – size of each output sample

 + `bias` – If set to False, the layer will not learn an additive bias. Default: True

这里的关键在于使用`nn.Linear()`的对象实例调用参数 `each.out_features`得到隐藏层的units数目，例如第一层隐藏层的`in_features`是输入层的个数，`out_feature`是第一隐藏层的单元数



### Pytorch 语法4 -- 加载神经网络参数2

现在，保存的checkpoint文件具有所有必要的信息来重建训练后的模型。根据需要轻松地将(加载模型参数到模型中)设置为函数：

- 读取字典 `checkpoint`
- 使用`fc_model.Network(<>, <>, <[]>)` 创建一个全连接的网络
- 使用 `model.load_state_dict()` 加载神经网络参数

```python
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
```

- 注意这个加载模型的函数必须根据实际情况编写！







---

### Pytorch 语法1 -- 使用来自 torchvision 的 `ImageFolder` 加载图片   

加载自定义图像数据最简单的方法是使用 `torchvision` ([documentation](http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder)) 中的 `datasets.ImageFolder`  
通常，像这样使用 `ImageFolder`：   

```python
dataset = datasets.ImageFolder(root='path/to/data', transform=transform)
```
+ 其中参数 `path/to/data` 是数据目录的文件路径   
+ `transform` 是由 `torchvision`中的 `transforms` 模块构建的处理步骤的列表(a list of processing steps)
+ `ImageFolder`希望文件和目录的构造如下:   


```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

即要求数据文件**按照其所属的类/class**都有自己的目录(例如 it's own directory `cat` and `dog`)，则这个方法加载时**会使用从目录名称中获取的类来标记(label)**图像
+ 例如 图像`123.png`将 be loaded with the class label `cat`.



### Pytorch 语法2 -- 数据变换1 `transforms.`

先来看我们在前面的笔记本加载数据时有：

```python
from torchvision import datasets, transforms
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
```
使用 `ImageFolder` 加载图片时有：

```python
dataset = datasets.ImageFolder('path/to/data', transform=transform)
```
即当使用 `ImageFolder` 加载数据时，需要定义一些数据所要做的**转换(transform)**。例如：
+ 例如，文件中图像的size不同，但是我们需要它们都具有相同的size以**输入网络**进行训练。则可以做的 transform 为：
  + **resize** them with `transforms.Resize()`
  + **crop** with `transforms.CenterCrop()` or `transforms.RandomResizedCrop()`
+ **必要**：图像被加载进的格式一般是 `pillow image` 我们还需要使用 `transforms.ToTensor()` **将图像转换为PyTorch张量**   

通常，我们需要把这些transform操作组合到一个pipline中 with `transforms.Compose([])` -- 它 accept 一个 transform list 并按顺序运行。举个例子，我们定义一个对数据的transform操作序列：缩放(scale)，然后裁剪(crop)，然后转换为张量:   

```python
transform = transforms.Compose([transforms.Resize(255),   # (255,255)
                                transforms.CenterCrop(244), #(244,244)
                                transforms.ToTensor()])
```


有许多可用的转换，我将在后面介绍更多。



### Pytorch 语法3 -- 创建生成器 trainloader
先来看我们在前面的笔记本加载数据时有：   

```python
import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

以及在这里，加载`ImageFolder`后，  

```python
dataset = datasets.ImageFolder('path/to/data', transform=transform)
```

您必须将其传递给DataLoader:   

```python
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```
+ `torch.utils.data.DataLoader()` takes 一个数据集(例如从`ImageFolder`获得的数据集`dataset`）
+ 返回batch数量的的图像和标签(images and the corresponding labels)，参数：

  + 例如 `batch_size (per loop through)` 大小 

  + 以及是否在每个epoch之后对数据进行shuffle， 我们希望再次使用数据集的时候，读取数据时顺序是不同的



#### dataloader 是一个 generator

这里的 dataloader 是一个生成器(generator)(https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/)。 **要从生成器中获取数据**，需要
+ 遍历它(loop through it) -- 将它变成一个 迭代器，实际上是不停调用下面的`next()`
+ 或将其转换为**迭代器(iterator)**并调用`next()`(convert it to an iterator and call `next()`)   


```python
# Looping through it, get a batch on each loop 
for images, labels in dataloader:
    pass

# 每次循环等价于
# Get one batch
images, labels = next(iter(dataloader))
```



### Pytorch 语法4 数据变换2 transforms -- Data Augmentation 数据增强

> 阅读文档了解 `transformas` https://pytorch.org/docs/stable/torchvision/transforms.html 
>
> 你可以在此处找到所有可用转换的列表(the available transforms here)(http://pytorch.org/docs/0.3.0/torchvision/transforms.html)。
>
> + [PyTorch 学习笔记（三）：transforms的二十二个方法()](https://zhuanlan.zhihu.com/p/53367135)

训练神经网络的一个常见策略是在输入数据本身中引入随机性(randomness)。例如，我们可以在训练期间**随机旋转，镜像，缩放和/或裁剪图像(randomly rotate, mirror, scale, and/or crop your images during training)**. 这有助于网络的泛化，因为它看到的是相同的图像，但位置不同，大小不同，方向不同等。   

使用`transforms`定义**随机旋转，缩放和裁剪，翻转图像**(randomly rotate, scale and crop, then flip your images)，如下：   

```python
train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                       transforms.RandomResizedCrop(224), 
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.ToTensor(), 
                                       transforms.Nomalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])
```

【注】在设置test数据的transform时候，一开始只设置了 `transforms.Resize(224)` 没有设置 `transforms.CenterCrop(224)` 结果运行的时候报错

```python
RuntimeError: stack expects each tensor to be equal size, but got [3, 237, 224] at entry 0 and [3, 224, 400] at entry 1
```

区分两个：

- ```
  torchvision.transforms.Resize(size, interpolation=2)
  ```

  - size (sequence or int) – Desired output size. If size is a sequence like (h, w), output size will be matched to this. If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).

- ```
  torchvision.transforms.CenterCrop(size)
  ```

  - size (sequence or int) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.



####  关于 normalize images 输入图像__标准化__

传入图像数据时通常还需要使用`transforms.Normalize`对图像进行标准化(normalize)。向 `transforms.Normalize()` 传入
+ 均值列表(a list of means)
+ 和标准差列表(a list of standard deviations)   

然后将颜色通道标准化/归一化(the color channels are normalized)，如下所示   

什么是图像normalize？例如对于任意颜色通道，normalize就是：

$$
\rm input[channel] = \frac{(input[channel] - mean[channel])}{std[channel]}
$$

+ 图像值(input[channel])减去平均值：**中心化**(centers the data around 0)
+ 再除以图像的标准差std将值压缩到-1到1之间：归一化
+ **标准化/归一化/normalize有助于使网络工作权重保持接近零，从而使反向传播更加稳定。 如果没有规范化，网络将往往无法学习**   





---

### Pytorch 语法1 冻结模型参数 `.requires_grad=` + 自定义预训练网络的classifier

我们下面要做的就是保留/不更新这个网络的features部分的参数，retrain classifier部分的参数：

1. **冻结feature parameters**
     + 遍历模型(特征提取器部分)的参数
     + 执行 `param.requires_grad = False`：实现当我们通过网络计算时，it's not going to calculate the gradient：~~即不跟踪 operations~~
       + 在第5节我们在验证时候使用的是 `with torch.no_grad():` 关闭梯度/操作追踪的
       + 这样就保证 our **feature parameters** 不会更新，同时能够提高训练速度
     + **注：根据下面`model.classifier`表示分类器部分，这里 `model.parameters()` 就仅仅指features部分的参数？**     

2. 替换这个模型中的 classifier：replace the classifier with our own classifier
     + 使用Pytorch的`Sequential`模块定义一个classifer结构：给它一个要执行的**不同operations的列表**(a list of different operations you want to do)，然后classifer将自动按顺序 pass a tensor through them:
     + 这些有序操作用有序字典创建可以给网络的层命名：you can pass in an ordered dict to name each of these layers
     + 即定义一个小型的全连接网络含输出层



### Pytorch 语法2 python 容器：有序字典 – 与part2部分的补充

```python 
class collections.OrderedDict([items])
```
> https://www.cnblogs.com/zhenwei66/p/6596248.html

+ 输入是一个**列表类型**，列表的元素/items是元组类型，元组的两个元素，键值对(eg `'fc1', nn.Linear(1024, 500)`)

```python
# 首先冻结导入模型的！参数！
for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, 500)),    # feature部分的输出是1024
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(500, 2)), 
                            ('output', nn.LogSoftmax(dim=1))
                          ]))

# 更换导入的模型的分类器部分！
model.classifier = classifier
```

```
model.classifier
----
Sequential(
  (fc1): Linear(in_features=1024, out_features=500, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=500, out_features=2, bias=True)
  (output): LogSoftmax(dim=1)
)
```

- 注意替换对应的全连接层的方法！对于`densenet121`网络，通过查看model知道最后的分类器的名字是 `classifier` 因此用 `model.classifier`来替换，而例如 `resnet50`，下载查看后它的最后一层的名字是`fc`则需要使用 `model.fc=classifier`来替换！

+ 注 `model.classifier.parameters()` 表示model的分类器部分的参数

### Pytorch 语法3 使用cuda训练

**接下来：建立模型后，我们需要训练这个模型 -- 实际上是使用我们的数据集仅训练`classifier`部分的参数。**

但是，现在我们正在使用真正的深度神经网络。 如果您尝试像以前那样在CPU上进行训练，这将需要很长时间。instead，我们将使用GPU进行计算。在GPU上线性代数计算是并行完成，从而使训练速度可以提高了100倍。而且还可以在多个GPU上进行训练，从而进一步减少了训练时间。PyTorch和几乎所有其他的深度学习框架一样，使用[CUDA](https://developer.nvidia.com/cuda-zone)高效地计算GPU向前和向后的传递(forward and backwards passes)。

+ 在PyTorch中，可以使用 `model.to('cuda')` 将模型参数(model parameters)和其他张量移动到GPU内存中,
+ 可以用 `model.to('cpu')` 将它们从GPU移回来，这通常是你需要在PyTorch之外对网络输出进行操作时需要做的   

也可以使用 
+ `model.cuda()` 或者 `images.cuda()` 等移入GPU
+ 使用 `model.cpu` 返回cpu

我们可以查看是否可以在pytorch中使用本机的GPU

```python
torch.cuda.is_available()
---
True
```

于是实际上我们可以编写与设备无关的代码，让它判断能否使用cuda，如果启用了该代码，它将自动使用CUDA，如下所示：

```python
# at beginning of the script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

...

# then whenever you get a new Tensor or Module
# this won't copy if they are already on the desired device
input = data.to(device)
model = MyModule(...).to(device)
```





























