# 手写数字识别项目（Pytorch）

![](media/e876598a24165bfbe40cc7ad16b47c75.png)

# MNIST数据集

![](media/210926badff54f63143b63ebcc30a4fa.png)

![](media/8d87cc9f23cc6d2756274c3bd74fd8d9.png)

# 输出节点的归一化

![](media/678d829a94ba3f23ab6c5417e1ac6696.png)

![](media/7842d4cf5d83f6cc37bdc8c49c08fc73.png)

![](media/162e616300efc1aa328fe97f423d00e0.png)

# 激活函数

![](media/300f4a1629947a250a5fd6746cb973be.png)

# 过程描述

将图片拆分成一维像素阵列，输入到神经网络，通过节点间的计算公式图像信息传输到输出层，通过Softmax归一化，得到一个概率分布，再通过大量图像数据的训练，不断调整网络参数，让概率分布更接近真实值，故神经网络的本质就是一个数学函数，训练过程就是调整函数的参数。

# 环境配置

Python=3.10.14，其余配置文件

conda create -n HDR python==3.10.14

conda activate HDR

conda install pytorch== 2.4.0 torchvision==0.19.0 matplotlib==3.8.4

上述代码如果无法使用的话，则运行下列代码。

cd 自己本地的代码目录 （或者在本地代码目录的上方打开cmd）

pip install -r requirements.txt

# 实现结果

![](media/88a94dc9dae7c245894ac49636486020.png)

![](media/937064158d129b703d19294b3f45bf78.png)![](media/181dba29b17be417150570df007b0add.png)

参考链接：https://www.bilibili.com/video/BV1GC4y15736/
