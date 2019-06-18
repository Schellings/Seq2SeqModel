# 1、前言
基于小黄鸡50w对话语料构建的SequenceToSequence生成式对话模型。

本项目是参照慕课网“NLP实践TensorFlow打造聊天机器人”实战课程实现了一个基于SequenceToSequence模型的单轮聊天机器人。

我们先来看看模型效果：

![image](http://chatbot.xielin.top/img/test.png)

# 2、我的工作
## 2.1、替换了一个全新的语料库
本例采用的是网络上公布的小黄鸡50w对话语料，相比于项目原有的电影对话语料，它语料库更为精简，对话质量更高，数据噪音较少，处理起来较为轻松
## 2.2、全新的数据清洗规则
根据语料库的特点，增加根据针对性的数据清洗规则。
## 2.3、对模型网络结构进行修改
借鉴项目原有的网络结构与其他一些SequenceTosequence模型构建时的操作，对原有的模型结构进行一定修改。详情可参照在SequenceToSequence.py中关于模型构建的部分
## 2.4、超参数调优
基本的参数列表如下：

```
# 训练轮数
n_epoch = 200
# batch样本数
batch_size = 256
# 训练时dropout的保留比例
keep_prob = 0.8

# 有关语料数据的配置
data_config = {
    # 问题最短的长度
    "min_q_len": 1,
    # 问题最长的长度
    "max_q_len": 20,
    # 答案最短的长度
    "min_a_len": 2,
    # 答案最长的长度
    "max_a_len": 20,
    # 词与索引对应的文件
    "word2index_path": "data/w2i.pkl",
    # 原始语料路径
    "path": "data/xiaohuangji50w_fenciA.conv",
    # 原始语料经过预处理之后的保存路径
    "processed_path": "data/data.pkl",
}

# 有关模型相关参数的配置
model_config = {
    # rnn神经元单元的状态数
    "hidden_size": 256,
    # rnn神经元单元类型，可以为lstm或gru
    "cell_type": "lstm",
    # 编码器和解码器的层数
    "layer_size": 4,
    # 词嵌入的维度
    "embedding_dim": 300,
    # 编码器和解码器是否共用词嵌入
    "share_embedding": True,
    # 解码允许的最大步数
    "max_decode_step": 80,
    # 梯度裁剪的阈值
    "max_gradient_norm": 3.0,
    # 学习率初始值
    "learning_rate": 0.001,
    "decay_step": 100000,
    # 学习率允许的最小值
    "min_learning_rate":1e-6,
    # 编码器是否使用双向rnn
    "bidirection":True,
    # BeamSearch时的宽度
    "beam_width":200
}
```
如果你想要对相应参数进行调参，可以在CONFIG.py文件中进行修改。
## 2.5、使用tornado作为web容器来向外暴露api接口

# 3、部署指南

首先需要部署SequenceToSequence模型构建项目。大体步骤如下：
1、安装好python3.6开发环境
2、本项目中使用的是TensorFlow-GPU 1.6.0的版本，根据各自的显卡配置，安装好cuda和cnDnn
3、（可跳过自行选择）启动命令行，使用以下命令安装虚拟环境：

```
pip install virtualenvwrapper-win -i https://pypi.mirrors.ustc.edu.cn/simple/ 
```

```
mkvirtualenv nlp  （nlp为虚拟环境名，可自己取）
```

4、进入nlp项目根路径，此处有一个requestments.txt的包文件使用以下命令批量安装包：

```
pip install -r requestments.txt -i https://pypi.mirrors.ustc.edu.cn/simple/ 
```
5、使用Pycharm或其他Python开发工具打开nlp项目（记得选择刚刚创建的虚拟环境作为项目环境）

![image](http://chatbot.xielin.top/test.jpg)

6、项目运行：
（1）如果你想要自己训练一个模型出来，可以删除data文件夹下的两个pkl文件，然后运行程序Train.py。

它会帮你去重新构建词表和进行模型的训练与保存。

（2）直接使用的话，模型由于github的限制上传大于100m的文件，可以在此处下载我训练好的模型。

链接：https://pan.baidu.com/s/1wmgRH_lmGQdzsit6gBlJzQ 

提取码：wdwt 

然后将其解压放到项目的model文件夹下。

（3）运行文件RestfulAPI.py开启web容器，监听8000端口。（启动并没有日志信息）

7、模型测试
使用postman来测试一下模型是否能够正常使用：如下图：

![image](http://chatbot.xielin.top/test01.jpg)

后端日志信息
![image](http://chatbot.xielin.top/test02.jpg)