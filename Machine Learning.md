# 机器学习 Machine Learning
**Github**：https://github.com/IvenStarry  

**学习视频网站**：B站吴恩达机器学习2022版
https://www.bilibili.com/video/BV1Pa411X76s?p=4&vd_source=6fd71d34326a08965fdb07842b0124a7
## 有监督的机器学习：回归与分类 Supervised Machine Learning Regression and Classification
### Week1 机器学习入门
#### 监督学习和非监督学习
**机器学习**：从广义上来说，机器学习是一种能够赋予机器学习的能力以此让它完成直接编程无法完成的功能的方法；但从实践的意义上来说，机器学习是一种通过利用数据，训练出模型，然后使用模型预测的一种方法

**监督学习**：算法学习预测输入、输出或X到Y的映射，学习算法从引用正确答案中学习
主要分为两类
(1)**回归算法**：学习算法从无限多的可能数字中预测数字
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407152307430.png)
(2)**分类算法**：学习算法对一个类别进行预测，无需是数字
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407152313340.png)

**无监督学习**：只有输入x而没有标签y，算法寻找数据中的某种结构或者某种模式
(1)**聚类算法**：获取没有标签的数据并尝试自动将它们分组到集群中
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407152328784.png)
(2)**异常检测**：检测异常事件，检测异常数据点
(2)**降维**：将大数据集减小为小数据集，并尽可能减小丢失的信息

**有监督学习和无监督学习的区别**：
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407152350050.png)


#### 线性回归模型
Linear Regression model
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407161330625.png)
训练集：训练模型的数据
x = 输入变量或者称为特征或输入特征
y = 输出变量或者目标变量
m = 训练样本数量
(x, y) = 单个训练样本
f = 模型函数function
y-hat = y预测值
线性回归方程: f(x) = wx + b
单变量线性回归：具有一个输入变量的线性模型

#### 代价函数
Cost function
平方误差成本函数
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407161350321.png)
f(x) = wx + b
w 权重 b 偏置项
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407161421997.png)

#### 梯度下降
梯度下降原理示意图
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407171729282.png)
参数更新规则
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407171735775.png)
链式求导法则
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407171740476.png)
梯度下降原理图
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407171737831.png)
Alpha : 学习率
设置太小 参数更新慢 时间变长
设置太大 参数更新过快 无法收敛 代价函数达不到最小值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407171744195.png)
随着梯度的变小，即使不改变Alpha值，参数的更新也会自动变得更加缓慢
批量梯度下降（Batch gradient desent）:每一次梯度的下降都使用全部的训练样本
### Week2 多输入向量回归
### Week3 分类
