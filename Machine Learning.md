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
#### 多维特征
多元线性回归（区别于多元回归）
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407181042940.png)

#### 矢量化
NumPy点函数在计算机中使用并行硬件，加快计算速度
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407181258340.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407181255933.png)

#### 用于多元线性回归的梯度下降法
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407190928259.png)
在线性回归可以替代梯度下降法的方法————正规方程法
这种方法直接联立成本函数对w和对b偏导值分别等于0直接求解极值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407190933758.png)

#### 特征缩放
当有不同的特征，取值范围差别较大时，可能导致梯度下降运行缓慢，但我们可以重新缩放不同的特征，是它们都具有可比较的取值范围
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407190949944.png)
1.除以区间最大值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407190951860.png)
2.均值归一化
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407190952722.png)
3.Z分数归一化 以0为中心
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407190953228.png)
需要进行特征缩放的例子
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407190954220.png)

#### 判断梯度下降是否收敛
通过观察学习曲线下降情况或使用epsilon自动测试梯度下降是否正常
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407191000005.png)

#### 设置学习率
不同的学习曲线特点对应着可能出现的不同的错误情况
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407191007527.png)
**调参思路**：观察学习曲线下降情况，令学习率三倍增长，直至取到一个成本函数不会下降太慢或往返震荡之间的适当的学习率
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407191007329.png)

#### 特征工程
通过直觉或一些先验知识设计产生新的特征，通常采用转化或者是组合原特征的方法。
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407191013769.png)

#### 多项式回归
通过多元线性回归和特征工程结合的思想可以帮助我们将曲线、非线性函数拟合入数据当中
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407191019136.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407191019335.png)

### Week3 分类
#### 动机与目的
简单利用线性回归设定决策分界线来完成二元分类问题是有问题的，尤其是有偏差较大的样本存在，影响线性回归方程的建立，推动决策分界线右，从而大幅减小预测的准确率
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260048753.png)
逻辑回归名字里有回归二字但实际解决的是分类问题，尤其是输出标签为1或0的二元分类问题

#### 逻辑回归
将数据集拟合成一条看起来像S形曲线的分界线
**逻辑回归公式**Sigmoid
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260119079.png)

#### 决策边界
在决策边界左右两侧，y预测的类别不同
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260127358.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260127925.png)
通过多元线性回归和特征工程结合的思想可以拟合出更加复杂的非线性决策边界

#### 逻辑回归中的代价函数
逻辑回归不使用平方差损失函数，因为逻辑回归方程在平方差损失函数的展开曲线不是凸类型，更新参数很容易陷入局部极小值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260139187.png)
如果样本标签为1且f预测趋向于1，则误差趋向于0
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260139076.png)
如果样本标签为0且f预测趋向于0，则误差趋向于0
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260139080.png)

#### 简化逻辑回归代价函数
**最大似然估计**：已知实验结果，反推什么环境条件下得到实验结果的概率最大  
抛硬币得到7次正面，三次反面，建立似然函数，求取L最大值时theta的取值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260156562.png)
选择这个成本函数的原因是使用最大似然估计的思想推导出来的，简化思想，样本标签只为1或0，所以可以直接抵消另一种情况下选择的损失函数
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260201130.png)

#### 实现梯度下降
逻辑回归的成本函数对w和对b的偏导值和前面线性回归的偏导公式一样，但要注意逻辑回归的fwx=sigmoid(wx+b)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260212322.png)

#### 过拟合问题
**泛化**：模型在从未见过的全新示例上也可以做出良好的预测
**过拟合（高方差）**：如果将模型拟合到略有不同的数据集，会得到完全不同的预测或高度可变预测，泛化能力过差
**欠拟合（高偏差）**：模型对训练集的样本预测偏差较大，拟合效果太差，偏离最优模型
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260229291.png)
特征过多导致过拟合，特征过少欠拟合
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260230792.png)

#### 解决过拟合
**解决方法**：
1. **获得更多数据**：学习算法适应一个波动较小的函数
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260238433.png)
2. **使用更少的特征**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260238005.png)
3. **正则化**：保留所有特征，防止特征产生过大的影响，鼓励学习算法缩小参数值，却不要求参数设置为0（通常只对w1,w2...正则化，对b正则化对模型影响较小）
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260239289.png)

#### 正则化
**正则化参数lambda**：惩罚力度
在实际的实验中，由于不清楚具体哪个特征会造成过拟合，因此成本函数应加上所有w值的正则化项，这样成本函数会由平方误差成本和正则化项两部分相加（不加b的正则化项，因为b对模型影响太小，添加b的正则化项意义不大）
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260245091.png)

#### 用于线性回归的正则方法
参数更新分为常规更新和新增更新，其中常规更新和无正则化的线性回归参数更新一致，加入正则化后每次参数更新时，w的值前会先乘以一个接近1的系数再减去常规更新完成参数更新
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260257387.png)

#### 用于逻辑回归的正则方法
类似于线性回归的正则方法参数更新规则，仅这里的f函数不同
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407260305397.png)

## 高级学习算法 Advanced Learning Algorithms
### Week1 神经网络
#### 神经元和大脑
生物学中的神经元与神经网络的神经元对比
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408031256457.png)

#### 需求预测
输入层--隐藏层--输出层
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408031436253.png)
多层感知器
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408031436044.png)

#### 图像感知
不同的隐藏层关注的图像细节有所不同，不同特征最终拼凑在一起识别图像
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408051757944.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408051757184.png)

#### 神经网络中的网络层
网络层数计算：除了输入层以外的其他层
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408060952530.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408060952788.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408060952598.png)

#### 更复杂的神经网络
激活值a：上标为第几层的参数，下标为第几个参数
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408060954538.png)

#### 神经网络前向传播
计算每层的激活值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408060956278.png)

#### 如何用代码实现推理
与逻辑回归类似，设置阈值输出预测结果
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408060956723.png)

#### Tensorflow中的数据形式
Tensorflow使用矩阵表示数据，而非一维数组，这样在内部的计算效率更高
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408060957831.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408060958428.png)

#### 搭建一个神经网络
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408060958673.png)

#### 单个网络层上的前向传播
前向传播的python实现
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061651258.png)

#### 前向传播的一般实现
dense函数：输入上一层的激活，给定当前层的函数，返回下一层的激活  
sequential函数：按顺序将密集层串联起来，便于神经网络实现前向传播
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061659626.png)

#### 强人工智能
**AI**
1. **ANI**(人工狭义智能)：用于完成一个特定的任务，完成度较高，有较高价值
2. **AGI**(通用人工智能)：构建可以做任何普通人能做的人工智能系统
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061643611.png)

视觉与听觉的神经元没有本质区别，而是数据的输入导致的神经元内参数不同，从而导致功能不同，例如，听力丧失的人，听觉皮层会通过学习看到的视觉图像学会看的能力，视觉能力会增强
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061644163.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061645553.png)

#### 神经网络为何如此高效 
神经网络中前向传播的矢量化实现  
matmul矩阵乘法函数
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061651474.png)

#### 矩阵乘法
向量点积
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061653766.png)
向量矩阵乘法
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061653265.png)
矩阵矩阵乘法
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061653768.png)

#### 矩阵乘法规则
矩阵形状 (m,a)*(a,n) = (m,n) (a长度必须相等)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061654776.png)

#### 矩阵乘法代码
matmul原理
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408061657864.png)

### Week2 神经网络训练
#### Tensorflow实现

#### 模型训练细节

#### Sigmoid激活函数的替代方案

#### 如何选择激活函数

#### 多分类问题

#### Softmax

#### 神经网络的Softmax输出

#### Softmax的改进实现

#### 多个输出的分类

#### 高级优化方法

#### 其他的网络层类型

#### 什么是导数

#### 计算图

#### 大型神经网络案例

### Week3 应用机器学习的建议
#### 决定下一步做什么

#### 模型评估

#### 模型选择&交叉验证测试集的训练方法

#### 通过偏差与方法进行诊断

#### 正则化、偏差、方差

#### 制定一个用于性能评估的基准

#### 学习曲线

#### 决定下一步做什么

#### 方差与偏差

#### 机器学习开发的迭代

#### 误差分析

#### 添加更多数据

#### 迁移学习--使用其他任务中的数据

#### 机器学习项目的完整周期

#### 公平、偏见与伦理

#### 倾斜数据集的误差指标

#### 精确率与召回率的权衡

### Week4 决策树
#### 决策树模型

#### 学习过程

#### 纯度

#### 选择拆分信息增益

#### 整合

#### 独热编码One-hot

#### 连续有价值的功能

#### 回归树

#### 使用多个决策树

#### 有放回抽样

#### 随机森林

#### XGBoost

#### 何时使用决策树