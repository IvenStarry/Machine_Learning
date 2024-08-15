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
`model.compile`语句定义了一个损失函数，并指定了优化器。  
`model.fit`语句运行梯度下降，并将权重拟合到数据上。
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
1. `Sequential` 连接网络层 `Dense` 创建网络层
2. `complie` 指定损失函数和优化器
3. `fit` 输入样本X和样本标签Y 输入时期Epoch(梯度下降次数，参数更新次数) 执行梯度下降
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408080949663.png)

#### 模型训练细节
1. 指定逻辑回归的输入到输出函数
2. 指定损失函数和成本函数 损失函数L是学习算法对于单个训练样例预测结果和样本标签的损失，成本函数实是整个训练集上计算的损失函数的平均值(这里的成本函数 J 中的 W B 不是单纯的向量，指的是每一层每个神经元所对应的参数的集合)
3. 使用一种特定的梯度下降算法最小化 w,b 的成本函数J，将其作为参数 w,b 的函数最小化
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408080950647.png)
- 逻辑损失也称为**二元交叉熵** binary cross entropy
- **均方损失函数** mean squared error

#### Sigmoid激活函数的替代方案
- **线性激活函数** (相当于没有使用激活函数)
- **ReLU激活函数**
- **Sigmoid激活函数**
- **Sofxmax激活函数**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408081002597.png)

#### 如何选择激活函数
1. 对于**输出层**：
- sigmoid函数：用于二元分类问题
- 线性激活函数：用于取值范围任意的回归问题(股价涨跌)
- ReLU激活函数：用于只能取非负值的回归问题(房价预测)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408081500672.png)
2. 对于**隐藏层**：常使用**ReLU函数**，因为ReLU函数只有左侧区域平坦，而Sigmoid函数在z值较小或较大时均平坦，在平坦的区域梯度下降较慢(尽管梯度下降优化的是成本函数 J 而非激活函数，但激活函数是计算的一部分，直接导致了成本函数 J 有更多的地方也是平坦的，并且梯度很小，减慢了学习速度)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408081501532.png)

#### 为什么模型需要激活函数
如果每一层都使用线性激活函数，这样与线性回归没什么不同，这个神经网络无法适应比线性回归模型更复杂的特征；同理，如果隐藏层全使用先行激活函数，输出层采用sigmoid函数，神经网络等同于逻辑回归模型，无法完成其他复杂任务。
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408081514123.png)

#### 多分类问题
对于样本标签有多个结果可能时，选择不同的分类算法，实现决策边界
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408081535048.png)

#### Softmax
用e为底数的作用时保证最终获得概率值为正数  
若 N=2 softmax回归和逻辑回归最终计算的结果基本相同
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408081608725.png)
计算损失函数时，根据样本的标签值，损失值仅被计算一次，我们希望模型对于样本标签这一类别的预测概率尽可能大，从而损失尽可能小
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408081610165.png)

#### 神经网络的Softmax输出
输出层的激活值特点
- 逻辑回归：只跟 z1 有关
- softmax：跟 全部z值 有关
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408081636001.png)

多分类任务的损失函数:sparse categorical cross entropy
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408081641837.png)

#### Softmax的改进实现
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408090008772.png)
因为计算机的硬件缺陷，保存数据总会损失一部分精度，如果逐层计算激活值a是有误差的，带着误差的a去计算loss误差就更大了，由于CPU精度高于存储精度，所以可以一起计算a，从代码层面规避计算机硬件缺陷
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408090010241.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408090026774.png)
具体操作如下：将输出层设置为仅使用线性激活函数，并将激活函数和交叉熵损失到图片下面所示的损失函数规范当中，在 `from_logits` 参数中设置
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408090017343.png)
> 注意现在的神经网络输出是 z ，而不是预测概率 a

#### 多个输出的分类
- **多类分类**：每个实例里标签有多个类别，神经网络预测最终结果只能是多个标签中的一个
- **多标签分类**：每个实例有多个标签同时存在，如一张图片中识别是否有车，是否有行人，是否有公交车
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408090047636.png)

多标签分类输出层的激活值是一个向量，分别表示多个标签的概率
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408090051825.png)

#### 高级优化方法
**自适应学习率算法--Adam**
- 如果 *w b* 持续朝着大致的方向运动，增加学习率
- 如果 *w b* 持续振荡，减小学习率
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408090123913.png)

#### 其他的网络层类型
- **密集层**：后一层隐藏层是前一层的每个激活值的函数
- **卷积层**：每个神经元只查看前一层的部分输入  

优点：加速计算，需要更少的训练数据，减少过拟合风险
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408090214075.png)

#### 什么是导数
在本节中，导数可以表示为如下式子：如果 w 上升了 Epsilon，那么 w 的 J 上升了多少常数 k 乘以 Epsilon，这个常数 k 就是导数，k 取决于函数 J 是什么，以及 w 的值是多少
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408120938217.png)
**导数**：一个函数在某一点上的瞬时变化率。几何上，导数代表了函数图形在该点上的切线斜率。
**计算导数的库**: sympy
```python
import sympy
J, w = sympy.symbols('J, w') # 注明符号
J = w ** 3
print(J)
dJ_dw = sympy.diff(J, w)
print(dJ_dw) # 计算导数
print(dJ_dw.subs([(w, 2)])) # 将 w=2 带入该表达式求值
```

#### 计算图
**计算图**：用于表示数学计算或程序执行的图结构。它在深度学习和神经网络的实现中尤为重要，展示了如何计算神经网络输出a的前向传播和反向传播的步骤
反向传播计算导数值思想：链式求导法则
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408120958657.png)
反向传播是有效计算导数值的方法，因为如果计算了 J 对 a 的导数值一次，就可以一起计算 J 对 w 和 J 对 b 的导数值，可以大大减小计算步骤
- 节点Node: 图中黄色框图
- 参数Parameter: 图中蓝色参数

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408121005830.png)

#### 大型神经网络案例
Tensorflow、PyTorch框架的**优势**：自动求导/自动微分机制
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408121556394.png)

### Week3 应用机器学习的建议
#### 决定下一步做什么
调试学习算法的方法：
- 获取更多训练样本
- 尝试更少的特征
- 尝试额外的特征
- 尝试多项式特征
- 尝试减小 λ
- 尝试增加 λ

通过使用不同的诊断方法，可以指导自己如何提高算法的性能，选择正确的调试方法
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408121559463.png)

#### 模型评估
如果输入特征过多，无法绘制 f 的函数图像，则需要别的方法来评估模型性能
**训练集**和**测试集**的概念:
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408121622361.png)
J test 和 J train 不是使用逻辑损失来计算测试误差，而是使用训练误差来衡量测试集的分数和算法错误分类的分数(区别于成本函数 Cost)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408121627361.png)
> 不包含正则化项可以更好地了解学习算法表现如何

#### 模型选择&交叉验证测试集的训练方法
- ***第一种方式***：把数据集全部作为训练集，然后用训练集训练模型，用训练集验证模型（如果有多个模型需要进行选择，那么最后选出训练误差最小的那个模型作为最好的模型）
**方法缺陷**：训练误差(J train)不是一个很好的评估模型性能的指标，不能表示它在新例子的泛化能力如何，因为训练误差可以无限接近0(可能为过拟合情况)，远低于实际的泛化误差，因此使用测试误差(J test)会更好。  
- ***第二种方式***：把数据集随机分为训练集和测试集，然后用训练集训练模型，用测试集验证模型（如果有多个模型需要进行选择，那么最后选出测试误差最小的那个模型作为最好的模型）
**方法缺陷**(三种思路)：
  1. 训练集用来训练参数 w 和 b ，但不能评估 w 和 b 的好坏，如果测试集用来选择多项式模型(即参数 d)，同理测试集自己也不能用来评估参数 d 的好坏。
  2. 因为选择参数 d 的过程是依赖于测试集数据的，这个 d 值可能恰好只是对于测试集来说的最优，但如果再用这个模型在测试集上评估性能，就不准确了，即对模型泛化能力的乐观估计。
  3. 模型评估意义在于了解模型对于新数据的泛化能力，此时模型已经确定，但选择模型的过程同样还在训练模型的过程中，如果使用测试误差最小的模型，那么此时模型评估的就不是新数据的泛化能力，而是在训练模型已经使用过的测试集(旧数据)，因此此方法有缺陷。
**总结**：J test 可以评估模型好坏，但不是模型选择的标准。如果想要在多个模型中选择效果最好的模型(超参数)，应该引入交叉验证集的概念。
- ***第三种方式***：把数据集随机分为训练集，验证集和测试集，然后用训练集训练模型，用验证集验证模型，根据情况不断调整模型，选择出其中最好的模型，再用训练集和验证集数据训练出一个最终的模型，最后用测试集评估最终的模型
**交叉验证集(验证集/开发集)**：用来检查或信任检查不同模型的有效性或准确性，寻找超参数
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408121755811.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408121856124.png)
**模型训练与选择方法**：先训练每一个模型的最优参数 -> 交叉验证寻找最好的模型 -> 测试误差进行模型评估  
**方法优点**：这种方法可以确保模型性能评估更公平，而不是对模型的泛化能力的乐观估计
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408121756025.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408121756241.png)

#### 通过偏差与方法进行诊断
- 高偏差(High bias -> underfit)：J train 值很高, J cv 值很高
- 高方差(High variance -> overfit)：J train 值很低, J cv 值很高，J cv 远高于J train，模型在看到的数据上比在没看到的数据上做的更好
- 刚好拟合(Just right)：J train 值很低, J cv 值很低

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408130941800.png)
偏差和方差对应着模型的**拟合能力**和**泛化能力**，在神经网络中，当然有的模型拟合能力和泛化能力都很差，既高偏差又高方差
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408130942515.png)

#### 正则化、偏差、方差
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408130952970.png)
利用交叉误差来为正则化参数 λ 选择一个合适的值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408130953418.png)
选择不同的 λ 值对应着不同的 J train 和 J cv 情况
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408130953491.png)

#### 制定一个用于性能评估的基准
对于语音识别问题，即便训练误差看起来很大，但与人类表现水平相比较，差距很小，即这个模型在训练集上的表现跟人类去识别语音的能力相差无几；训练误差与交叉验证误差相比，差距较大，因此这个模型更可能有高方差问题而非高偏差问题
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408131003471.png)
你希望模型可以达到什么样的误差/期望性能水平？(baseline)
- 人类表现水平
- 竞品算法表现
- 根据经验猜测

评判模型性能的参数
|条件|训练误差与基准表现的差值较大|训练误差与基准表现的差值较小|
|:---:|:---:|:---:|
|**训练误差与交叉验证误差的差值大**|高偏差|高方差|
|**训练误差与交叉验证误差的差值小**|几乎不可能出现|恰好拟合|

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408131007435.png)

#### 学习曲线
**学习曲线**：帮助理解学习算法如何作为它拥有的经验量的函数的方法，经验指的是如它拥有的训练示例的数量
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408131518157.png)
- 训练集越大，模型越难完美拟合所有训练示例
- 交叉验证误差通常高于训练误差，因为学习算法将参数拟合到训练集

对于具有**高偏差**的学习算法：
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408131519531.png)
- 即使得到越来越多的训练样本，模型也不会有太多变化(线性回归完成不了非线性任务)
- 基准性能水平与 J train 和 J cv 均有较大差距

对于具有**高方差**的学习算法：
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408131520296.png)
- J cv 远大于 J train ，在训练集上的表现比在交叉验证集上好得多
- J train < 基准性能水平 < J cv
- 获取更多训练样本，训练误差将上升，交叉验证误差会下降

#### 决定下一步做什么
什么情况下使用什么样的调试学习算法的方法：
1. 高方差 (过拟合)
   - 获取更多训练样本
   - 尝试更少的特征
   - 尝试增加 λ
2. 高偏差 (欠拟合)
   - 尝试额外的特征
   - 尝试多项式特征
   - 尝试减小 λ

#### 方差与偏差
为了获得更好的模型，需要的操作：**权衡偏差和方差**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408131633170.png)
调试流程：
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408131629934.png)
神经网络正则化：
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408131631461.png)
小结：
- 具有良好的正则化的大型神经网络通常与较小的神经网络一样好或更好，只要正则化是正确的，拥有一个更大的神经网络几乎没有坏处。
- 扩大神经网络结构可能带来的负面影响：减慢训练和推理过程
- 只要训练集不是很大，那么一个新的网络，尤其是大型神经网络往往是一个低偏差机器

#### 机器学习开发的迭代
**机器学习模型建立循环**：选择结构 -> 训练模型 -> 诊断模型 -> 选择结构
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408150930339.png)
垃圾邮件分类器的建立
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408150935231.png)
改进措施：
- 收集更多数据
- 建立更复杂的特征(邮件路由 经过地址)
- 根据电子邮件正文定义更复杂的特征(不同单词不同意思)
- 设计算法去发现错误拼写
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408150935165.png)

#### 错误分析
诊断方法：
- 偏差、方差分析
- 错误分析

抽取错误样本中的一个子集查看错误的种类，根据错误示例种类出现的次数决定改善模型的优先级(去大费周章地改进一些出现频率较小的错误收效很低)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408151002407.png)

#### 添加更多数据
**数据增强**：修改一个已存在的训练样本去生成一个新的训练样本
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160109707.png)
引入扭曲：字母扭曲，音频噪音
添加完全随机或无意义的噪音进数据通常没有任何帮助
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160110408.png)
**数据合成**：使用人工数据创建全新的训练样本
OCR字符识别任务：利用电脑字体截图生成训练样本
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160114009.png)
**数据工程**：
常规以模型为中心的方法：修改算法和模型结构提升模型质量
以数据为中心的方法：提升数据质量部(在现代算法逐渐完善的情况下，现在数据工程更加注重对数据质量的改善)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160116868.png)

#### 迁移学习--使用其他任务中的数据
**迁移学习思想**：使用其他类似、已训练完成的神经网络任务的除了输出层以外的所有层作为参数的起点，然后运行优化算法
**迁移学习步骤**：
1. **监督预训练**：在大型数据集上训练得到参数
   - 对于很多神经网络，已经有研究人员在大数据集上训练好了参数，可以直接使用下载别人训练好的神经网络，但应使用相同的输入特征类型(图像、音频、文本)
2. **微调**：从已初始化或从监督预训练中获得的参数。进一步梯度下降微调权重，以适应自己的任务
   - 训练小训练集：只训练输出层参数
   - 训练大训练集：训练所有参数(以已经被训练的网络模型的参数做初始化)

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160144422.png)
**迁移学习原理**：神经网络学习了通用的图像特征(边缘，拐角，曲线/基本形状)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160144811.png)

#### 机器学习项目的完整周期
1. 确定项目范围：决定项目是什么
2. 收集数据：确定并收集数据
3. 训练模型：训练，错误分析，迭代改进
4. 在生产中部署：部署、监视并维护系统

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160157678.png)
**部署过程**：
通过***API***和***推理服务器***调用应用程序
软件工程所需要完成的工作：
- 确保预测的可靠和有效
- 管理大量用户的扩展
- 记录用户数据(用户隐私同意)
- 监视系统
- 更新模型

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160158656.png)

#### 公平、偏见与伦理
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160204245.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160205625.png)
准则：在部署前检查系统可能存在的危害
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160205823.png)

#### 倾斜数据集的误差指标
**倾斜数据集**：一个数据集中的正面和负面例子的比例非常不平衡，比如数据集中，结果为1的占比20%，结果为0的占比80%

*e.g* ：如果一组实验组中只有0.5%是患有稀有病因的病人( 1 )，其余结果是正常人( 0 )。一个模型的预测准确度是99.5%，预测了所有数据的结果都是正常，这个模型的准确度很高，但是预测不出稀有病例，这不能代表这个模型是好模型。因此需要引入其他的误差度量方式来评估模型好坏。

**混淆矩阵**
- True Positive  （真正, TP）被模型预测为正的正样本
- True Negative （真负 , TN）被模型预测为负的负样本
- False Positive （假正, FP）被模型预测为正的负样本
- False Negative（假负 , FN）被模型预测为负的正样本

***精确率 (Precision)*** = *TP / ( TP + FP )*  (所有判别为真的样本，找的准不准)
***召回率 (Recall)*** = *TP / ( TP + FN )*  (所有标签为真的样本，找的全不全)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160230510.png)

#### 精确率与召回率的权衡
对于逻辑回归问题：
- 当提高阈值，能提高精确率，但是会降低召回率
   (宁可错过一千，不愿错杀一个)
- 当降低阈值，能提高召回率，但是会降低精确率
  (宁可错杀一千，不愿放过一个)

小结：精确率与召回率成**负相关**关系
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160246830.png)
**如何权衡精确率和召回率**：
- 根据P-R曲线图手动设置阈值，权衡利弊
- 使用 F1 score (也称为调和平均数)，是一种取平均值的方法，结合精确率和召回率，若精确率和召回率某一方过低会执行惩罚，最终计算结果越大说明模型质量更高。

***F1 score***（P为精确率，R为召回率）：
***F1 score*** = *1 / ( ( 1 / P + 1 / R ) / 2)* = *2 P R / (P + R)*
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408160252287.png)

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