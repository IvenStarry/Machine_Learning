# 机器学习 Machine Learning
**Github**：https://github.com/IvenStarry  
**学习视频网站**：B站吴恩达机器学习2022版
https://www.bilibili.com/video/BV1Pa411X76s?p=4&vd_source=6fd71d34326a08965fdb07842b0124a7
## 有监督的机器学习：回归与分类 Supervised Machine Learning Regression and Classification

### Week3 分类
#### 动机与目的
简单利用线性回归设定决策分界线来完成二元分类问题是有问题的，尤其是有偏差较大的样本存在，影响线性回归方程的建立，推动决策分界线向右移动，从而大幅减小预测的准确率
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