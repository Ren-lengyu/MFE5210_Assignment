# MFE5200_Assignment BYD因子挖掘与回测
## 1. 项目背景

本项目基于比亚迪（BYD，股票代码：002594.SZ）股票数据，计算得到五个量化因子（`Turnover_Volatility_20D`, `Alpha6`, `VPR`, `HLM`, `DVM`）并进行回测，因子来自于学术论文、研究报告、网络公开资料与市场直觉。

我们需要：
- 确保因子间低相关性（最大相关系数 ≤ 0.5）。
- 评估因子对未来回报的预测能力（信息系数，IC）。
- 分析因子驱动策略的收益与风险（夏普比率、最大回撤等）。

**数据来源**：
- **Tushare**：提供比亚迪日线数据（2022年4月30日至2025年4月30日），包括开盘价、收盘价、成交量、成交额等。
- **CSMAR 数据库**：补充部分因子计算所需的其他股票数据或财务指标。

## 2. 数据与因子描述

### 2.1 数据集
- **原始数据**：通过 Tushare 与 CSMAR 获取比亚迪日线数据，保存为 `BYD.csv`，包含字段：
  - `Clsprc`：收盘价。
  - `Opnprc`：开盘价。
  - `Volume`：成交量。
  - `Amount`：成交额。
  - `ToverOs`：换手率。
- **因子数据**：基于 `BYD.csv` 计算因子，保存为 `BYD_factors.csv`，包含五个因子值。
- **时间范围**：2022年4月30日至2025年4月30日，三年数据。
- **数据量**：清洗后 707 个交易日，支持年度化指标计算。

### 2.2 因子定义与数学表达式
以下为五个因子的定义及其数学表达式：
1. **Turnover_Volatility_20D**（20天换手率波动率）：
   - **定义**：衡量换手率在20天窗口的波动性，反映市场交易活跃度的变化。
   - **数学表达式**：
     \[
     \text{Turnover_Volatility}_{20D,t} = \sqrt{\frac{1}{20} \sum_{i=t-19}^{t} (\text{ToverOs}_i - \bar{\text{ToverOs}})^2}
     \]
     其中，\(\text{ToverOs}_i\) 为第 \(i\) 天的换手率，\(\bar{\text{ToverOs}}\) 为20天换手率均值。

2. **Alpha6**（开盘价与成交量负相关性）：
   - **定义**：计算10天窗口内开盘价与成交量的负相关系数，捕捉价格-成交量背离。
   - **数学表达式**：
     \[
     \text{Alpha6}_t = - \text{Corr}(\text{Opnprc}_{t-9:t}, \text{Volume}_{t-9:t})
     \]
     其中，\(\text{Corr}\) 为 Pearson 相关系数，\(\text{Opnprc}\) 和 \(\text{Volume}\) 分别为开盘价和成交量。

3. **VPR**（成交量-价格共振）：
   - **定义**：当5天成交量斜率和价格动量同为正时，赋值为1，否则为0，表示成交量与价格同步上涨。
   - **数学表达式**：
     \[
     \text{VPR}_t = 
     \begin{cases} 
     1, & \text{if } \text{Slope}(\text{Volume}_{t-4:t}) > 0 \text{ and } \text{Momentum}_{5D,t} > 0 \\
     0, & \text{otherwise}
     \end{cases}
     \]
     其中，\(\text{Slope}(\text{Volume})\) 为成交量线性回归斜率，\(\text{Momentum}_{5D,t} = \frac{\text{Clsprc}_t}{\text{Clsprc}_{t-5}} - 1\)。

4. **HLM**（异质流动性动量）：
   - **定义**：结合20天价格动量与上下行成交量比率，反映流动性驱动的动量效应。
   - **数学表达式**：
     \[
     \text{HLM}_t = \text{sign}(\text{Momentum}_{20D,t}) \cdot \frac{\text{Up_Volume_Mean}_{20D,t}}{\text{Down_Volume_Mean}_{20D,t}}
     \]
     其中，\(\text{Momentum}_{20D,t} = \frac{\text{Clsprc}_t}{\text{Clsprc}_{t-20}} - 1\)，\(\text{Up_Volume_Mean}_{20D,t}\) 和 \(\text{Down_Volume_Mean}_{20D,t}\) 分别为20天上涨和下跌成交量均值。

5. **DVM**（方向性成交量动量）：
   - **定义**：衡量20天窗口内价格方向与成交量变化方向的一致性比例。
   - **数学表达式**：
     \[
     \text{DVM}_t = \frac{1}{20} \sum_{i=t-19}^{t} \mathbb{1}(\text{Price_Direction}_i \cdot \text{Volume_Change_Dir}_i > 0)
     \]
     其中，\(\text{Price_Direction}_i = \text{sign}(\text{Clsprc}_i - \text{Clsprc}_{i-1})\)，\(\text{Volume_Change_Dir}_i = \text{sign}(\text{Volume}_i - \text{Volume}_{i-1})\)，\(\mathbb{1}\) 为指示函数。

### 2.3 回测前数据预处理
- **每日回报**：计算 \(\text{Daily_Return}_t = \frac{\text{Clsprc}_t}{\text{Clsprc}_{t-1}} - 1\)。
- **缺失值处理**：
  - 因子列缺失值：`Turnover_Volatility_20D`（19个）、`HLM`（20个）、`DVM`（19个）、`Alpha6` 和 `VPR`（0个）。
  - `Daily_Return` 缺失值：1个（首行）。
  - 移除包含 NaN 的行，清洗后数据量为 707 个交易日。

## 3. 回测方法

### 3.1 回测思路
回测采用单资产（BYD）因子驱动纯多（long only）策略，步骤如下：
1. **因子信号生成**：
   - 对于每个因子，若 \(\text{Factor}_t > 0\)，生成多头信号（\(\text{Position}_t = 1\)）；否则不持有（\(\text{Position}_t = 0\)）。
   - 信号延迟一天（\(\text{Position.shift(1)}\)），模拟基于前一天因子值的交易决策。
2. **投资组合构建**：
   - 组合回报：\(\text{Portfolio_Return}_t = \text{Position}_{t-1} \cdot \text{Daily_Return}_t\)。
   - 假设全额投资，**无交易成本**。
3. **绩效指标计算**：
   - **IC**：
   - 因子值与下一交易日回报的 Spearman 相关系数，IC 越高（接近 1 或 -1），因子对回报的预 
     测能力越强；接近 0 则表示预测能力弱。
   - 计算方法：\(\text{Corr}_{\text{Spearman}}(\text{Factor}_t, \text{Daily_Return}_{t+1})\)。
   - **夏普比率**：
   - 评估因子策略的风险调整后收益，夏普比率越高，策略在单位风险下的收益越优。
   - 计算方法：\(\frac{\text{mean}(\text{Portfolio_Return})}{\text{std}(\text{Portfolio_Return})} \cdot \sqrt{252}\)。
   - **收益风险比**：
   - 因子策略平均日收益率与日收益率标准差的比值，反映每单位风险带来的收益。数值越高，策略 
     效率越高。
   - 计算方法：\(\frac{\text{mean}(\text{Portfolio_Return})}{\text{std}(\text{Portfolio_Return})}\)。
   - **最大回撤**：
   - 衡量因子策略在回测期间累计回报的最大损失比例，反映策略的最大风险。数值越低，策略抗风 
     险能力越强。
   - 计算方法：\(\max\left(\frac{\text{rolling_max}_t - \text{cumulative_return}_t}{\text{rolling_max}_t}\right)\)，其中 \(\text{cumulative_return}_t = \prod_{i=1}^t (1 + \text{Portfolio_Return}_i)\)。
   - **年化收益率**：
   - 将因子策略的平均日收益率换算为年化收益率，反映策略的长期收益能力。数值越高，策略盈利 
     能力越强。
   - 计算方法：\((1 + \text{mean}(\text{Portfolio_Return}))^{252} - 1\)。
   - **年化波动率**：
   - 因子策略日收益率标准差的年化值，衡量回报的波动性。数值越低，策略稳定性越高。
   - 计算方法：\(\text{std}(\text{Portfolio_Return}) \cdot \sqrt{252}\)。
4. **因子间相关性分析**：
   - 计算因子间 Pearson 相关系数矩阵。
   - 绘制热力图，检查最大相关系数是否 ≤ 0.5。

### 3.2 关键假设
- **单资产**：仅回测 BYD（002594.SZ）。
- **无交易成本**：忽略交易费用、滑点。
- **年度化**：一年 252 个交易日。
- **信号逻辑**：因子值正负驱动多头信号，未考虑做空。

## 4. 因子绩效分析

以下为五个因子的回测绩效：

| 因子                    | IC       | 夏普比率 | 收益风险比 | 最大回撤 | 年化收益率 | 年化波动率 |
|-------------------------|----------|----------|------------|----------|------------|------------|
| Turnover_Volatility_20D | -0.0178  | 0.3365   | 0.0212     | 0.5276   | 0.1233     | 0.3455     |
| Alpha6                  | 0.0299   | 0.1677   | 0.0106     | 0.4323   | 0.0391     | 0.2287     |
| VPR                     | 0.0291   | 0.7449   | 0.0469     | 0.2683   | 0.1437     | 0.1803     |
| HLM                     | -0.0200  | 0.2650   | 0.0167     | 0.3521   | 0.0673     | 0.2459     |
| DVM                     | -0.0152  | 0.3365   | 0.0212     | 0.5276   | 0.1233     | 0.3455     |

## 5. 因子相关性分析

### 5.1 相关系数矩阵
因子间的 Pearson 相关系数矩阵如下（实际结果）：

|                         | Turnover_Volatility_20D | Alpha6 | VPR    | HLM    | DVM    |
|-------------------------|-------------------------|--------|--------|--------|--------|
| Turnover_Volatility_20D | 1.0000                  | -0.0668 | 0.0071 | 0.3343 | 0.0326 |
| Alpha6                  | -0.0668                 | 1.0000 | -0.1699 | -0.1015 | -0.1388 |
| VPR                     | 0.0071                  | -0.1699 | 1.0000 | 0.1802 | 0.1727 |
| HLM                     | 0.3343                  | -0.1015 | 0.1802 | 1.0000 | 0.2670 |
| DVM                     | 0.0326                  | -0.1388 | 0.1727 | 0.2670 | 1.0000 |


### 5.2 热力图
相关系数矩阵通过热力图可视化（见图 1），红色表示正相关，蓝色表示负相关，白色表示低相关。

**图 1：因子相关系数热力图**  
![factors_correlation_heatmap](https://github.com/user-attachments/assets/40aa0daf-cffe-423f-aaec-6691c5e0544a)


### 5.3 分析
- 因子间相关性较低，最大值为 0.3343，表明因子具有较好独立性。
- 低相关性支持多因子模型构建，减少冗余风险。

## 6. 结论

本次回测分析了比亚迪股票的五个量化因子，得出以下结论：
- `VPR` 表现出较强的预测能力和投资绩效（夏普比率 0.7449，年化收益率 14.37%）。
- **因子间最大相关系数为 0.3343**，满足低相关性要求，支持多因子策略。
- **五因子平均夏普比率为 0.3701**，整体表现中等。

## 7. 参考文献
-《101 formulaic alphas》  
-《中国A股市场量化因子白皮书》  
-《Size and Value in China》

以上原文可在Reference Branch中查看
