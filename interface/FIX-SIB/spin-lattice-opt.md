# 路线 A：基于 Product Manifold 的自旋-晶格协同优化算法

## 1. 问题定义

我们考虑如下联合最小化问题：

\[
\min_{R,S} E(R,S)
\]

其中：

- \(R \in \mathbb{R}^{3N}\)：原子坐标
- \(S = (\mathbf S_1,\dots,\mathbf S_N)\)：自旋变量
- 每个自旋满足约束
  \[
  \|\mathbf S_i\| = 1
  \]

因此，变量空间天然不是普通欧氏空间，而是一个乘积流形：

\[
\mathcal M = \mathbb R^{3N} \times (S^2)^N
\]

如果进一步考虑晶胞自由度，可以扩展为：

\[
\mathcal M = \mathbb R^{3N} \times (S^2)^N \times \mathcal C
\]

---

## 2. 这条路线的核心思想

路线 A 的出发点是：

> 将自旋-晶格协同优化写成定义在乘积流形  
> \(\mathbb R^{3N} \times (S^2)^N\) 上的联合优化问题，  
> 再在这个空间上构造预条件 metric，并结合 Riemannian L-BFGS 进行加速。

这和常见方法的区别在于：

- 不把自旋伪装成普通欧氏变量
- 不采用简单的“原子一步、自旋一步”的交替最小化
- 显式利用原子自由度与自旋自由度的不同几何结构
- 通过联合预条件器处理二者尺度不匹配和耦合问题

---

## 3. 为什么这条路线有意义

### 3.1 现有方法的局限

传统做法通常是：

1. 固定自旋，优化原子
2. 固定原子，优化自旋
3. 两步交替进行直到收敛

这类方法的问题包括：

- 原子与自旋强耦合时容易震荡
- 两类自由度的收敛速度通常很不匹配
- 无法有效利用 mixed Hessian 信息

### 3.2 路线 A 的优势

将问题写成乘积流形优化后，可以自然做到：

- 自旋长度约束天然保持
- 原子与自旋使用不同的预条件器
- 更适合引入 block quasi-Newton 结构
- 更容易推广到带晶胞、应力或其他约束的情形

---

## 4. 数学结构

## 4.1 联合变量

定义联合变量：

\[
x=(R,S)\in\mathcal M=\mathbb R^{3N}\times(S^2)^N
\]

其中：

- \(R=(\mathbf r_1,\dots,\mathbf r_N)\)
- \(S=(\mathbf S_1,\dots,\mathbf S_N)\)

---

## 4.2 切空间

对于原子部分：

\[
T_R \mathbb R^{3N}=\mathbb R^{3N}
\]

对于单个自旋 \(\mathbf S_i\in S^2\)，其切空间为：

\[
T_{\mathbf S_i}S^2
=
\{\boldsymbol\xi_i\in\mathbb R^3 \mid \mathbf S_i\cdot \boldsymbol\xi_i=0\}
\]

因此，联合切空间为：

\[
T_x\mathcal M
=
\mathbb R^{3N}\times\prod_{i=1}^N T_{\mathbf S_i}S^2
\]

这意味着自旋的搜索方向必须与当前自旋方向正交。

---

## 5. 联合梯度

设总能量为 \(E(R,S)\)。

### 5.1 原子部分梯度

\[
g_R=\nabla_R E
\]

这对应于普通几何优化中的力。

### 5.2 自旋部分梯度

先定义欧氏导数：

\[
\widetilde g_{S_i}=\frac{\partial E}{\partial \mathbf S_i}
\]

再投影到切空间：

\[
g_{S_i}
=
\widetilde g_{S_i}
-
(\widetilde g_{S_i}\cdot \mathbf S_i)\mathbf S_i
\]

于是联合 Riemannian 梯度可以写成：

\[
\mathrm{grad}\,E(x)
=
(g_R,g_S)
\]

其中 \(g_S=(g_{S_1},\dots,g_{S_N})\)。

---

## 6. 路线 A 的核心：联合预条件 metric

路线 A 最关键的创新点在于，不直接使用裸梯度，而是在乘积流形上定义一个联合 metric，从而实现预条件优化。

### 6.1 最基本的块结构

可以定义一个块形式的预条件算子：

\[
M_x
=
\begin{pmatrix}
P_R & C_{RS}\\
C_{SR} & P_S
\end{pmatrix}
\]

其中：

- \(P_R\)：原子部分预条件块
- \(P_S\)：自旋部分预条件块
- \(C_{RS}, C_{SR}\)：原子-自旋耦合块

---

### 6.2 第一阶段推荐：先做块对角形式

最稳妥的第一版建议是先从：

\[
M_x
=
\begin{pmatrix}
P_R & 0\\
0 & \lambda P_S
\end{pmatrix}
\]

开始。

这里：

- \(P_R\) 用来处理原子部分的刚度差异
- \(P_S\) 用来处理自旋部分的刚度差异
- \(\lambda\) 用来平衡原子梯度和自旋梯度的尺度

这一步已经足以形成一个完整的方法原型。

---

## 7. 原子部分预条件器 \(P_R\)

原子部分可以借鉴几何优化中的稀疏预条件思想，构造类似图 Laplacian 的近似 Hessian：

\[
P_R = L_{\text{bond}} + \mu I
\]

其中：

- \(L_{\text{bond}}\)：基于邻接关系构造的加权 Laplacian
- \(\mu I\)：正则项

它的物理含义是：

- 局域短键对应较大刚度
- 长波模式对应较小刚度

这种结构在材料几何优化中已经被证明非常有效。

---

## 8. 自旋部分预条件器 \(P_S\)

自旋部分是路线 A 最有可能做出新意的地方。

如果自旋能量主要由交换作用主导：

\[
E_{\text{spin}} \sim -\sum_{ij} J_{ij}\mathbf S_i\cdot\mathbf S_j
\]

则可以构造一个交换图 Laplacian 型预条件器：

\[
P_S = L_J + \nu I
\]

其中：

- \(L_J\)：由交换常数 \(J_{ij}\) 构造的加权图 Laplacian
- \(\nu I\)：正则项

---

### 8.1 三个层次的实现方式

#### 方案 A：最简单版本

\[
P_S = I
\]

即只用一个全局平衡因子 \(\lambda\)。

#### 方案 B：局域对角近似

\[
P_S=\mathrm{diag}(\alpha_1,\dots,\alpha_N)
\]

每个自旋一个局域尺度。

#### 方案 C：交换图预条件

\[
P_S=L_J+\nu I
\]

这是最有“新算法”味道的版本。

---

## 9. 平衡因子 \(\lambda\)

原子梯度和自旋梯度通常量纲不同、数值尺度也差很多，因此需要动态平衡参数。

一个简单做法是：

\[
\lambda_k=\frac{\|g_R\|}{\|g_S\|+\varepsilon}
\]

更稳妥一些，可以定义为：

\[
\lambda_k
=
\sqrt{
\frac{\langle g_R, P_R^{-1}g_R\rangle}
{\langle g_S, P_S^{-1}g_S\rangle+\varepsilon}
}
\]

它的作用是使原子与自旋两个子空间在联合搜索方向上具有可比权重。

---

## 10. 搜索方向：Preconditioned Riemannian L-BFGS

路线 A 的推荐主算法不是简单梯度下降，而是：

- 先做预条件
- 再做 Riemannian L-BFGS

### 10.1 基本形式

给定联合梯度 \(g_k=(g_{R,k},g_{S,k})\)，先做预条件：

\[
z_k=M_k^{-1}g_k
\]

然后利用 L-BFGS 近似逆 Hessian：

\[
p_k=-H_k z_k
\]

其中：

- \(M_k^{-1}\) 负责处理尺度和刚度不匹配
- \(H_k\) 负责利用历史曲率信息加速收敛

---

## 11. 更新方式

## 11.1 原子更新

原子部分可以普通更新：

\[
R_{k+1}=R_k+\alpha_k p_{R,k}
\]

---

## 11.2 自旋更新

自旋不能直接加法更新，否则会破坏长度约束。

最简单的 retraction 写法为：

\[
\mathbf S_{i,k+1}
=
\frac{
\mathbf S_{i,k}+\alpha_k p_{S_i,k}
}{
\|\mathbf S_{i,k}+\alpha_k p_{S_i,k}\|
}
\]

因为 \(p_{S_i,k}\) 已经在切空间内，所以这种归一化更新是合理的。

更几何化的版本也可以使用 Rodrigues 旋转形式，但第一版没必要上来就做得太复杂。

---

## 12. 历史向量与 Riemannian L-BFGS

如果要严格做 Riemannian L-BFGS，需要考虑：

- retraction
- vector transport
- 切空间之间的历史向量转移

标准历史向量可写为：

\[
s_k=\mathcal T_{x_k\to x_{k+1}}(\alpha_k p_k)
\]

\[
y_k
=
\mathrm{grad}E(x_{k+1})
-
\mathcal T_{x_k\to x_{k+1}}(\mathrm{grad}E(x_k))
\]

其中 \(\mathcal T\) 表示 vector transport。

如果第一版想简化实现，也可以先采用近似版本，再逐步升级。

---

## 13. 线搜索

联合更新后的 trial point 为：

\[
(R_{\text{trial}},S_{\text{trial}})
=
(R_k+\alpha p_R,\ \mathrm{Retr}_{S_k}(\alpha p_S))
\]

然后选择步长 \(\alpha\) 使得能量满足 Armijo 条件或 Wolfe 条件。

第一版建议直接用 **Armijo backtracking**：

1. 初始 \(\alpha=1\)
2. 若下降不足，则 \(\alpha\leftarrow \beta\alpha\)
3. 直到满足充分下降条件

这样最简单，也最稳。

---

## 14. 算法伪代码

```text
Input:
    initial (R0, S0)
    memory size m
    preconditioners PR, PS
    tolerances εR, εS

for k = 0,1,2,...
    1. Compute energy E(Rk, Sk)

    2. Compute Euclidean gradients:
           gR = ∂E/∂R
           gS_tilde = ∂E/∂S

    3. Project spin gradients onto tangent space:
           gS_i = gS_tilde_i - (gS_tilde_i · S_i) S_i

    4. Form joint gradient:
           gk = (gR, gS)

    5. Build/update preconditioner:
           Mk = diag(PR, λk PS)

    6. Precondition:
           zk = Mk^{-1} gk

    7. Use Riemannian L-BFGS to generate direction:
           pk = - Hk zk

    8. Perform line search over α:
           Rtrial = Rk + α pR
           Strial = Retr_Sk(α pS)

    9. Accept update:
           Rk+1 = Rtrial
           Sk+1 = Strial

   10. Update LBFGS history

   11. Check convergence:
           ||gR|| < εR and ||gS|| < εS
end for
