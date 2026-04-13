# Bug 修复总结

## 已修复的确定性 Bug

### 1. 严重：线搜索后重复推进（实际步长翻倍）

**问题**：
- `armijo_line_search()` 内部已经调用 `advance_atoms/advance_spins(alpha)` 并接受步长
- 但在 `iterate()` 主循环中又调用了一次 `advance_atoms/advance_spins(alpha)`
- 导致每次迭代实际走了两步，步长翻倍

**修复**：
- 删除了 `iterate()` 中第 193-194 行的重复调用
- 添加注释说明线搜索已经内部完成了推进

**位置**：min_spin_lattice.cpp:193-194（已删除）

---

### 2. 严重：nlocal 变化时数组越界

**问题**：
- 仅在迭代开始前检查一次 `nlocal_max < nlocal`
- `energy_force()` 可能触发原子迁移，改变 `nlocal`
- 后续的梯度/方向计算可能写越界

**修复**：
- 在每次迭代开始时检查并重新分配数组
- 在 `energy_force()` 调用后再次检查并重新分配
- 确保所有数组容量始终 >= nlocal
- 包括 fm_full 数组（如果使用 PairSpinML）

**位置**：min_spin_lattice.cpp:151-162, 186-198

---

### 3. 高风险：自旋梯度方向错误

**问题**：
- 原实现使用 torque（sp × fm）作为梯度
- 但 torque 是用于 precession 动力学的（dŝ/dt = ŝ × fm）
- 对于能量最小化，需要的是 Riemannian 梯度：-dE/dS 投影到切空间

**修复（严格流形版）**：

对于 **PairSpinML**（如 pair_spin_step）：
```cpp
// 使用 fm_full（已经是 -dE/dm，无任何转换）
pair_spin_ml->distribute_full_mag_forces(fm_full, nlocal);

// 直接投影到切空间
double dot = fm_full[i] · sp[i];
g_spin[i] = fm_full[i] - dot * sp[i];
```

对于 **标准 PairSpin**：
```cpp
// fm = H_eff（有效场）
// 能量梯度：-dE/dS = -mag * H_eff = -mag * fm
double grad = -mag * fm[i];

// 投影到切空间
double dot = grad · sp[i];
g_spin[i] = grad - dot * sp[i];
```

**关键区别**：
- `fm_full`：原始能量梯度 -dE/dm（eV/μ_B），用于最小化
- `fm`：转换后的场（用于动力学），对 pair_spin_step 已乘 mag/hbar

**位置**：min_spin_lattice.cpp:305-360

---

### 4. 中等：ftol 量纲不一致

**问题**：
- `fnorm_max()` 和 `fnorm_inf()` 返回平方值（f²）
- `max_torque()` 和 `inf_torque()` 返回开方后的值（√(τ²) * hbar）
- 直接比较会导致量纲不匹配

**修复**：
```cpp
// 对 fnorm 开方，使量纲一致
if (normstyle == MAX) fnorm = sqrt(fnorm_max());
else if (normstyle == INF) fnorm = sqrt(fnorm_inf());
else if (normstyle == TWO) fnorm = sqrt(fnorm_sqr());

// torque 函数已经开方，直接使用
if (normstyle == MAX) tnorm = max_torque();
else if (normstyle == INF) tnorm = inf_torque();
else tnorm = total_torque();
```

**位置**：min_spin_lattice.cpp:230-243

---

### 5. 中等：缺少 max_eval 停止条件

**问题**：
- 只有 `neval++`，没有检查 `neval >= update->max_eval`
- 可能超过用户设定的最大能量评估次数

**修复**：
```cpp
// 在每次 energy_force() 后检查
neval++;
if (neval >= update->max_eval) return MAXEVAL;
```

**位置**：min_spin_lattice.cpp:183, 210

---

## 实现细节

### PairSpinML 检测

在 `init()` 中自动检测：
```cpp
pair_spin_ml = dynamic_cast<PairSpinML*>(force->pair);
```

如果检测到 PairSpinML：
- 分配 `fm_full` 数组
- 使用 `distribute_full_mag_forces()` 获取真实能量梯度
- 输出信息："using PairSpinML fm_full"

否则：
- 使用标准 `fm` 数组
- 输出信息："using standard PairSpin"

### 数组管理

所有数组在两个位置检查并重新分配：
1. 每次迭代开始时
2. `energy_force()` 调用后（可能触发原子迁移）

包括的数组：
- `g_atom`, `g_spin`（梯度）
- `p_atom`, `p_spin`（搜索方向）
- `fm_full`（仅 PairSpinML）

---

## 测试建议

1. **编译测试**：
   ```bash
   cd build
   make -j8
   ```

2. **运行测试**：
   ```lammps
   min_style spin/lattice
   minimize 1e-8 1e-6 10000 100000
   ```

3. **检查输出**：
   - 确认使用 "PairSpinML fm_full" 模式
   - 能量应该单调下降
   - 检查收敛速度

4. **调试选项**（如需要）：
   ```lammps
   min_modify alpha_init 1.0          # 初始步长
   min_modify c1 1.0e-4               # Armijo 常数
   min_modify backtrack_factor 0.5    # 回溯因子
   ```

---

## 代码统计

- **min_spin_lattice.cpp**: 564 行（+85 行）
- **min_spin_lattice.h**: 86 行（+7 行）
- **新增功能**：PairSpinML 自动检测和 fm_full 支持
- **修复 bug**: 5 个确定性 bug 全部修复

