# STEP 多进程 / 多 GPU 测试版

本版实现了 MPI 区域分解、完整磁力回传，以及限制 GPU 显存峰值的分批求导。
本地已通过 LAMMPS 2 Aug 2023 Update 3 + LibTorch 2.9.1 的 C++ 编译和链接，
以及 CPU 上 1/2/4 MPI rank 的数值回归（含真实 Fe STEP 模型）。
**本机仅有一张物理 GPU，尚未验证多张 GPU 的绑定、显存分配和加速比。**
详细验证范围见 `tests/VALIDATION.md`。

## 安装

将本目录和同级 `FIX-SIB` 目录带到服务器。依赖启用 MPI、SPIN 和本项目
SIB 接口的 LAMMPS，以及与服务器 CUDA 环境匹配的 GPU 版 LibTorch。

如果服务器已有能运行旧版 spin/step 的 LAMMPS，更新其源码中的
`pair_spin_step.cpp`、`pair_spin_step.h`、`step_utils.cpp`、`step_utils.h`。
同时更新 `src/USER-SPIN-STEP/` 中相应文件（如存在），避免再次安装包时覆盖新版。
保留当前可运行的旧二进制，然后按原有配置重新编译 LAMMPS。
**必须让使用 pair_spin_step.h 的文件也重新编译，包括 force.cpp；不要仅替换一个 .o。**
**2026-09-11 修订还修改了 FIX-SIB，必须同时升级**
`fix_nve_spin_sib.cpp/.h` 和 `fix_nh_spin_sib.cpp/.h`，
并重新编译其派生类 `fix_nvt_spin_sib.cpp`、`fix_npt_spin_sib.cpp`。
修复了 setup 时重复累加力，以及在旧邻居表上计算移动后的第二个自旋半步的问题。
服务器仍需安装本目录依赖的 `pair_spin_ml.h`。

本仓库顶层的 `Install.sh` 是另一个组件的 CPU 依赖下载脚本，不用于本次升级。

## 输入文件

```lammps
units       metal
atom_style  spin
atom_modify map array
newton      on

# 在此 read_data ...，设置质量、初始自旋等
pair_style  spin/step device auto batch_size 256
pair_coeff  * * /path/to/model.pt Fe
neighbor    0.5 bin
neigh_modify delay 0 every 1 check yes

# 自旋动力学采用集合调用的 SIB 积分器
fix         integrate all nve/spin/sib lattice moving
# 固定晶格时将 moving 改为 frozen
```

选项：

- `device auto`：默认。按节点内 MPI rank 分配可见 GPU；没有可用 CUDA 时使用 CPU。
  遵守 `CUDA_VISIBLE_DEVICES`，支持调度器为每个 rank 只暴露一张 GPU。
- `device cpu`：用于 MPI 数值验收。
- `device cuda:N`：显式指定当前进程可见的 GPU 编号。所有 rank 看见同样设备时，
  不要统一指定 cuda:0，否则会集中使用同一张卡。
- `batch_size N`：每次求导的本地原子能量中心数。默认 0 表示一次处理全部本地中心。
  256 是试跑起点，并非显存保证；显存不足时逐次减小。
  每批还需要中心的多层邻居，故实际 GPU 节点数通常超过 N。
- `halo_layers L`：默认读取模型 `config.json` 的 `num_layers`。
  旧模型缺少该字段时必须明确提供真实消息传递层数，不能为了节约显存缩小它。
  `config.json` 中必须包含与模型一致的正数 `r_max`，不再默认为 5 Å。

代码针对当前 STEP/MagNequIP 导出的局域原子能量模型：每层传播一跳，输出每个
输入节点的能量。全局注意力、跨原子归一化等非局域模型不满足这个分解假设。

MPI 模式不支持原版 `fix nve/spin` 的逐原子自旋扫描；请使用
`nve/spin/sib`、`nvt/spin/sib` 或 `npt/spin/sib`。测试脚本覆盖 nve/spin/sib，
含纵向 glangevin/spin/sib 路径，并检查 NVT/NPT 的初始化和两个自旋半步。
NPT 长时间变胞轨迹仍需在服务器验收。不支持 rRESPA。
保持默认 `comm_modify mode single`，不支持 `neigh_modify include`。
全局 virial 可计算；本次没有实现逐原子 virial，勿将 stress/atom 用作已验证输出。

## 单节点四卡启动示例

在已分配到四张 GPU 的节点上，一张卡一个 MPI rank：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mpirun -np 4 /path/to/lmp_mpi -in in.step
```

在独占节点上如需自行指定四张可见卡，可在启动前设置
`CUDA_VISIBLE_DEVICES=0,1,2,3`；调度器已经设置设备可见性时保留其设置。
集群使用 Slurm 等调度器时，按管理员提供的 MPI 启动方式申请和绑定资源。

模型权重在每个进程复制一份；GPU 激活和求导图只包含该进程当前批次的必要邻域。
通信 halo 为 `num_layers * r_max + skin`，模型邻居边仍按 `r_max` 截断。
因此加卡不会让显存严格按卡数等分，小子域或较深模型的 halo 开销可能很大。
分批可以降低激活显存，但不会降低 LAMMPS 的 CPU ghost/邻居表内存；也会增加重算。
每步临时计算图在每批结束时释放，不缓存上一帧 GPU 梯度。

## 服务器验收

Python 环境需有 numpy 和 torch；MPI 启动器需与 LAMMPS 所链接的 MPI 匹配。
在申请到至少四个 CPU task 的资源内，从项目根目录执行：

```bash
python interface/USER-SPIN-STEP/tests/test_mpi.py \
  --lammps /path/to/lmp_mpi --device cpu --workdir /path/to/test-results-cpu
```

脚本创建一个非线性三层模型，以独立完整周期图的 PyTorch 能量和导数为参考，比较
1/2/4 MPI rank、无分批/小批次、正交/倾斜周期盒、空 rank、孤立原子，以及短程 SIB
横向/纵向轨迹。失败时保留输入、日志和 dump，并以非零状态退出。
另有 SIB 初始化重复加力、跨进程迁移和新邻居出现的回归测试。
本机 CPU 回归已经通过；服务器应重新运行以验收自己的构建和模型。

随后在分配到四张 GPU 的环境测试：

```bash
python interface/USER-SPIN-STEP/tests/test_mpi.py \
  --lammps /path/to/lmp_mpi --device auto --workdir /path/to/test-results-gpu
```

`auto` 可能回退到 CPU，请结合日志和 nvidia-smi 确认 GPU 实际使用情况。
对于单元素 Fe 模型，可追加 `--model /path/to/Fe.pt` 验证真实导出模型。
其他元素模型需用自己的小体系，按原子 ID 对齐比较单/多 rank 的能量、原子力、磁力和
全局压力。FP32 与不同求和顺序会有小误差，要求数值接近，不要求逐位相同。

先比较固定构型 `run 0`，再比较无随机热浴的短轨迹，最后逐步增加体系规模。
非零温度随机热浴在不同 rank 数下通常使用不同随机序列，不应要求逐原子轨迹一致。
反馈问题时请保留模型元数据、输入文件、完整 stdout/log.lammps、rank 数和每卡显存。
