import sys
import os
import glob
import math
import argparse
import warnings
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch_scatter import scatter_add
from ase.io import read
from ase import Atoms
from ase.neighborlist import build_neighbor_list  # [修改] 改用与预处理一致的类
from sympy.physics.wigner import clebsch_gordan
from itertools import product
from collections import defaultdict
from typing import List, Dict, Tuple

# =============================================================================
#  全局配置 (CONFIG)
#  [请在此处修改路径和参数]
# =============================================================================
CONFIG = {
    # --- [新增] 路径配置 ---
    "paths": {
        # 训练好的模型路径（支持.pth和.pt两种格式）
        # .pth: PyTorch state_dict格式
        # .pt/_traced.pt: TorchScript traced格式
        "model_path": "./best_model_traced.pt",
        
        # 存放 .xyz 文件的文件夹路径 (例如: "/home/user/test_data")
        # 脚本会自动读取该文件夹下所有的 .xyz 文件
        "xyz_folder": "./",
    },

    # --- 描述符参数 (必须与训练时一致) ---
    "descriptor": {
        'elements': ['Cr', 'I'],     # 元素列表
        'rc': 4.1,                   # 截断半径
        'n_max': 5,                  # 径向基函数阶数
        'l_max': 3,                  # 角向基函数阶数
        'nu_max': 2,                 # 相关阶数
        'm_cut': 3.5,                # 磁矩截断值
        'use_spin_invariants': True, # 是否使用自旋不变量
        'pos_scale': 200.0,          # 位置缩放因子
        'spin_scale': 1.0,           # 自旋缩放因子
        'epsilon': 1e-7,             # 数值稳定性常数
        'mag_noise_threshold': 0.35, # 磁矩噪声阈值
        'pos_noise_threshold': 1e-8, # 位置噪声阈值
        'pole_threshold': 1e-6,      # 极点判定阈值
    },
    
    # --- 模型参数 (必须与训练时一致) ---
    "model": {
        "hidden_dim": 512,
        "dropout_rate": 0.0,
    },
    
    # --- 运行设备 ---
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

DEVICE = torch.device(CONFIG["device"])

# =============================================================================
#  PART 1: 模型定义 (MagneticNEP)
# =============================================================================
class MagneticNEP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout_rate: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            weight_norm(nn.Linear(input_dim, hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(hidden_dim, hidden_dim)), 
            nn.GELU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(hidden_dim, 1))  
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
#  PART 2: 描述符定义 (MagneticACEDescriptor & Helpers)
# =============================================================================

def _torch_sph_harm_custom(l_max, theta, phi, is_zero_vec):
    Y_lm = {}
    x = torch.cos(theta)
    P_lm = {}
    P_lm[0, 0] = torch.ones_like(x)
    sin_theta = torch.sin(theta)

    for l in range(1, l_max + 1):
        P_lm[l, l] = - (2*l - 1) * sin_theta * P_lm[l-1, l-1]
        if l > 0:
            P_lm[l, l-1] = (2*l - 1) * x * P_lm[l-1, l-1]
        if l > 1:
            for m in range(l - 2, -1, -1):
                denominator = l - m
                P_lm[l, m] = ((2*l - 1) * x * P_lm[l-1, m] - (l + m - 1) * P_lm[l-2, m]) / (denominator + 1e-9)

    for l in range(l_max + 1):
        for m in range(l + 1):
            log_fact_l_minus_m = math.lgamma(l - m + 1)
            log_fact_l_plus_m = math.lgamma(l + m + 1)
            log_norm_factor_sq = math.log(2*l + 1) + log_fact_l_minus_m - (math.log(4 * math.pi) + log_fact_l_plus_m)
            norm_factor = math.exp(0.5 * log_norm_factor_sq)
            
            exp_im_phi = torch.exp(1j * m * phi.to(torch.complex64))
            Y_lm[l, m] = norm_factor * P_lm[l, m] * exp_im_phi
            if m > 0:
                Y_lm[l, -m] = ((-1)**m) * torch.conj(Y_lm[l, m])

    mask = (~is_zero_vec).to(torch.complex64)
    Y_lm[0, 0] = Y_lm[0, 0]
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            if (l, m) in Y_lm:
                Y_lm[l, m] = Y_lm[l, m] * mask     
    return Y_lm

class MagneticACEDescriptor:
    def __init__(self, elements, rc=5.0, n_max=3, l_max=4, nu_max=2, m_cut=4.0,
                 use_spin_invariants=True, pos_scale=1.0, spin_scale=1.0,
                 epsilon=1e-7, mag_noise_threshold=0.35, pos_noise_threshold=1e-8,
                 pole_threshold=1e-6):
        self.rc = rc
        self.n_max = n_max
        self.l_max = l_max
        self.m_cut = m_cut
        self.nu_max = nu_max
        self.pos_scale = pos_scale
        self.spin_scale = spin_scale
        self.epsilon = epsilon
        self.mag_noise_threshold = mag_noise_threshold
        self.pos_noise_threshold = pos_noise_threshold
        self.pole_threshold = pole_threshold
        self.use_spin_invariants = use_spin_invariants
        self.l_phi_max = 2 * self.l_max
        self.elements = sorted(elements)
        self.element_map = {e: i for i, e in enumerate(self.elements)}
        self.number_map = {Atoms(e).numbers[0]: i for i, e in enumerate(self.elements)}
        self.cg_couplings_phi = defaultdict(list)
        self.cg_couplings_D_AA = defaultdict(list)
        self.cg_couplings_D_MA = defaultdict(list)
        self._angular_dims_A, self._angular_splits_A = self._get_angular_dims(self.l_phi_max)
        self.n_angular_total_A = sum(self._angular_splits_A)
        self._angular_dims_M, self._angular_splits_M = self._get_angular_dims(self.l_max)
        self.n_angular_total_M = sum(self._angular_splits_M)
        self._precompute_cg_coefficients()
        self._precompute_torch_variables()
        self.n_channels_j = self.n_max * self.n_max * len(self.elements)
        self.n_channels_i = self.n_max
        self.n_channels_spin = self.n_max * len(self.elements)
        self.descriptor_dimension = self.get_descriptor_dimension()

    def _safe_norm(self, x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        return torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=keepdim) + self.epsilon)

    def _get_angular_dims(self, l_max_val):
        dims = {}
        splits = []
        idx = 0
        for l in range(l_max_val + 1):
            m_indices = list(range(-l, l + 1))
            for m in m_indices:
                dims[(l, m)] = idx
                idx += 1
            splits.append(len(m_indices))
        return dims, splits

    def get_descriptor_dimension(self) -> int:
        total_dim = 0
        total_dim += (self.n_channels_j + self.n_channels_i)
        if self.nu_max >= 2:
            n_combos_AA = (self.n_channels_j * (self.n_channels_j + 1)) // 2
            total_dim += n_combos_AA * self.l_phi_max 
            total_dim += self.n_channels_i * self.n_channels_j * (self.l_max + 1)
            if self.use_spin_invariants:
                n_combos_SS = (self.n_channels_spin * (self.n_channels_spin + 1)) // 2
                total_dim += n_combos_SS * (self.l_max + 1)
                total_dim += self.n_channels_i * self.n_channels_spin * (self.l_max + 1)
        return total_dim

    def _precompute_cg_coefficients(self):
        for l_r in range(self.l_max + 1):
            for l_m in range(self.l_max + 1):
                L_phi_max = l_r + l_m
                for L_phi in range(abs(l_r - l_m), L_phi_max + 1):
                    key = (l_r, l_m, L_phi)
                    if key in self.cg_couplings_phi: continue
                    couplings = []
                    for m_r in range(-l_r, l_r + 1):
                        for m_m in range(-l_m, l_m + 1):
                            M_phi = m_r + m_m
                            if abs(M_phi) > L_phi: continue
                            cg = float(clebsch_gordan(l_r, l_m, L_phi, m_r, m_m, M_phi))
                            if abs(cg) > 1e-9: couplings.append((m_r, m_m, M_phi, cg))
                    if couplings: self.cg_couplings_phi[key] = couplings
        for l in range(1, self.l_phi_max + 1):
            couplings = []
            for m in range(-l, l + 1):
                cg = float(clebsch_gordan(l, l, 0, m, -m, 0))
                if abs(cg) > 1e-9: couplings.append((m, -m, 0, cg))
            if couplings: self.cg_couplings_D_AA[(l,l,0)] = couplings
        for l in range(self.l_max + 1):
            couplings = []
            if l == 0: couplings.append((0, 0, 0, 1.0))
            else:
                for m in range(-l, l + 1):
                    cg = float(clebsch_gordan(l, l, 0, m, -m, 0))
                    if abs(cg) > 1e-9: couplings.append((m, -m, 0, cg))
            if couplings: self.cg_couplings_D_MA[(l,l,0)] = couplings

    def _precompute_torch_variables(self):
        def to_torch(couplings_dict):
            torch_dict = {}
            for key, couplings in couplings_dict.items():
                m_and_coeffs = torch.tensor(couplings, dtype=torch.float32, device=DEVICE)
                torch_dict[key] = {
                    "m1s": m_and_coeffs[:, 0].long(),
                    "m2s": m_and_coeffs[:, 1].long(),
                    "m3s": m_and_coeffs[:, 2].long(),
                    "coeffs": m_and_coeffs[:, 3]
                }
            return torch_dict
        self.torch_cg_couplings_phi = to_torch(self.cg_couplings_phi)
        self.torch_cg_couplings_D_AA = to_torch(self.cg_couplings_D_AA)
        self.torch_cg_couplings_D_MA = to_torch(self.cg_couplings_D_MA)

    def compute_from_precomputed_neighbors(self, positions, numbers, magmoms, center_indices, neighbor_indices, shifts, cell, batch_idx):
        n_total_atoms = positions.shape[0]
        n_pairs = center_indices.shape[0]
        if n_pairs == 0:
            return torch.zeros(n_total_atoms, self.descriptor_dimension, device=DEVICE, dtype=torch.float32)
        
        if magmoms.ndim == 1:
            magmoms_3d = torch.zeros_like(positions)
            magmoms_3d[:, 2] = magmoms
        else:
            magmoms_3d = magmoms

        pos_i = positions[center_indices]
        pos_j = positions[neighbor_indices]
        numbers_j = numbers[neighbor_indices]
        batch_idx_of_pairs = batch_idx[center_indices]
        cell_of_pairs = cell[batch_idx_of_pairs]
        cartesian_shifts = torch.einsum("ni,nij->nj", shifts, cell_of_pairs)
        r_ij_vec = pos_j - pos_i + cartesian_shifts

        theta_r, phi_r, is_zero_r = self._torch_cartesian_to_spherical(r_ij_vec, is_position=True)
        mag_j_3d = magmoms_3d[neighbor_indices]
        theta_m_j, phi_m_j, is_zero_m_j = self._torch_cartesian_to_spherical(mag_j_3d, is_position=False)

        Y_r_all = self._get_Y_lm_matrix(self.l_max, theta_r, phi_r, is_zero_r)
        Y_m_j_all = self._get_Y_lm_matrix(self.l_max, theta_m_j, phi_m_j, is_zero_m_j)

        r_ij_dist = self._safe_norm(r_ij_vec, dim=-1)
        mag_j_norm = self._safe_norm(mag_j_3d, dim=-1)
        
        B_R = self._torch_radial_basis(r_ij_dist)    
        B_Mj = self._torch_magnetic_basis(mag_j_norm) 
        B_Z = self._torch_species_encoding(numbers_j) 

        mag_i_3d = magmoms_3d
        theta_m_i, phi_m_i, is_zero_m_i = self._torch_cartesian_to_spherical(mag_i_3d, is_position=False)
        Y_m_i_all = self._get_Y_lm_matrix(self.l_max, theta_m_i, phi_m_i, is_zero_m_i)
        
        mag_i_norm = self._safe_norm(mag_i_3d, dim=-1)
        B_Mi = self._torch_magnetic_basis(mag_i_norm) 

        non_angular_phi = torch.einsum("ip,jp,kp->pijk", B_R, B_Mj, B_Z).reshape(n_pairs, -1).T
        angular_phi_dict = self._couple_phi_tensors(Y_r_all, Y_m_j_all)
        angular_phi = self._pack_tensors_to_matrix(angular_phi_dict, self.l_phi_max)
        phi_j_weighted = (non_angular_phi.unsqueeze(1) * angular_phi.unsqueeze(0)).reshape(-1, n_pairs)
        A_i_flat = torch.zeros(self.n_channels_j * self.n_angular_total_A, n_total_atoms, device=DEVICE, dtype=torch.complex64)
        scatter_add(phi_j_weighted, center_indices, dim=1, out=A_i_flat)
        A_i = A_i_flat.reshape(self.n_channels_j, self.n_angular_total_A, n_total_atoms)

        angular_M = self._pack_tensors_to_matrix(Y_m_i_all, self.l_max)
        M_i = B_Mi.unsqueeze(1) * angular_M.unsqueeze(0)

        A_i_dict = self._unpack_tensor(A_i, self.l_phi_max, self._angular_splits_A)
        M_i_dict = self._unpack_tensor(M_i, self.l_max, self._angular_splits_M)

        D_i_list = []
        if 0 in A_i_dict and A_i_dict[0].shape[1] == 1: D_i_list.append(A_i_dict[0].squeeze(1).real)
        if 0 in M_i_dict and M_i_dict[0].shape[1] == 1: D_i_list.append(M_i_dict[0].squeeze(1).real)

        if self.nu_max >= 2:
            D_AA_list = self._compute_invariants(A_i_dict, A_i_dict, self.torch_cg_couplings_D_AA, self.l_phi_max)
            D_MA_list = self._compute_invariants(M_i_dict, A_i_dict, self.torch_cg_couplings_D_MA, self.l_max)
            if D_AA_list: D_i_list.append(self.pos_scale * torch.cat(D_AA_list, dim=0))
            if D_MA_list: D_i_list.append(self.pos_scale * torch.cat(D_MA_list, dim=0))

            if self.use_spin_invariants:
                S_i_dict = {}
                non_angular_spin = torch.einsum("ip,jp->pij", B_Mj, B_Z).reshape(n_pairs, -1).T
                for l in range(self.l_max + 1):
                    phi_spin_l = non_angular_spin.unsqueeze(1) * Y_m_j_all[l].unsqueeze(0)
                    n_ang = 2*l+1
                    S_flat = torch.zeros(self.n_channels_spin * n_ang, n_total_atoms, device=DEVICE, dtype=torch.complex64)
                    scatter_add(phi_spin_l.reshape(-1, n_pairs), center_indices, dim=1, out=S_flat)
                    S_i_dict[l] = S_flat.reshape(self.n_channels_spin, n_ang, n_total_atoms)
                
                D_SS_list = self._compute_invariants(S_i_dict, S_i_dict, self.torch_cg_couplings_D_MA, self.l_max)
                D_MS_list = self._compute_invariants(M_i_dict, S_i_dict, self.torch_cg_couplings_D_MA, self.l_max)
                if D_SS_list: D_i_list.append(self.spin_scale * torch.cat(D_SS_list, dim=0))
                if D_MS_list: D_i_list.append(self.spin_scale * torch.cat(D_MS_list, dim=0))

        if not D_i_list: return torch.zeros(n_total_atoms, self.descriptor_dimension, device=DEVICE, dtype=torch.float32)
        return torch.cat(D_i_list, dim=0).T

    def _couple_phi_tensors(self, Y_r_all, Y_m_all):
        n_pairs = list(Y_r_all.values())[0].shape[1]
        phi_tensors = {l: torch.zeros(2*l+1, n_pairs, dtype=torch.complex64, device=DEVICE) for l in range(self.l_phi_max+1)}
        for l_r in range(self.l_max + 1):
            for l_m in range(self.l_max + 1):
                coupled = self._couple_two_tensors(Y_r_all[l_r], l_r, Y_m_all[l_m], l_m, self.torch_cg_couplings_phi)
                for L, t in coupled.items():
                    if L <= self.l_phi_max: phi_tensors[L] += t
        return phi_tensors

    def _compute_invariants(self, T1_dict, T2_dict, cg_couplings, l_max_val):
        invariants = []
        n_atoms = list(T1_dict.values())[0].shape[-1]
        is_self = (T1_dict is T2_dict)
        for l in range(l_max_val + 1):
            key = (l,l,0)
            if key not in cg_couplings or l not in T1_dict or l not in T2_dict: continue
            c = cg_couplings[key]
            m1, m2, coeffs = c["m1s"]+l, c["m2s"]+l, c["coeffs"]
            if m1.max() >= T1_dict[l].shape[1]: continue
            term = coeffs.view(1,1,-1,1) * T1_dict[l][:,m1,:].unsqueeze(1) * T2_dict[l][:,m2,:].unsqueeze(0)
            inv = torch.sum(term, dim=2)
            if is_self:
                idx = torch.triu_indices(T1_dict[l].shape[0], T1_dict[l].shape[0], device=DEVICE)
                inv = inv[idx[0], idx[1]]
            else:
                inv = inv.reshape(-1, n_atoms)
            invariants.append(inv.real)
        return invariants

    def _pack_tensors_to_matrix(self, tensor_dict, l_max_val):
        return torch.cat([tensor_dict.get(l, torch.zeros(2*l+1, list(tensor_dict.values())[0].shape[1], dtype=torch.complex64, device=DEVICE)) for l in range(l_max_val+1)], dim=0)

    def _unpack_tensor(self, flat, l_max_val, splits):
        t = torch.split(flat, splits, dim=1)
        return {l: t[l] for l in range(l_max_val+1)}

    def _get_Y_lm_matrix(self, l_max, theta, phi, is_zero):
        Y = _torch_sph_harm_custom(l_max, theta, phi, is_zero)
        return {l: torch.stack([Y.get((l,m), torch.zeros_like(theta, dtype=torch.complex64)) for m in range(-l,l+1)], dim=0) for l in range(l_max+1)}

    def _couple_two_tensors(self, Y1, l1, Y2, l2, cg):
        out = defaultdict(lambda: torch.zeros(1, Y1.shape[1], dtype=torch.complex64, device=DEVICE))
        for L in range(abs(l1-l2), l1+l2+1):
            key = (l1,l2,L)
            if key not in cg: continue
            c = cg[key]
            val = c["coeffs"].view(-1,1) * Y1[c["m1s"]+l1] * Y2[c["m2s"]+l2]
            if out[L].shape[0] != 2*L+1: out[L] = torch.zeros(2*L+1, Y1.shape[1], dtype=torch.complex64, device=DEVICE)
            scatter_add(val, c["m3s"]+L, dim=0, out=out[L])
        return out

    def _torch_chebyshev_basis(self, x, n_max):
        if x.dim()==0: x=x.unsqueeze(0)
        T = [torch.ones_like(x), x]
        for n in range(2, n_max): T.append(2*x*T[-1] - T[-2])
        return torch.stack(T, dim=0)

    def _torch_cartesian_to_spherical(self, vec, is_position=True):
        r = self._safe_norm(vec, dim=-1)
        th = self.pos_noise_threshold if is_position else self.mag_noise_threshold
        is_zero = r < th
        is_pole = self._safe_norm(vec[...,:2], dim=-1) < self.pole_threshold
        safe_mask = is_zero | is_pole
        vx = torch.where(safe_mask, torch.tensor(1.0, device=DEVICE), vec[...,0])
        vy = torch.where(safe_mask, torch.tensor(0.0, device=DEVICE), vec[...,1])
        vz = torch.where(safe_mask, torch.tensor(0.0, device=DEVICE), vec[...,2])
        r_xy = self._safe_norm(torch.stack([vx, vy], dim=-1), dim=-1)
        theta = torch.atan2(r_xy, vz)
        phi = torch.atan2(vy, vx)
        theta = torch.where(is_zero, torch.zeros_like(theta), theta)
        phi = torch.where(is_zero | is_pole, torch.zeros_like(phi), phi)
        return theta, phi, is_zero

    def _torch_radial_basis(self, r):
        r = torch.clamp(r, 0.0, self.rc)
        fc = 0.5 * (1 + torch.cos(torch.pi * r / self.rc))
        return fc * self._torch_chebyshev_basis(2*(r/self.rc)-1, self.n_max)

    def _torch_magnetic_basis(self, m):
        return self._torch_chebyshev_basis(2*(torch.clamp(m, 0.0, self.m_cut)/self.m_cut)-1, self.n_max)

    def _torch_species_encoding(self, nums):
        enc = torch.zeros(nums.shape[0], len(self.elements), device=DEVICE)
        for n, i in self.number_map.items(): enc[nums.long()==n, i] = 1.0
        return enc.T

# =============================================================================
#  PART 3: 邻居列表构建 & 推理逻辑
#  [重要更新] 此函数逻辑已与 preprocess.py 完全对齐
# =============================================================================

def build_neighbor_list_consistent(atoms: Atoms, rc: float) -> Dict[str, torch.Tensor]:
    """
    使用 ase.neighborlist.build_neighbor_list 构建邻居列表，
    确保处理逻辑、NaN 清洗、磁矩读取与训练集预处理脚本 (preprocess.py) 完全一致。
    """
    # 1. 基本信息提取 & NaN 清洗
    positions = torch.nan_to_num(torch.tensor(atoms.get_positions(), dtype=torch.float32), nan=0.0).to(DEVICE)
    numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).to(DEVICE)
    
    # 使用 complete=True 确保获取 3x3 晶胞矩阵
    cell = torch.tensor(atoms.get_cell(complete=True).array, dtype=torch.float32).to(DEVICE).unsqueeze(0) # (1, 3, 3)
    # pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool).to(DEVICE) # 暂时不需要 pbc tensor，build_neighbor_list 内部会用

    # 2. 磁矩提取 (增强版逻辑，对齐 preprocess.py)
    magmoms_raw = None
    if atoms.has('magmoms'):
        magmoms_raw = atoms.get_array('magmoms')
    elif 'magnetic_moment' in atoms.arrays:
        magmoms_raw = atoms.get_array('magnetic_moment')
    elif 'magnetic_moments' in atoms.arrays:
        magmoms_raw = atoms.get_array('magnetic_moments')
    elif atoms.has('initial_magmoms'):
        magmoms_raw = atoms.get_initial_magnetic_moments()
    else:
        magmoms_raw = np.zeros(len(atoms))
    
    # 维度修正
    if magmoms_raw.ndim == 1:
        magmoms_3d = np.zeros((len(atoms), 3))
        magmoms_3d[:, 2] = magmoms_raw
        magmoms = magmoms_3d
    else:
        magmoms = magmoms_raw
    
    # 转 Tensor & 清理 NaN
    magmoms_tensor = torch.nan_to_num(torch.tensor(magmoms, dtype=torch.float32), nan=0.0).to(DEVICE)

    # 打印一下最大磁矩，确认数据读取状态
    max_mag = torch.max(torch.norm(magmoms_tensor, dim=1)).item()
    print(f"  [Info] Max Magmom norm: {max_mag:.4f}")

    # 3. 邻居列表构建 (使用 build_neighbor_list 类)
    # 逻辑: cutoff = rc/2 对所有原子，总截断距离 = rc
    cutoffs = [rc / 2.0] * len(atoms)
    nl = build_neighbor_list(atoms, cutoffs=cutoffs, self_interaction=False, bothways=True)
    
    center_indices_list = []
    neighbor_indices_list = []
    shifts_list = []
    
    for i in range(len(atoms)):
        indices_j, offsets = nl.get_neighbors(i)
        center_indices_list.extend([i] * len(indices_j))
        neighbor_indices_list.extend(indices_j)
        shifts_list.extend(offsets)
        
    # 转换为 Tensor
    center_indices = torch.tensor(center_indices_list, dtype=torch.long, device=DEVICE)
    neighbor_indices = torch.tensor(neighbor_indices_list, dtype=torch.long, device=DEVICE)
    shifts = torch.tensor(np.array(shifts_list), dtype=torch.float32, device=DEVICE)
    
    # Batch Index (单帧推理，全为0)
    batch_idx = torch.zeros(len(atoms), dtype=torch.long, device=DEVICE)

    return {
        "positions": positions,
        "numbers": numbers,
        "magmoms": magmoms_tensor,
        "center_indices": center_indices,
        "neighbor_indices": neighbor_indices,
        "shifts": shifts,
        "cell": cell,
        "batch_idx": batch_idx
    }

def process_single_frame(atoms: Atoms, model: MagneticNEP, desc_gen: MagneticACEDescriptor, frame_idx: int, filename: str):
    """处理单帧"""
    # 1. 准备数据 (使用更新后的一致性函数)
    input_data = build_neighbor_list_consistent(atoms, CONFIG["descriptor"]["rc"])
    
    # 2. 开启梯度计算
    input_data["positions"].requires_grad_(True)
    input_data["magmoms"].requires_grad_(True)
    
    # 3. 计算描述符
    descriptors = desc_gen.compute_from_precomputed_neighbors(
        input_data["positions"], input_data["numbers"], input_data["magmoms"],
        input_data["center_indices"], input_data["neighbor_indices"], input_data["shifts"],
        input_data["cell"], input_data["batch_idx"]
    )

    # 导出描述符用于对比
#    np.savetxt('nep_cpp/build/python_descriptors.txt', descriptors.detach().cpu().numpy(), fmt='%.10f')

    # 4. 前向传播
    pred_atomic_energies = model(descriptors)
    pred_total_energy = torch.sum(pred_atomic_energies)
    
    # 5. 反向传播 (Forces & MagForces)
    grads = torch.autograd.grad(
        outputs=pred_total_energy,
        inputs=[input_data["positions"], input_data["magmoms"]],
        create_graph=False
    )
    
    pred_forces = -grads[0]
    pred_magforces = grads[1]

    # 6. 输出结果
    energy_val = pred_total_energy.item()
    forces_val = pred_forces.detach().cpu().numpy()
    magforces_val = pred_magforces.detach().cpu().numpy()

    print(f"--- Results for {filename} (Frame {frame_idx}) ---")
    print(f"Total Energy: {energy_val:.6f} eV")
    
    print("Forces (eV/A):")
    for row in forces_val:
        print(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")

    print("Magnetic Forces (eV/mu_B):")
    for row in magforces_val:
        print(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
    print("\n" + "="*40 + "\n")

def process_single_file(xyz_path, model, desc_gen):
    """读取文件中的所有帧并处理"""
    try:
        # 读取所有帧
        atoms_list = read(xyz_path, index=':')
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
    except Exception as e:
        print(f"Error reading {xyz_path}: {e}")
        return

    print(f"Processing {os.path.basename(xyz_path)}: Found {len(atoms_list)} frames.")
    
    for i, atoms in enumerate(atoms_list):
        process_single_frame(atoms, model, desc_gen, i, os.path.basename(xyz_path))

def main():
    model_path = CONFIG["paths"]["model_path"]
    xyz_folder = CONFIG["paths"]["xyz_folder"]

    print(f"--- MagneticNEP Batch Prediction ---")
    print(f"Model Path: {model_path}")
    print(f"XYZ Folder: {xyz_folder}")
    print(f"Device: {DEVICE}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(xyz_folder):
        print(f"Error: XYZ folder not found at {xyz_folder}")
        sys.exit(1)

    xyz_files = sorted(glob.glob(os.path.join(xyz_folder, "*.xyz")))
    if not xyz_files:
        print(f"Warning: No .xyz files found in {xyz_folder}")
        sys.exit(0)
    
    print(f"Found {len(xyz_files)} structure files (checking for multi-frames in each).\n")

    print("Initializing Descriptor and Model...")
    desc_gen = MagneticACEDescriptor(**CONFIG["descriptor"])
    input_dim = desc_gen.descriptor_dimension

    # 自动检测模型格式并加载
    try:
        # 尝试作为TorchScript模型加载 (traced.pt格式)
        if model_path.endswith('_traced.pt') or model_path.endswith('.pt'):
            try:
                model = torch.jit.load(model_path, map_location=DEVICE)
                model.eval()
                print(f"Model loaded successfully as TorchScript (traced format).\n")
            except:
                # 如果失败，尝试作为state_dict加载
                print("TorchScript loading failed, trying state_dict format...")
                model = MagneticNEP(input_dim,
                                    hidden_dim=CONFIG["model"]["hidden_dim"],
                                    dropout_rate=CONFIG["model"]["dropout_rate"]).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                print("Model loaded successfully as state_dict (.pth format).\n")
        else:
            # .pth格式，使用state_dict加载
            model = MagneticNEP(input_dim,
                                hidden_dim=CONFIG["model"]["hidden_dim"],
                                dropout_rate=CONFIG["model"]["dropout_rate"]).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            print("Model loaded successfully as state_dict (.pth format).\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    start_time = time.time()
    for xyz_file in xyz_files:
        process_single_file(xyz_file, model, desc_gen)
    
    end_time = time.time()
    print(f"Processing complete. Time taken: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()
