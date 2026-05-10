"""Generate Fig. 2 (trainability) from cached experiment outputs.

Run from the experiments/ directory so relative paths to ../dataset/,
./models/, and ./output_PO_* resolve correctly.
"""

import numpy as np
import pandas as pd
import cudaq
from cudaq import spin
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import plotly.graph_objects as go
from math import sqrt, pi
import plotly.express as px
from tqdm import tqdm
import sys
import os

import _paths  # noqa: F401
from sklearn.decomposition import PCA
import torch

from Utils.qaoaCUDAQ import po_normalize, ret_cov_to_QUBO, qubo_to_ising, process_ansatz_values, state_to_return, pauli_to_int, int_to_pauli, basis_T_to_pauli,\
    reversed_str_bases_to_init_state, kernel_qaoa_Preserved, kernel_flipped, get_optimizer, optimizer_names, all_state_to_return, get_init_states

import time

cudaq.set_target("nvidia")

np.random.seed(42)

# !! warning: Representing Pauli words using INTEGER !!

print(4**30 // 10**18)

import seaborn as sns
from copulas.multivariate import GaussianMultivariate
import joblib
data_cov = pd.read_csv("../dataset/top_50_us_stocks_data_20250526_011226_covariance.csv")
data_ret_p = pd.read_csv("../dataset/top_50_us_stocks_returns_price.csv")

data = data_ret_p.drop("Ticker", axis=1)
sns.jointplot(data=data, x='Price', y='Average_Return', kind='reg')
plt.show()

print(data["Price"].min(), data["Price"].max())
print(data["Average_Return"].min(), data["Average_Return"].max())

covv = np.array(data_cov.iloc[:, 1:])
print(covv.min(), covv.max())

for i in range(covv.shape[0]):
    covv[i] = np.roll(covv[i], -i)

GM2 = GaussianMultivariate()
print(data_cov)
print(covv[:4, :4])

GM2_load = joblib.load('./models/gaussian_copula_covariance.pkl')
samples_cov = GM2_load.sample(50)
samples_cov = np.array(samples_cov)
samples_cov = np.abs(samples_cov)
for i in range(samples_cov.shape[0]):
    samples_cov[i] = np.roll(samples_cov[i], i)
samples_cov = (samples_cov + samples_cov.T) / 2
print(covv.min(), covv.max())
print(samples_cov.min(), samples_cov.max())
print(samples_cov[:4, :4])

GM_loaded = joblib.load('./models/gaussian_copula.pkl')
samples = GM_loaded.sample(50)
print(samples["Average_Return"].min(), samples["Average_Return"].max())
sns.jointplot(data=samples, x='Price', y='Average_Return', kind='reg')
plt.show()

# # HAMILTONIAN BY CUDAQ

# ### by random values

B = 100
ret = np.array([1.0, 1.25, 1.5])
cov = np.random.rand(3, 3)
cov += cov.T
print(cov)
P = np.array([100, 100, 100])
lamb = 0 # Budget Penalty
q = 0 # Volatility Weight

# P_b, ret_b, cov_b = po_normalize(B, P, ret, cov)
P_bb, ret_bb, cov_bb, n_qubit, n_max, C = po_normalize(B, P, ret, cov)

print("n_qubit:", n_qubit)
print("n_max:", n_max)

QU = -ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)
H = qubo_to_ising(QU, lamb)

print("Hamiltonian:", H)
idx_1, coeff_1, idx_2_a, idx_2_b, coeff_2 = process_ansatz_values(H)

print(idx_1)
print(coeff_1, end="\n\n")
print(idx_2_a)
print(idx_2_b)
print(coeff_2)

# ### by stock values

data_cov = pd.read_csv("../dataset/top_50_us_stocks_data_20250526_011226_covariance.csv")
data_ret_p = pd.read_csv("../dataset/top_50_us_stocks_returns_price.csv")

nn = 3 # num assets
# B = 1500
B = 270
lamb = 0.01 # Budget Penalty
q = 1 # Volatility Weight

data_cov = data_cov.drop("Ticker", axis=1).iloc[:nn, :nn]
print(data_cov)
data_ret_p = data_ret_p.iloc[:nn]
# data_ret_p.loc[0, "Average_Return"] = 0.002070
# data_ret_p.loc[1, "Average_Return"] = 0.000050
# data_ret_p.loc[2, "Average_Return"] = 0.002070
print(data_ret_p)
stock_names = data_ret_p["Ticker"].tolist()
data_ret_p = data_ret_p.drop("Ticker", axis=1)

data_cov = data_cov.to_numpy()
data_ret_p = data_ret_p.to_numpy()

data_ret = data_ret_p[:, 0]
data_p = data_ret_p[:, 1]

data_p = np.array([174.34238699, 129.46979175, 163.35661173])
data_ret = np.array([0.00147772, 0.00055953, 0.00173287])
B = 281.25

print(data_cov.shape)
print(data_ret.round(5))
print(data_p.round(2))
print(stock_names)

a = 5
a += spin.z(0)
print(a.canonicalize())

P_bb, ret_bb, cov_bb, n_qubit, n_max, C = po_normalize(B, data_p, data_ret, data_cov)

print(cov_bb)
print("n_qubit:", n_qubit)

QU = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)
H = -qubo_to_ising(QU, lamb).canonicalize() * 250

print(QU)
print("Hamiltonian:", H/500)
idx_1, coeff_1, idx_2_a, idx_2_b, coeff_2 = process_ansatz_values(H)

print(idx_1)
print(coeff_1, end="\n\n")
print(idx_2_a)
print(idx_2_b)
print(coeff_2)

@cudaq.kernel
def kernel_simple(qb:int, idxs: List[int])-> None:
    qvec = cudaq.qvector(qb)
    for idx in idxs:
        x(qvec[idx])

v = np.array(list(map(int, "0111")))
lambb = 0.01 * 1
qq = 1
QU2 = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lambb, qq)
QU2_lamb = ret_cov_to_QUBO(np.zeros_like(ret_bb), np.zeros_like(cov_bb), P_bb, lambb*1, 0.0)
HH = -qubo_to_ising(QU2, lambb).canonicalize()
print(QU2)
print(v @ QU2 @ v - lambb)
print(v @ QU2_lamb @ v - lambb)

def get_init_states_(state_return, num_init_bases, n_qubits):
    sorted_idx = np.argsort(-state_return)
    init_states = []
    for i in sorted_idx[:num_init_bases]:
        init_states.append(bin(i)[2:].zfill(n_qubits))
    return init_states

state = all_state_to_return(len(v), lambb, QU2)
print(state)
init_states = get_init_states(state, 4, len(v))
print(init_states)

sss = cudaq.get_state(kernel_simple, n_qubit, [1, 2, 3])
exp = cudaq.observe(kernel_simple, HH, n_qubit, [1, 2, 3]).expectation()
print(sss)
print(exp)

print((P_bb.sum()-1)**2)

s = 1
for i in range(len(v)):
    # s -= P_bb[i]**2 * v[i]
    s -= 2 * P_bb[i] * v[i]
for i in range(len(v)):
    for j in range(len(v)):
        s += 1 * P_bb[i] * P_bb[j] * v[i] * v[j]
print(s * lambb)

print(0.0005**2)

print(P_bb)
print(ret_bb)
print(cov_bb, end="\n\n")

print(lambb * (v@P_bb - 1)**2) # Penalty term
print(-1 * v@ret_bb + qq * v@cov_bb@v + lambb * (v@P_bb - 1)**2) # Min objective

# # Time Benchmark

print(list(map(int, "0010")))

# qb = 22
# st = time.time()
# l = np.zeros((1<<qb, qb))
#     s = bin(i)[2:].zfill(qb)
#     ll = np.array(list(map(int, s)))
#     l[i] = ll

# qb = 25
# st = time.time()
# l = np.zeros((qb, 1<<qb), dtype=np.float32)
# a_0 = np.zeros(1<<qb, dtype=np.float32)
# a_1 = np.ones(1<<qb, dtype=np.float32)
# idxx = np.arange(1<<qb, dtype=np.int32)
# # print(a_0, a_1)
#     l[i] = np.where(idxx%(1<<(qb-i))<(1<<(qb-i-1)), a_0,  a_1)
# # print(l)

# qb = 25
# st = time.time()
# lt = torch.zeros((qb, 1<<qb), dtype=torch.float32, device='cuda')
# a_0 = torch.zeros(1<<qb, dtype=torch.float32, device='cuda')
# a_1 = torch.ones(1<<qb, dtype=torch.float32, device='cuda')
# idxx = torch.arange(1<<qb, device='cuda')
#     lt[i] = torch.where(idxx%(1<<(qb-i))<(1<<(qb-i-1)), a_0,  a_1)
# lt = lt.numpy(force=True)

# def state_to_return(s, B, C, d_ret, d_p):
#     l = np.array(list(map(int, s)))
#     P = d_p @ C
#     ret_C = (d_ret * d_p) @ C
#     ss = l @ ret_C
#     bud = l @ P
#     return ss, bud <= B

# ex_ret, in_budget = state_to_return("0001", B, C, data_ret, data_p)

# # CUDA QAOA

idx_1_use, coeff_1_use = idx_1, coeff_1
idx_2_a_use, idx_2_b_use, coeff_2_use = idx_2_a, idx_2_b, coeff_2

# idx_1_use, coeff_1_use = idx_1_qis, coeff_1_qis
# idx_2_a_use, idx_2_b_use, coeff_2_use = idx_2_a_qis, idx_2_b_qis, coeff_2_qis

print(H)
print(n_qubit)

# bases = ["0001", "0010"]
# T = np.array([[0, 1], [1, 0]])
# mixer_s, mixer_c, A_all = basis_T_to_pauli(bases, T, len(bases[0]))

# # AA_all = A_all.copy()
# AA_all = 0.25 * spin.x(0) * spin.x(2) + 0.25 * spin.x(0) * spin.z(1) * spin.x(2) \
#        + 0.25 * spin.y(0) * spin.y(2) + 0.25 * spin.y(0) * spin.z(1) * spin.y(2)

# init_bases = ["100", "010", "001"]
# T = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

# init_bases = ["1000", "0100", "0001"]
# init_bases = ['0001', '1000', '0010']
# init_bases = ["100000", "010000", "000100"]

state = all_state_to_return(n_qubit, lamb, QU)
init_bases = get_init_states(state, 3, n_qubit)
# init_bases = ['0000', '1011', '1111']
print(init_bases)

T = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

mixer_s, mixer_c = basis_T_to_pauli(init_bases, T, n_qubit) 

print(f"{mixer_s}\n{mixer_c}")
init_bases = reversed_str_bases_to_init_state(init_bases, n_qubit)
print("init_bases:", abs(init_bases))

# state_return, in_budget = all_state_to_return(B, C, data_ret, data_p, 1.0)
# init_state = get_init_states(state_return, in_budget, 0.02, n_qubit)
# # init_state = get_init_states(state_return, in_budget, 0.2, n_qubit)
# n_bases = len(init_state)
# # print(init_state)
# T = np.zeros((n_bases, n_bases), dtype=np.float32)
# T[:-1, 1:] += np.eye(n_bases - 1, dtype=np.float32)
# T[1:, :-1] += np.eye(n_bases - 1, dtype=np.float32)
# T[0, -1] = T[-1, 0] = 1.0
# mixer_s, mixer_c = basis_T_to_pauli(init_state, T, n_qubit)
# # mixer_c *= 100
# init_bases = reversed_str_bases_to_init_state(init_state, n_qubit)
# # print(f"{mixer_s}\n{mixer_c}")

print(mixer_s[:2])
print(mixer_c[:2])
print(coeff_1_use)

a = "AB"
for i, c in enumerate(a):
    print(i, c)
print()
for i, c in reversed(list(enumerate(a[:-1]))):
    print(i, c)

def nplize(a):
    for i in range(len(a)):
        a[i] = np.array(a[i])
    return a

a = [[1, 2, np.array(3)], [1, 2, np.array(3)]]
print(nplize(a))
a = [1, 2, np.array(3)]
b = [*a]
print(b)

def prepare_preserving_ansatz(qubit_count: int, idx_1: List[int], coeff_1: List[float], idx_2_a: List[int], idx_2_b: List[int], coeff_2: List[float], mixer_str: List[cudaq.pauli_word], mixer_coeff: List[float]):
    def generate_list():
        return [[] for _ in range(qubit_count)]
    type_l, zeta_l, entang_l, runnum_l = [generate_list() for _ in range(4)]
    cou = 0
    all_gate, mk = [], []
    def remove(idx):
        tp = type_l[idx].pop()
        zt = zeta_l[idx].pop()
        et = entang_l[idx].pop()
        rn = runnum_l[idx].pop()
        mk[rn] = False
    def add(typee, idx, zeta, idx_en):
        nonlocal cou
        type_l[idx].append(typee)
        zeta_l[idx].append(zeta)
        entang_l[idx].append(idx_en)
        runnum_l[idx].append(cou)
        mk.append(typee != 5)
        all_gate.append([typee, idx, *zeta, idx_en])
        cou += 1
    def nplize(a):
        for i in range(len(a)):
            a[i] = np.array(a[i], dtype=np.float32)
        return a
    def is_zero(val, bound=1e-8):
        return abs(val) < bound
    def push(typee, idx, zeta=[0, 0, 0], idx_en=-1): # typee[0: RX, 1: RY, 2: RZ, 3: H, 4: CX-control, 5: CX-target], zeta: (const_coeff, problem_coeff, mixer_coeff)
        nonlocal cou
        if len(type_l[idx]) > 0 and type_l[idx][-1] == typee:
            if typee <= 3:
                if not (is_zero(zeta_l[idx][-1][0] + zeta[0]) and is_zero(zeta_l[idx][-1][1] + zeta[1]) and is_zero(zeta_l[idx][-1][2] + zeta[2])):
                    for i in range(3):
                        zeta_l[idx][-1][i] += zeta[i]
                        all_gate[runnum_l[idx][-1]][i + 2] = zeta_l[idx][-1][i]
                    print("^", idx, runnum_l[idx][-1], typee, zeta_l[idx][-1])
                else:
                    remove(idx)
            if typee == 4:
                if len(entang_l[entang_l[idx][-1]]) > 0 and idx_en == entang_l[idx][-1] and idx == entang_l[entang_l[idx][-1]][-1]:
                    remove(idx)
                    remove(entang_l[idx][-1])
                else:
                    add(typee, idx, zeta, idx_en)
                    add(5, idx_en, [0, 0, 0], idx)
        else:
            add(typee, idx, zeta, idx_en)
            if typee == 4:
                add(5, idx_en, [0, 0, 0], idx)

    def push_pauli_string(strr, coeff):
        for i, p in enumerate(strr):
            if p == "X":
                push(3, i)
            elif p == "Y":
                push(0, i, [pi/2, 0, 0])
                # push(3, i)
        ll = -1
        for i, p in enumerate(strr):
            if p in ["X", "Y", "Z"]:
                if ll != -1:
                    push(4, ll, idx_en=i)
                    # push(5, i, idx_en=ll)
                ll = i
        push(2, ll, [0, 0, -2 * coeff])
        ll = -1
        for i, p in reversed(list(enumerate(strr))):
            if p in ["X", "Y", "Z"]:
                if ll != -1:
                    push(4, i, idx_en=ll)
                    # push(5, ll, idx_en=i)
                ll = i
        for i, p in enumerate(strr):
            if p == "X":
                push(3, i)
            elif p == "Y":
                # push(3, i)
                push(0, i, [-pi/2, 0, 0])
    
    for j in range(len(idx_1)):
        push(2, idx_1[j], [0, 2 * coeff_1[j], 0])
    for j in range(len(idx_2_a)):
        push(4, idx_2_a[j], idx_en=idx_2_b[j])
        # push(5, idx_2_b[j], idx_en=idx_2_a[j])
        push(2, idx_2_b[j], [0, 2 * coeff_2[j], 0])
        push(4, idx_2_a[j], idx_en=idx_2_b[j])
        # push(5, idx_2_b[j], idx_en=idx_2_a[j])
    for j in range(len(mixer_str)):
        push_pauli_string(mixer_str[j], mixer_coeff[j])

    return [nplize(i) for i in[type_l, zeta_l, entang_l, runnum_l]] + nplize([all_gate, mk])

type_l, zeta_l, entang_l, runnum_l, all_gate, mk = prepare_preserving_ansatz(n_qubit, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use, mixer_s, mixer_c.tolist())
# type_l, zeta_l, entang_l, runnum_l, all_gate, mk = prepare_preserving_ansatz(n_qubit, idx_1_use, coeff_1_use, [0], [1], [0.3], ["ZZII", "XXZI"], mixer_c[:2].tolist())
all_gate = all_gate[mk == 1].reshape(-1)

print(mk)

ag = all_gate.reshape(-1, 6)
al = [[] for _ in range(4)]
for i in range(len(ag)):
    al[int(ag[i][1])].append(ag[i])
for ii in range(len(al)):
    print(ii)
    for i in range(len(al[ii])):
        print(al[ii][i].tolist())
    print()

for ii in range(len(type_l)):
    print(ii)
    for i in range(len(type_l[ii])):
        print(type_l[ii][i], zeta_l[ii][i].tolist(), entang_l[ii][i], runnum_l[ii][i])
    print()

@cudaq.kernel
def kernel_cmpz_Preserved(thetas: List[float], qubit_count: int, layer_count: int, params: List[float], init_sup: List[complex]):
    qreg = cudaq.qvector(init_sup)

    for i in range(layer_count):
        for j in range(len(params) // 6):
            typee = params[6 * j]
            idx = int(params[6 * j + 1])
            zeta = params[6*j+2] + thetas[i] * params[6*j+3] + thetas[i+layer_count] * params[6*j+4]

            if typee  == 0: # RX
                rx(zeta, qreg[idx])
            elif typee == 1: # RY
                ry(zeta, qreg[idx])
            elif typee == 2: # RZ
                rz(zeta, qreg[idx])
            elif typee == 3: # H
                h(qreg[idx])
            elif typee == 4: # CX-control
                cx(qreg[idx], qreg[int(params[6*j+5])])

print(cudaq.draw(kernel_qaoa_Preserved, [1.0]*4, n_qubit, 1, idx_1_use, coeff_1_use, [0], [1], [0.3], ["ZZII", "XXZI"], mixer_c[:2], init_bases))

for i in ag:
    print(i.tolist())

print(cudaq.draw(kernel_cmpz_Preserved, [1.0]*4, n_qubit, 1, all_gate, init_bases))

print(mixer_s)

print(cudaq.draw(kernel_qaoa_Preserved, [1]*4, n_qubit, 1, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use, mixer_s, mixer_c, init_bases))

# # Ansatz Architecture

idx = 3
layer_count = 5
ansatz_idx = 0

parameter_count = layer_count * 2
optimizer, optimizer_name, FIND_GRAD = get_optimizer(idx)

optimizer.max_iterations = 1000

optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8, parameter_count)
print("Initial parameters = ", optimizer.initial_parameters)

print(init_bases)
kernel_qaoa_use = None
ansatz_fixed_param = None
if ansatz_idx == 0:
    ansatz_fixed_param = (int(n_qubit), layer_count, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use, mixer_s, mixer_c, init_bases)
    kernel_qaoa_use = kernel_qaoa_Preserved
elif ansatz_idx == 1:
    preserving_gates, mk = prepare_preserving_ansatz(n_qubit, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use, mixer_s, mixer_c.tolist())[-2:]
    preserving_gates = preserving_gates[mk == 1].reshape(-1)
    ansatz_fixed_param = (int(n_qubit), layer_count, preserving_gates, init_bases)
    kernel_qaoa_use = kernel_cmpz_Preserved

print(optimizer.episodes)

assert False

def plot_px_surface(points, nx=80, ny=80, method="linear", title="3D Surface", opacity=0.95):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ux = np.linspace(-5, 5, nx)
    uy = np.linspace(-5, 5, ny)
    Xg, Yg = np.meshgrid(ux, uy)
    Zg = griddata(points=np.c_[x, y], values=z, xi=(Xg, Yg), method=method)
    # Zg[np.isnan(Zg)] = griddata(np.c_[x, y], z, (Xg[np.isnan(Zg)], Yg[np.isnan(Zg)]), method="nearest")

    fig = go.Figure(data=[go.Surface(x=Xg, y=Yg, z=Zg, colorscale="Viridis", opacity=opacity)])
    fig.update_layout(
        title=title,
        # width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title="PCA-1 (init param)",
            yaxis_title="PCA-2 (init param)",
            zaxis_title="Loss / Eval",
            camera=dict(eye=dict(x=0.5, y=0.5, z=0.5)),
        ),
    )
    fig.show()

n_points = 3000
points = np.random.uniform(-np.pi, np.pi, (n_points, parameter_count))
pca = PCA(n_components=2)
points_2d = pca.fit_transform(points)
expec = []
pbar = tqdm(range(len(points)))
for i in pbar:
    expec.append(float(cudaq.observe(kernel_qaoa_use, H, points[i], *ansatz_fixed_param).expectation()))
points_3d = np.concatenate([points_2d, np.array(expec).reshape(-1, 1)], axis=1)
print(points_3d.shape)

exp_np = np.array(expec)

plot_px_surface(points_3d, title="Expectation Landscape", opacity=1, method="cubic")

# # Optimize

expectations = []

def cost_func(parameters):
    # return cudaq.observe(kernel_qaoa, H, n_qubit, layer_count, parameters, 0).expectation()
    return cudaq.observe(kernel_qaoa_use, H, parameters, *ansatz_fixed_param).expectation()

def objective(parameters):
    expectation = cost_func(parameters)
    expectations.append(expectation)
    return expectation

def objective_grad_cuda(parameters):
    expectation = cost_func(parameters)
    expectations.append(expectation)

    gradient = cudaq.gradients.ForwardDifference().compute(parameters, cost_func, expectation)

    return expectation, gradient

objective_func = objective_grad_cuda if FIND_GRAD else objective
print("Required Gradient = ", FIND_GRAD)

st = time.time()
optimal_expectation, optimal_parameters = optimizer.optimize(
    dimensions=parameter_count, function=objective_func)
et = time.time()

if not os.path.exists("./output_PO_mixer"):
    os.makedirs("./output_PO_mixer")
np.save(f"./output_PO_mixer/expectations_{optimizer_name}.npy", np.array(expectations))

print('optimal_expectation =', optimal_expectation)
print('optimal_parameters =', optimal_parameters)
print('Time taken = ', et - st)

shots_count = int(1e6)
print(f"Sampling {shots_count} times...")
# result = cudaq.sample(kernel_qaoa, int(n_qubit), layer_count, optimal_parameters, 0, shots_count=shots_count)
result = cudaq.sample(kernel_qaoa_use, optimal_parameters, *ansatz_fixed_param, shots_count=shots_count)

print("Finding the best solution...")
idx_b2 = result.most_probable()
idx = int(idx_b2, 2)
idx_r = 2**n_qubit - 1 - int(idx_b2, 2)
idx_r_b2 = bin(idx_r)[2:].zfill(n_qubit)

print(idx_b2, result[idx_b2], result[idx_b2]/shots_count)
print("|q0>|q1>|q2>...")

def state_to_return_(s, QU, lamb):
    l = np.array(list(map(int, s)))
    ss = l @ QU @ l.T
    return ss + lamb
print(state_to_return_("1111", QU, lamb))

state = cudaq.get_state(kernel_qaoa_use, optimal_parameters, *ansatz_fixed_param)

rows = []
col = ["State", "Probability", "Return"]
ret_sum = 0
state_best, return_best = "", 0
state_high, return_high = result.most_probable(), 0
for i in range(len(state)):
    bb = bin(i)[2:].zfill(n_qubit)
    ret = state_to_return(bb, QU, lamb)
    if ret > return_best:
        state_best, return_best = bb, ret
    if state_high == bb:
        return_high = ret
    prob = abs(state[i])**2
    al = np.array([bb, round(abs(state[i])**2, 4), round(ret, 4)])
    rows.append(al)
    ret_sum += ret * prob

df = pd.DataFrame(rows, columns=col)
print("Expected Return:", round(ret_sum, 4))

# colorr = ["blue" if in_bud == "True" else "red" for in_bud in df["In_Budget"]]
ex_ret = df["Return"].to_numpy()

print("Best state:", state_best, "Return:", return_best)
print("Most probable state:", state_high, "Return:", return_high)

print(np.array(state))

assert False

print(result)
print(np.abs(np.array(state))**2)

stt = cudaq.get_state(kernel_flipped, state, n_qubit)
print(np.abs(np.array(stt))**2)

result_final = np.zeros(2**n_qubit)
for i in result:
    result_final[int(i, 2)] = result[i]

# plt.figure(figsize=(100, 15))
plt.figure(figsize=(15, 5))
x = np.arange(2**n_qubit)

# plt.bar(range(2**qubit_count), list(result.values()))
plt.bar(range(2**n_qubit), result_final)
# plt.bar(range(int(2**(n_qubit-3)*0.1)), result_final[2**(n_qubit-3)*4:int(2**(n_qubit-3)*4.1)])
plt.ylabel('Frequency')
plt.title('Distribution of Preserving Mixer')
# plt.gca().set_xticklabels([])
# plt.xticks(rotation=90)
# plt.xticks(visible=False)
# plt.xticks(xlocs, xlabs)
plt.xticks(x, [f"{i:0{n_qubit}b}" for i in x])
xlocs, xlabs = plt.xticks()
for i, s in enumerate(ex_ret):
    plt.text(xlocs[i]-0.4, result_final[i]+result_final[int(state_high, 2)]/80, s)

plt.show()

# Exhaustive Search
#
# x: qubit (100 samples per qubit)
# y: approx ratio (best vs real best)

plt.figure(figsize=(20, 10))
for i in range(len(optimizer_names)):
    if os.path.exists(f"./output_PO_mixer/expectations_{optimizer_names[i]}.npy"):
        print(f"Loading expectations from {optimizer_names[i]}")
    else:
        print(f"Expectations file not found for {optimizer_names[i]}")
        continue
    expectations = np.load(f"./output_PO_mixer/expectations_{optimizer_names[i]}.npy")
    plt.plot(expectations, label=optimizer_names[i])
plt.xlabel('Iterations')
plt.ylabel('Expectation')
plt.title(f'Expectations vs Iterations ({n_qubit} qubits)')
plt.legend()
plt.show()

plt.figure(figsize=(20, 10))
for i in range(len(optimizer_names)):
    if os.path.exists(f"./output_PO_mixer/expectations_{optimizer_names[i]}.npy"):
        print(f"Loading expectations from {optimizer_names[i]}")
    else:
        print(f"Expectations file not found for {optimizer_names[i]}")
        continue
    expectations = np.load(f"./output_PO_mixer/expectations_{optimizer_names[i]}.npy")
    plt.plot(expectations, label=optimizer_names[i])
plt.xlabel('Iterations')
plt.ylabel('Expectation')
plt.title(f'Expectations vs Iterations ({n_qubit} qubits)')
plt.legend()
plt.show()
