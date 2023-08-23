import taichi as ti
import math

ti.init(arch=ti.cpu)

# global control
paused = False
damping_toggle = ti.field(ti.i32, ()) # 阻尼
curser = ti.Vector.field(2, ti.f32, ())
picking = ti.field(ti.i32,())
using_auto_diff = False

# procedurally setting up the cantilever
init_x, init_y = 0.1, 0.6
N_x = 20
N_y = 4
# N_x = 2
# N_y = 2
N = N_x*N_y
N_edges = (N_x-1)*N_y + N_x*(N_y - 1) + (N_x-1) * \
    (N_y-1)  # horizontal + vertical + diagonal springs
N_triangles = 2 * (N_x-1) * (N_y-1)
dx = 1/32
curser_radius = dx/2

# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ()) # 杨氏模量
PoissonsRatio = ti.field(ti.f32, ()) # 泊松率
LameMu = ti.field(ti.f32, ())  # 拉梅常数mu
LameLa = ti.field(ti.f32, ())  # 拉梅常数lambda

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time integration)
dh = h/substepping

# simulation components
x = ti.Vector.field(2, ti.f32, N, needs_grad=True) # 顶点的位置, needs_grad=True表示将针对此字段计算与梯度有关的自动微分
v = ti.Vector.field(2, ti.f32, N) # 模拟中顶点的2D速度
total_energy = ti.field(ti.f32, (), needs_grad=True) # 标量字段，存储仿真的总能量
grad = ti.Vector.field(2, ti.f32, N) # 总能量相对于顶点位置x的梯度
elements_Dm_inv = ti.Matrix.field(2, 2, ti.f32, N_triangles) # 仿真中每个三角形元素的变形梯度的逆
elements_V0 = ti.field(ti.f32, N_triangles) # 存储仿真中每个三角形元素的初始（未变形）面积

# geometric components
triangles = ti.Vector.field(3, ti.i32, N_triangles) # 用于表示三角形元素的信息。每个三角形由三个顶点的索引构成，因此它是一个3D向量（ti.Vector）
edges = ti.Vector.field(2, ti.i32, N_edges) # 用于表示边的信息。每条边由两个顶点的索引构成，因此它是一个2D向量（ti.Vector） 


def ij_2_index(i, j): return i * N_y + j # 用于将二维坐标 (i, j) 映射到一个一维索引值，通常在一个二维数组或网格中使用。这种映射在处理多维数据时非常常见，可以将多维数据映射到线性内存中或者用于索引。


# -----------------------meshing and init----------------------------
@ti.kernel
def meshing(): # 用于设置三角形和边的信息，形成一个网格结构
    # setting up triangles
    for i,j in ti.ndrange(N_x - 1, N_y - 1): # 这个嵌套循环迭代网格的每个单元（格子）。这是为了设置三角形的信息。N_x 和 N_y 是网格的维度。
        # triangle id 生成三角形：通过计算出每个三角形的顶点索引，填充 triangles 字段。
        tid = (i * (N_y - 1) + j) * 2 # tid 是三角形的 ID，计算方式保证了不同格子生成不同的 ID
        triangles[tid][0] = ij_2_index(i, j)
        triangles[tid][1] = ij_2_index(i + 1, j)
        triangles[tid][2] = ij_2_index(i, j + 1)

        tid = (i * (N_y - 1) + j) * 2 + 1
        triangles[tid][0] = ij_2_index(i, j + 1)
        triangles[tid][1] = ij_2_index(i + 1, j + 1)
        triangles[tid][2] = ij_2_index(i + 1, j)
        # 对于每个格子 (i, j)，有两个三角形。第一个三角形的顶点是 (i, j)，(i+1, j) 和 (i, j+1)。第二个三角形的顶点是 (i, j+1)，(i+1, j+1) 和 (i+1, j)。这种方式下，每个格子贡献了两个三角形。

    # setting up edges
    # edge id
    eid_base = 0
    #通过计算出每条边的顶点索引，填充 edges 字段
    # horizontal edges 生成水平边：每个格子产生一条连接 (i, j) 和 (i+1, j) 的边
    for i in range(N_x-1):
        for j in range(N_y):
            eid = eid_base+i*N_y+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i+1, j)]

    eid_base += (N_x-1)*N_y
    # vertical edges 生成垂直边：每个格子产生一条连接 (i, j) 和 (i, j+1) 的边
    for i in range(N_x):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i, j+1)]

    eid_base += N_x*(N_y-1)
    # diagonal edges 生成对角边：每个格子产生一条连接 (i+1, j) 和 (i, j+1) 的边
    for i in range(N_x-1):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i+1, j), ij_2_index(i, j+1)]

@ti.kernel
def initialize():
    YoungsModulus[None] = 1e6 # 存储材料的杨氏模量
    paused = True # 允许暂停
    # init position and velocity
    for i, j in ti.ndrange(N_x, N_y): # 使用 ti.ndrange 迭代每个格点 (i, j)，为每个格点初始化位置和速度：
        index = ij_2_index(i, j) #  将 (i, j) 坐标映射为一维索引。
        x[index] = ti.Vector([init_x + i * dx, init_y + j * dx]) # 设置位置为 [init_x + i * dx, init_y + j * dx]，其中 init_x 和 init_y 是初始坐标，dx 是间隔
        v[index] = ti.Vector([0.0, 0.0]) # 初始化速度为 [0.0, 0.0]

@ti.func
def compute_D(i): # 计算给定三角形的变形梯度矩阵 D
    a = triangles[i][0] # a, b, c 是三角形的顶点索引，triangles[i] 表示第 i 个三角形的顶点索引。
    b = triangles[i][1]
    c = triangles[i][2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]]) # 返回一个由两个列向量组成的矩阵，表示变形梯度。

@ti.kernel
def initialize_elements():
    for i in range(N_triangles): # 迭代每个三角形，计算变形梯度矩阵 Dm，然后存储它的逆矩阵到 elements_Dm_inv 字段，同时计算初始面积并存储到 elements_V0 字段。
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()
        elements_V0[i] = ti.abs(Dm.determinant())/2

# ----------------------core-----------------------------
@ti.func
def compute_R_2D(F): # 函数接受一个参数 F，这是一个二维变形梯度矩阵
    R, S = ti.polar_decompose(F, ti.f32) # 用于执行极分解（极分解将变形梯度矩阵拆分为旋转矩阵和缩放矩阵的乘积）。第一个返回值 R 是极分解中的旋转矩阵，而第二个返回值 S 是缩放矩阵。
    return R

@ti.kernel
def compute_gradient():
    # clear gradient
    for i in grad:
        grad[i] = ti.Vector([0, 0]) # 清除梯度：通过迭代 grad 字段中的每个元素，将梯度初始化为零向量

    # gradient of elastic potential
    for i in range(N_triangles): # 弹性势能的梯度：迭代每个三角形，计算相关的梯度。
        Ds = compute_D(i) # Ds：计算当前三角形的变形梯度矩阵。
        F = Ds@elements_Dm_inv[i] # F：计算变形梯度矩阵 F，使用 Ds 与 elements_Dm_inv[i] 相乘得到。
        # co-rotated linear elasticity
        R = compute_R_2D(F) # R：使用之前定义的 compute_R_2D 函数计算变形梯度矩阵 F 的极分解中的旋转矩阵
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        # first Piola-Kirchhoff tensor
        P = 2*LameMu[None]*(F-R) + LameLa[None]*((R.transpose())@F-Eye).trace()*R # P：计算一阶 Piola-Kirchhoff 应力张量，根据线性弹性模型的公式计算。
        #assemble to gradient
        H = elements_V0[i] * P @ (elements_Dm_inv[i].transpose()) # H：计算对能量密度的梯度乘以逆变形梯度矩阵的转置，以组装梯度。 
        a,b,c = triangles[i][0],triangles[i][1],triangles[i][2]
        gb = ti.Vector([H[0,0], H[1, 0]])
        gc = ti.Vector([H[0,1], H[1, 1]])
        ga = -gb-gc
        grad[a] += ga
        grad[b] += gb
        grad[c] += gc     
        # 将梯度组装到 grad 字段中：根据三角形的三个顶点 a, b, c，计算每个顶点的梯度，并将其添加到 grad 字段中对应的位置。

@ti.kernel
def compute_total_energy():
    for i in range(N_triangles): # 通过迭代每个三角形，计算其能量密度，根据线性弹性模型的公式计算
        Ds = compute_D(i)
        F = Ds @ elements_Dm_inv[i]
        # co-rotated linear elasticity
        R = compute_R_2D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        element_energy_density = LameMu[None]*((F-R)@(F-R).transpose()).trace() + 0.5*LameLa[None]*(R.transpose()@F-Eye).trace()**2

        total_energy[None] += element_energy_density * elements_V0[i] # 将每个三角形的能量密度乘以其初始面积，并累积到 total_energy 字段中，以计算整个模拟的总能量。

@ti.kernel
def update(): # 对每个顶点进行时间积分，以更新顶点的位置和速度。
    # perform time integration
    for i in range(N):
        # symplectic integration
        # elastic force + gravitation force, divding mass to get the acceleration
        if using_auto_diff:
            acc = -x.grad[i]/m - ti.Vector([0.0, g])
            v[i] += dh*acc
        else:
            acc = -grad[i]/m - ti.Vector([0.0, g])
            v[i] += dh*acc
        x[i] += dh*v[i]

    # explicit damping (ether drag)
    for i in v:
        if damping_toggle[None]:
            v[i] *= ti.exp(-dh*5) # 实现了显式阻尼，使速度指数级减小。

    # enforce boundary condition 强制实施边界条件，例如在鼠标光标附近固定顶点，或将顶点与墙连接
    for i in range(N):
        if picking[None]:           
            r = x[i]-curser[None]
            if r.norm() < curser_radius:
                x[i] = curser[None]
                v[i] = ti.Vector([0.0, 0.0])
                pass

    for j in range(N_y):
        ind = ij_2_index(0, j)
        v[ind] = ti.Vector([0, 0])
        x[ind] = ti.Vector([init_x, init_y + j * dx])  # rest pose attached to the wall

    for i in range(N):
        if x[i][0] < init_x:
            x[i][0] = init_x
            v[i][0] = 0


@ti.kernel
def updateLameCoeff(): # 这个函数用于更新 Lame 系数，它们与材料的弹性性质相关
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu)) # 基于 Young's 模量和泊松比，计算 Lame 系数

# init once and for all
meshing()
initialize()
initialize_elements()
updateLameCoeff()

gui = ti.GUI('Linear FEM', (800, 800))
while gui.running:

    picking[None]=0

    # key events
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'r':
            initialize()
        elif e.key == '0':
            YoungsModulus[None] *= 1.1
        elif e.key == '9':
            YoungsModulus[None] /= 1.1
            if YoungsModulus[None] <= 0:
                YoungsModulus[None] = 0
        elif e.key == '8':
            PoissonsRatio[None] = PoissonsRatio[None]*0.9+0.05 # slowly converge to 0.5
            if PoissonsRatio[None] >= 0.499:
                PoissonsRatio[None] = 0.499
        elif e.key == '7':
            PoissonsRatio[None] = PoissonsRatio[None]*1.1-0.05
            if PoissonsRatio[None] <= 0:
                PoissonsRatio[None] = 0
        elif e.key == ti.GUI.SPACE:
            paused = not paused
        elif e.key =='d' or e.key == 'D':
            damping_toggle[None] = not damping_toggle[None]
        elif e.key =='p' or e.key == 'P': # step-forward
            for i in range(substepping):
                if using_auto_diff:
                    total_energy[None]=0
                    with ti.Tape(total_energy):
                        compute_total_energy()
                else:
                    compute_gradient()
                update()
        updateLameCoeff()

    if gui.is_pressed(ti.GUI.LMB):
        curser[None] = gui.get_cursor_pos()
        picking[None] = 1

    # numerical time integration
    if not paused:
        for i in range(substepping):
            if using_auto_diff:
                total_energy[None]=0
                with ti.Tape(total_energy):
                    compute_total_energy()
            else:
                compute_gradient()
            update()

    # render
    pos = x.to_numpy()
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        gui.line((pos[a][0], pos[a][1]),
                 (pos[b][0], pos[b][1]),
                 radius=1,
                 color=0xFFFF00)
    gui.line((init_x, 0.0), (init_x, 1.0), color=0xFFFFFF, radius=4)

    if picking[None]:
        gui.circle((curser[None][0], curser[None][1]), radius=curser_radius*800, color=0xFF8888)

    # text
    gui.text(
        content=f'9/0: (-/+) Young\'s Modulus {YoungsModulus[None]:.1f}', pos=(0.6, 0.9), color=0xFFFFFF)
    gui.text(
        content=f'7/8: (-/+) Poisson\'s Ratio {PoissonsRatio[None]:.3f}', pos=(0.6, 0.875), color=0xFFFFFF)
    if damping_toggle[None]:
        gui.text(
            content='D: Damping On', pos=(0.6, 0.85), color=0xFFFFFF)
    else:
        gui.text(
            content='D: Damping Off', pos=(0.6, 0.85), color=0xFFFFFF)
    gui.show()