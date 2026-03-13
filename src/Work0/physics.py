import taichi as ti
from .config import *

pos = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
color = ti.field(dtype=int, shape=NUM_PARTICLES)  

@ti.kernel
def init_particles():
    """初始化每一个粒子的随机坐标和随机颜色"""
    for i in range(NUM_PARTICLES):
        pos[i] = [ti.random(), ti.random()]
        vel[i] = [0.0, 0.0]
    
        rand_idx = ti.cast(ti.random() * COLOR_COUNT, ti.int32)
        
        if rand_idx == 0:
            color[i] = COLOR_0
        elif rand_idx == 1:
            color[i] = COLOR_1
        elif rand_idx == 2:
            color[i] = COLOR_2
        elif rand_idx == 3:
            color[i] = COLOR_3
        else: 
            color[i] = COLOR_4

@ti.kernel
def update_particles(mouse_x: float, mouse_y: float):
    """物理更新：由 GPU 并行执行"""
    for i in range(NUM_PARTICLES):
        # 计算方向与距离
        mouse_pos = ti.Vector([mouse_x, mouse_y])
        dir = mouse_pos - pos[i]
        dist = dir.norm()
        
        # 施加引力与阻力
        if dist > 0.05:
            vel[i] += dir.normalized() * GRAVITY_STRENGTH
            
        vel[i] *= DRAG_COEF  
        pos[i] += vel[i]

        # 边框碰撞检测
        for j in ti.static(range(2)):
            if pos[i][j] < 0:
                pos[i][j] = 0.0
                vel[i][j] *= BOUNCE_COEF
            elif pos[i][j] > 1:
                pos[i][j] = 1.0
                vel[i][j] *= BOUNCE_COEF