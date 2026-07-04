# --- 物理系统参数 ---
# 你指定的随机颜色列表
COLOR_LIST = [0xFFEE99, 0xFFCCCC, 0xEEFFBB, 0x00BFFF, 0xFFFFF]  
NUM_PARTICLES = 10000      # 粒子总数 (卡顿可改2000)
GRAVITY_STRENGTH = 0.001   # 鼠标引力强度
DRAG_COEF = 0.98           # 空气阻力系数
BOUNCE_COEF = -0.8         # 边界反弹能量损耗

# --- 渲染系统参数 ---
WINDOW_RES = (800, 600)    # 窗口分辨率
PARTICLE_RADIUS = 1.5      # 粒子绘制半径
PARTICLE_COLOR = 0x00BFFF  # 粒子默认颜色 (天蓝色)
COLOR_COUNT = len(COLOR_LIST) # 颜色数量

# 提取颜色列表为独立常量（适配Taichi 1.7.4）
COLOR_0 = COLOR_LIST[0]  # 0xE0BFB8
COLOR_1 = COLOR_LIST[1]  # 0xB9A9C7
COLOR_2 = COLOR_LIST[2]  # 0xA3B899
COLOR_3 = COLOR_LIST[3]  # 0x00BFFF
COLOR_4 = COLOR_LIST[4]  # 0xD9C7B8