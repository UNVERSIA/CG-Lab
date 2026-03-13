import taichi as ti
import sys
from pathlib import Path 

sys.path.append(str(Path(__file__).parent))
ti.init(arch=ti.gpu)  # 初始化GPU 最前

from .config import WINDOW_RES, PARTICLE_RADIUS
from .physics import init_particles, update_particles, pos, color

def run():
    print("正在编译 GPU 内核，请稍候...")
    init_particles()  # 初始化粒子位置和颜色
    
    # 创建窗口
    gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES)
    print("编译完成！请在弹出的窗口中移动鼠标。")
    
    # 渲染主循环
    while gui.running:
        mouse_x, mouse_y = gui.get_cursor_pos()  
        
        update_particles(mouse_x, mouse_y)
        
        gui.circles(pos.to_numpy(), color=color.to_numpy(), radius=PARTICLE_RADIUS)
        gui.show()  # 刷新窗口

if __name__ == "__main__":
    run()