import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
BEZIER_SAMPLES = 1000
BSPLINE_SAMPLES_PER_SEGMENT = 80
MAX_CURVE_POINTS = 12000

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)


@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def draw_curve_pixels(n: ti.i32, mode: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x = ti.cast(pt[0] * (WIDTH - 1), ti.i32)
        y = ti.cast(pt[1] * (HEIGHT - 1), ti.i32)

        color = ti.Vector([0.0, 0.85, 1.0])
        if mode == 1:
            color = ti.Vector([1.0, 0.72, 0.12])

        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    pixels[nx, ny] = color


def de_casteljau(points, t):
    temp = np.array(points, dtype=np.float32)
    count = len(temp)

    while count > 1:
        for i in range(count - 1):
            temp[i] = (1.0 - t) * temp[i] + t * temp[i + 1]
        count -= 1

    return temp[0]


def build_bezier_curve(points):
    curve = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32)

    if len(points) < 2:
        return curve, 0

    count = min(BEZIER_SAMPLES + 1, MAX_CURVE_POINTS)

    for i in range(count):
        t = i / (count - 1)
        curve[i] = de_casteljau(points, t)

    return curve, count


def cubic_bspline_point(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t

    b0 = (-t3 + 3 * t2 - 3 * t + 1) / 6.0
    b1 = (3 * t3 - 6 * t2 + 4) / 6.0
    b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
    b3 = t3 / 6.0

    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def build_bspline_curve(points):
    curve = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32)

    if len(points) < 4:
        return curve, 0

    pts = np.array(points, dtype=np.float32)
    count = 0

    for start in range(len(pts) - 3):
        p0 = pts[start]
        p1 = pts[start + 1]
        p2 = pts[start + 2]
        p3 = pts[start + 3]

        for j in range(BSPLINE_SAMPLES_PER_SEGMENT + 1):
            if count >= MAX_CURVE_POINTS:
                return curve, count

            t = j / BSPLINE_SAMPLES_PER_SEGMENT
            curve[count] = cubic_bspline_point(p0, p1, p2, p3, t)
            count += 1

    return curve, count


def upload_control_points(control_points):
    point_data = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    current_count = len(control_points)

    if current_count > 0:
        point_data[:current_count] = np.array(control_points, dtype=np.float32)

    gui_points.from_numpy(point_data)

    index_data = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
    indices = []

    for i in range(current_count - 1):
        indices.extend([i, i + 1])

    if len(indices) > 0:
        index_data[:len(indices)] = np.array(indices, dtype=np.int32)

    gui_indices.from_numpy(index_data)


def main():
    window = ti.ui.Window(
        "Work03 - Bezier and B-Spline Curve",
        (WIDTH, HEIGHT)
    )
    canvas = window.get_canvas()

    control_points = []
    use_bspline = False

    print("操作说明：")
    print("鼠标左键：添加控制点")
    print("B：切换 Bezier / B-Spline")
    print("C：清空控制点")
    print("Z：撤销上一个控制点")
    print("Esc：退出程序")

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False

            elif event.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"添加控制点：{pos}")

            elif event.key in ["c", "C"]:
                control_points.clear()
                print("已清空控制点")

            elif event.key in ["z", "Z"]:
                if len(control_points) > 0:
                    control_points.pop()
                    print("已撤销上一个控制点")

            elif event.key in ["b", "B"]:
                use_bspline = not use_bspline
                mode_name = "B-Spline" if use_bspline else "Bezier"
                print(f"当前模式：{mode_name}")

        clear_pixels()

        current_count = len(control_points)

        if use_bspline:
            curve_points, curve_count = build_bspline_curve(control_points)
        else:
            curve_points, curve_count = build_bezier_curve(control_points)

        if curve_count > 0:
            curve_points_field.from_numpy(curve_points)
            draw_curve_pixels(curve_count, 1 if use_bspline else 0)

        canvas.set_image(pixels)

        if current_count > 0:
            upload_control_points(control_points)

            canvas.circles(
                gui_points,
                radius=0.007,
                color=(1.0, 0.15, 0.15)
            )

            if current_count >= 2:
                canvas.lines(
                    gui_points,
                    width=0.002,
                    indices=gui_indices,
                    color=(0.55, 0.55, 0.55)
                )

        window.show()


if __name__ == "__main__":
    main()