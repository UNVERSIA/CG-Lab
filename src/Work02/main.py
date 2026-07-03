import taichi as ti

ti.init(arch=ti.cpu)

PI = 3.141592653589793

vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)


@ti.func
def get_model_matrix(angle: ti.f32, scale: ti.f32):
    rad = angle * PI / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)

    return ti.Matrix([
        [scale * c, -scale * s, 0.0, 0.0],
        [scale * s,  scale * c, 0.0, 0.0],
        [0.0,        0.0,       scale, 0.0],
        [0.0,        0.0,       0.0,   1.0],
    ])


@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])


@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32,
                          z_near: ti.f32, z_far: ti.f32):
    n = -z_near
    f = -z_far

    fov_rad = eye_fov * PI / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    persp_to_ortho = ti.Matrix([
        [n,   0.0, 0.0,   0.0],
        [0.0, n,   0.0,   0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0,   0.0],
    ])

    ortho_translate = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0,           0.0,           0.0],
        [0.0,           2.0 / (t - b), 0.0,           0.0],
        [0.0,           0.0,           2.0 / (n - f), 0.0],
        [0.0,           0.0,           0.0,           1.0],
    ])

    return ortho_scale @ ortho_translate @ persp_to_ortho


@ti.kernel
def compute_transform(angle: ti.f32, scale: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.5])

    model = get_model_matrix(angle, scale)
    view = get_view_matrix(eye_pos)
    projection = get_projection_matrix(50.0, 1.0, 0.1, 80.0)

    mvp = projection @ view @ model

    for i in range(3):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])

        clip = mvp @ v4
        ndc = clip / clip[3]

        screen_coords[i][0] = (ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (ndc[1] + 1.0) / 2.0


def draw_triangle(gui):
    p = screen_coords.to_numpy()

    a = (p[0][0], p[0][1])
    b = (p[1][0], p[1][1])
    c = (p[2][0], p[2][1])

    gui.line(a, b, radius=3, color=0x3388FF)
    gui.line(b, c, radius=3, color=0xFF5555)
    gui.line(c, a, radius=3, color=0x66DD66)


def main():
    vertices[0] = [-1.8, -1.0, -2.0]
    vertices[1] = [1.7, -0.8, -2.0]
    vertices[2] = [0.0, 1.6, -2.0]

    gui = ti.GUI(
        "Work02 - 3D Transformation",
        res=(700, 700),
        background_color=0x0B1020,
    )

    angle = 0.0
    scale = 1.0
    auto_rotate = False

    while gui.running:
        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.ESCAPE:
                gui.running = False

            elif event.key in ["a", "A"]:
                angle += 8.0

            elif event.key in ["d", "D"]:
                angle -= 8.0

            elif event.key in ["w", "W"]:
                scale += 0.08
                if scale > 1.8:
                    scale = 1.8

            elif event.key in ["s", "S"]:
                scale -= 0.08
                if scale < 0.5:
                    scale = 0.5

            elif event.key in ["m", "M"]:
                auto_rotate = not auto_rotate

            elif event.key in ["r", "R"]:
                angle = 0.0
                scale = 1.0
                auto_rotate = False

        if auto_rotate:
            angle += 1.0

        compute_transform(angle, scale)
        draw_triangle(gui)

        gui.text(
            f"A/D: rotate | W/S: scale | M: auto rotate | R: reset | angle={angle:.1f}, scale={scale:.2f}",
            pos=(0.03, 0.95),
            color=0xFFFFFF,
        )

        gui.show()


if __name__ == "__main__":
    main()