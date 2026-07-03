import taichi as ti

try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)

RES_X = 800
RES_Y = 600

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

light_x = ti.field(ti.f32, shape=())
light_y = ti.field(ti.f32, shape=())
light_z = ti.field(ti.f32, shape=())


@ti.func
def normalize(v):
    return v / ti.sqrt(v.dot(v) + 1e-8)


@ti.func
def reflect(in_dir, normal):
    return in_dir - 2.0 * in_dir.dot(normal) * normal


@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])

    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c

    if delta > 0.0:
        sqrt_delta = ti.sqrt(delta)
        t1 = (-b - sqrt_delta) / 2.0
        t2 = (-b + sqrt_delta) / 2.0

        if t1 > 0.0:
            t = t1
        elif t2 > 0.0:
            t = t2

        if t > 0.0:
            p = ro + rd * t
            normal = normalize(p - center)

    return t, normal


@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])

    height = apex[1] - base_y
    k = (radius / height) * (radius / height)

    ro_local = ro - apex

    a = rd[0] * rd[0] + rd[2] * rd[2] - k * rd[1] * rd[1]
    b = 2.0 * (
        ro_local[0] * rd[0]
        + ro_local[2] * rd[2]
        - k * ro_local[1] * rd[1]
    )
    c = ro_local[0] * ro_local[0] + ro_local[2] * ro_local[2] - k * ro_local[1] * ro_local[1]

    if ti.abs(a) > 1e-6:
        delta = b * b - 4.0 * a * c

        if delta > 0.0:
            sqrt_delta = ti.sqrt(delta)
            t1 = (-b - sqrt_delta) / (2.0 * a)
            t2 = (-b + sqrt_delta) / (2.0 * a)

            if t1 > t2:
                temp = t1
                t1 = t2
                t2 = temp

            y1 = ro_local[1] + t1 * rd[1]
            y2 = ro_local[1] + t2 * rd[1]

            if t1 > 0.0 and -height <= y1 <= 0.0:
                t = t1
            elif t2 > 0.0 and -height <= y2 <= 0.0:
                t = t2

            if t > 0.0:
                p_local = ro_local + rd * t
                normal = normalize(
                    ti.Vector([
                        p_local[0],
                        -k * p_local[1],
                        p_local[2],
                    ])
                )

    return t, normal


@ti.func
def phong_shading(point, normal, view_pos, base_color):
    light_pos = ti.Vector([light_x[None], light_y[None], light_z[None]])
    light_color = ti.Vector([1.0, 1.0, 1.0])

    n = normalize(normal)
    l = normalize(light_pos - point)
    v = normalize(view_pos - point)

    ambient = Ka[None] * base_color

    ndotl = ti.max(0.0, n.dot(l))
    diffuse = Kd[None] * ndotl * base_color * light_color

    specular = ti.Vector([0.0, 0.0, 0.0])
    if ndotl > 0.0:
        r = normalize(reflect(-l, n))
        spec = ti.max(0.0, r.dot(v)) ** shininess[None]
        specular = Ks[None] * spec * light_color

    return ambient + diffuse + specular


@ti.kernel
def render():
    for i, j in pixels:
        u = (i - RES_X * 0.5) / RES_Y * 2.0
        v = (j - RES_Y * 0.5) / RES_Y * 2.0

        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.35]))

        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])

        sphere_center = ti.Vector([-1.15, -0.15, 0.0])
        t_sphere, n_sphere = intersect_sphere(ro, rd, sphere_center, 1.05)

        if 0.0 < t_sphere < min_t:
            min_t = t_sphere
            hit_normal = n_sphere
            hit_color = ti.Vector([0.82, 0.16, 0.10])

        cone_apex = ti.Vector([1.20, 1.15, 0.15])
        t_cone, n_cone = intersect_cone(ro, rd, cone_apex, -1.25, 0.95)

        if 0.0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.55, 0.18, 0.85])

        bg_t = j / RES_Y
        color = ti.Vector([
            0.025 + 0.025 * bg_t,
            0.075 + 0.035 * bg_t,
            0.085 + 0.045 * bg_t,
        ])

        if min_t < 1e9:
            hit_point = ro + rd * min_t
            color = phong_shading(hit_point, hit_normal, ro, hit_color)

        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)


def reset_parameters():
    Ka[None] = 0.22
    Kd[None] = 0.72
    Ks[None] = 0.52
    shininess[None] = 36.0

    light_x[None] = 2.2
    light_y[None] = 3.2
    light_z[None] = 4.0


def main():
    reset_parameters()

    window = ti.ui.Window("Work04 - Phong Shading Demo", (RES_X, RES_Y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    print("操作说明：")
    print("拖动右上角 UI 滑块：调节材质和光源参数")
    print("R：恢复默认参数")
    print("Esc：退出程序")

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key in ["r", "R"]:
                reset_parameters()

        render()
        canvas.set_image(pixels)

        with gui.sub_window("Material and Light", 0.66, 0.04, 0.32, 0.42):
            Ka[None] = gui.slider_float("Ka Ambient", Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float("Kd Diffuse", Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float("Ks Specular", Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)

            light_x[None] = gui.slider_float("Light X", light_x[None], -4.0, 4.0)
            light_y[None] = gui.slider_float("Light Y", light_y[None], -1.0, 5.0)
            light_z[None] = gui.slider_float("Light Z", light_z[None], 1.0, 8.0)

        window.show()


if __name__ == "__main__":
    main()