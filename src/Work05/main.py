import taichi as ti

ti.init(arch=ti.cpu)

RES_X = 800
RES_Y = 600
MAX_BOUNCES_LIMIT = 5
EPS = 1e-4

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

light_x = ti.field(ti.f32, shape=())
light_y = ti.field(ti.f32, shape=())
light_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
mirror_strength = ti.field(ti.f32, shape=())

MAT_DIFFUSE = 0
MAT_MIRROR = 1


@ti.func
def normalize(v):
    return v / ti.sqrt(v.dot(v) + 1e-8)


@ti.func
def reflect(direction, normal):
    return direction - 2.0 * direction.dot(normal) * normal


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
        t1 = (-b - sqrt_delta) * 0.5
        t2 = (-b + sqrt_delta) * 0.5

        if t1 > EPS:
            t = t1
        elif t2 > EPS:
            t = t2

        if t > 0.0:
            p = ro + rd * t
            normal = normalize(p - center)

    return t, normal


@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])

    if ti.abs(rd[1]) > 1e-6:
        temp_t = (plane_y - ro[1]) / rd[1]
        if temp_t > EPS:
            t = temp_t

    return t, normal


@ti.func
def checker_color(p):
    grid_scale = 2.0

    ix = ti.cast(ti.floor(p[0] * grid_scale), ti.i32)
    iz = ti.cast(ti.floor(p[2] * grid_scale), ti.i32)

    color = ti.Vector([0.78, 0.78, 0.78])
    if (ix + iz) % 2 == 0:
        color = ti.Vector([0.18, 0.18, 0.18])

    return color


@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    hit_color = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    t_red, n_red = intersect_sphere(
        ro,
        rd,
        ti.Vector([-1.25, 0.0, 0.0]),
        1.0
    )

    if 0.0 < t_red < min_t:
        min_t = t_red
        hit_normal = n_red
        hit_color = ti.Vector([0.85, 0.10, 0.08])
        hit_mat = MAT_DIFFUSE

    t_mirror, n_mirror = intersect_sphere(
        ro,
        rd,
        ti.Vector([1.15, 0.0, -0.15]),
        1.0
    )

    if 0.0 < t_mirror < min_t:
        min_t = t_mirror
        hit_normal = n_mirror
        hit_color = ti.Vector([0.95, 0.95, 0.95])
        hit_mat = MAT_MIRROR

    t_ground, n_ground = intersect_plane(ro, rd, -1.0)

    if 0.0 < t_ground < min_t:
        p = ro + rd * t_ground

        min_t = t_ground
        hit_normal = n_ground
        hit_color = checker_color(p)
        hit_mat = MAT_DIFFUSE

    return min_t, hit_normal, hit_color, hit_mat


@ti.func
def sky_color(rd):
    t = 0.5 * (rd[1] + 1.0)

    bottom = ti.Vector([0.05, 0.13, 0.16])
    top = ti.Vector([0.12, 0.22, 0.28])

    return bottom * (1.0 - t) + top * t


@ti.func
def shade_diffuse(point, normal, base_color, light_pos):
    n = normalize(normal)
    l = normalize(light_pos - point)

    ambient = 0.18 * base_color
    direct = ti.Vector([0.0, 0.0, 0.0])

    shadow_origin = point + n * EPS
    shadow_t, _, _, _ = scene_intersect(shadow_origin, l)
    dist_to_light = (light_pos - point).norm()

    in_shadow = 0
    if shadow_t > 0.0 and shadow_t < dist_to_light:
        in_shadow = 1

    if in_shadow == 0:
        diff = ti.max(0.0, n.dot(l))
        direct = 0.82 * diff * base_color

    return ambient + direct


@ti.kernel
def render():
    light_pos = ti.Vector([light_x[None], light_y[None], light_z[None]])

    for i, j in pixels:
        u = (i - RES_X * 0.5) / RES_Y * 2.0
        v = (j - RES_Y * 0.5) / RES_Y * 2.0

        ro = ti.Vector([0.0, 1.15, 5.2])
        rd = normalize(ti.Vector([u, v - 0.35, -1.45]))

        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        active = 1

        for bounce in range(MAX_BOUNCES_LIMIT):
            if active == 1 and bounce < max_bounces[None]:
                t, normal, obj_color, mat_id = scene_intersect(ro, rd)

                if t > 1e9:
                    final_color += throughput * sky_color(rd)
                    active = 0

                else:
                    point = ro + rd * t

                    if mat_id == MAT_MIRROR:
                        ro = point + normal * EPS
                        rd = normalize(reflect(rd, normal))
                        throughput *= mirror_strength[None] * obj_color

                    else:
                        color = shade_diffuse(point, normal, obj_color, light_pos)
                        final_color += throughput * color
                        active = 0

        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)


def reset_parameters():
    light_x[None] = 2.0
    light_y[None] = 4.0
    light_z[None] = 3.0
    max_bounces[None] = 3
    mirror_strength[None] = 0.82


def main():
    reset_parameters()

    window = ti.ui.Window("Work05 - Ray Tracing Demo", (RES_X, RES_Y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    print("操作说明：")
    print("拖动 UI 滑块：调节光源位置、反射次数和镜面反射强度")
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

        with gui.sub_window("Ray Tracing Controls", 0.70, 0.04, 0.28, 0.28):
            light_x[None] = gui.slider_float("Light X", light_x[None], -5.0, 5.0)
            light_y[None] = gui.slider_float("Light Y", light_y[None], 1.0, 8.0)
            light_z[None] = gui.slider_float("Light Z", light_z[None], -5.0, 6.0)
            max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)
            mirror_strength[None] = gui.slider_float("Mirror", mirror_strength[None], 0.1, 1.0)

        window.show()


if __name__ == "__main__":
    main()