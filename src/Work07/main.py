import taichi as ti

try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)

N = 20
NUM_PARTICLES = N * N
MAX_SPRINGS = N * N * 4

MASS = 1.0
DT = 5e-4
K_S = 10000.0
MAX_VELOCITY = 50.0

gravity = ti.Vector([0.0, -9.8, 0.0])

x = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
f = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
is_fixed = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)

x_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
f_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

spring_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_SPRINGS)
spring_lengths = ti.field(dtype=ti.f32, shape=MAX_SPRINGS)
spring_indices = ti.field(dtype=ti.i32, shape=MAX_SPRINGS * 2)
num_springs = ti.field(dtype=ti.i32, shape=())

damping = ti.field(dtype=ti.f32, shape=())


@ti.func
def pid(i, j):
    return i * N + j


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = pid(i, j)

        x[idx] = ti.Vector([
            i * 0.05 - 0.5,
            0.8,
            j * 0.05 - 0.5,
        ])

        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])

        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0


@ti.kernel
def init_springs():
    num_springs[None] = 0

    for i, j in ti.ndrange(N, N):
        idx = pid(i, j)

        if i < N - 1:
            idx_right = pid(i + 1, j)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()

        if j < N - 1:
            idx_down = pid(i, j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()


@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]


def init_cloth():
    init_positions()
    init_springs()
    init_spring_indices()


@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    for i in range(NUM_PARTICLES):
        force[i] = gravity * MASS - damping[None] * vel[i]

    for s in range(num_springs[None]):
        a = spring_pairs[s][0]
        b = spring_pairs[s][1]

        pa = pos[a]
        pb = pos[b]

        va = vel[a]
        vb = vel[b]

        direction = pa - pb
        length = direction.norm()

        if length > 1e-6:
            n = direction / length
            stretch = length - spring_lengths[s]

            spring_force = -K_S * stretch * n

            relative_v = va - vb
            spring_damping = -0.2 * damping[None] * relative_v.dot(n) * n

            total_force = spring_force + spring_damping

            ti.atomic_add(force[a], total_force)
            ti.atomic_add(force[b], -total_force)


@ti.func
def clamp_velocity(vel: ti.template(), idx: ti.i32):
    speed = vel[idx].norm()

    if speed > MAX_VELOCITY:
        vel[idx] = vel[idx] / speed * MAX_VELOCITY


@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            x[i] += v[i] * DT
            v[i] += f[i] / MASS * DT
            clamp_velocity(v, i)
        else:
            v[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            v[i] += f[i] / MASS * DT
            clamp_velocity(v, i)
            x[i] += v[i] * DT
        else:
            v[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def step_implicit():
    for i in range(NUM_PARTICLES):
        x_next[i] = x[i]
        v_next[i] = v[i]

    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)

        for i in range(NUM_PARTICLES):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + f_next[i] / MASS * DT
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * DT
            else:
                v_next[i] = ti.Vector([0.0, 0.0, 0.0])
                x_next[i] = x[i]

    for i in range(NUM_PARTICLES):
        x[i] = x_next[i]
        v[i] = v_next[i]


def main():
    damping[None] = 1.0
    init_cloth()

    window = ti.ui.Window("Games101 - Mass Spring System", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1
    paused = False

    while window.running:
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.38)

        window.GUI.text("Integration Method:")

        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth()

        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()

        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth()

        window.GUI.text("")

        window.GUI.text("Damping:")
        window.GUI.text(f"Current: {damping[None]:.1f}")

        if window.GUI.button("Set Damping = 1.0"):
            damping[None] = 1.0
            init_cloth()

        if window.GUI.button("Set Damping = 5.0"):
            damping[None] = 5.0
            init_cloth()

        window.GUI.text("")

        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused

        if window.GUI.button("Reset Cloth"):
            init_cloth()

        window.GUI.end()

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                else:
                    step_implicit()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))

        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()