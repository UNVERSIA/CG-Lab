import os
import math
import imageio
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from IPython.display import clear_output

from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
    look_at_view_transform,
)
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    mesh_edge_loss,
    mesh_normal_consistency,
)


OBJ_PATH = "cow.obj"
OUTPUT_DIR = "work06_outputs"

IMAGE_SIZE = 128
NUM_VIEWS = 20
EPOCHS = 300
SAVE_EVERY = 20

LR = 1.0
LAMBDA_LAPLACIAN = 1.0
LAMBDA_EDGE = 0.1
LAMBDA_NORMAL = 0.01


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()

    center = verts.mean(dim=0)
    verts = verts - center

    scale = verts.abs().max()
    verts = verts / scale

    verts = verts * 1.15

    return Meshes(
        verts=[verts],
        faces=[faces],
        textures=None,
    )


def build_cameras(device):
    elev = torch.linspace(10, 25, NUM_VIEWS, device=device)
    azim = torch.linspace(-180, 180, NUM_VIEWS, device=device)

    R, T = look_at_view_transform(
        dist=2.7,
        elev=elev,
        azim=azim,
        device=device,
    )

    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
    )

    return cameras


def build_silhouette_renderer(cameras, device):
    sigma = 1e-4

    blend_params = BlendParams(
        sigma=sigma,
        gamma=1e-4,
    )

    raster_settings = RasterizationSettings(
        image_size=IMAGE_SIZE,
        blur_radius=math.log(1.0 / sigma - 1.0) * sigma,
        faces_per_pixel=50,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftSilhouetteShader(
            blend_params=blend_params,
        ),
    )

    return renderer


def save_compare_figure(target_silhouette, pred_silhouette, epoch, loss, loss_silhouette):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

    ax[0].imshow(target_silhouette[0].detach().cpu().numpy(), cmap="gray")
    ax[0].set_title("Ground Truth Silhouette")
    ax[0].axis("off")

    ax[1].imshow(pred_silhouette[0].detach().cpu().numpy(), cmap="gray")
    ax[1].set_title(f"Optimizing... Epoch {epoch}")
    ax[1].axis("off")

    fig.suptitle(
        f"Epoch {epoch:03d} | Total Loss: {loss.item():.4f} | Silhouette Loss: {loss_silhouette.item():.4f}"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"compare_epoch_{epoch:03d}.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close(fig)

    return save_path


def make_gif(frame_paths, gif_path):
    frames = []
    for path in frame_paths:
        frames.append(imageio.v2.imread(path))

    imageio.mimsave(gif_path, frames, duration=0.45)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    if not os.path.exists(OBJ_PATH):
        raise FileNotFoundError(
            f"未找到 {OBJ_PATH}。请先将 cow.obj 上传到 main.py 同目录下。"
        )

    print("Loading target cow mesh...")
    target_mesh = load_objs_as_meshes([OBJ_PATH], device=device)
    target_mesh = normalize_mesh(target_mesh)

    print("Building cameras and renderer...")
    cameras = build_cameras(device)
    renderer = build_silhouette_renderer(cameras, device)

    with torch.no_grad():
        target_silhouette = renderer(
            target_mesh.extend(NUM_VIEWS),
            cameras=cameras,
        )[..., 3]

    print("Creating initial sphere mesh...")
    src_mesh = ico_sphere(4, device=device)
    src_mesh = normalize_mesh(src_mesh)

    deform_verts = torch.zeros_like(
        src_mesh.verts_packed(),
        device=device,
        requires_grad=True,
    )

    optimizer = torch.optim.SGD(
        [deform_verts],
        lr=LR,
        momentum=0.9,
    )

    frame_paths = []

    print("Start optimizing...")
    for epoch in tqdm(range(EPOCHS)):
        optimizer.zero_grad()

        new_src_mesh = src_mesh.offset_verts(deform_verts)

        pred_silhouette = renderer(
            new_src_mesh.extend(NUM_VIEWS),
            cameras=cameras,
        )[..., 3]

        loss_silhouette = ((pred_silhouette - target_silhouette) ** 2).mean()

        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh)
        loss_edge = mesh_edge_loss(new_src_mesh)
        loss_normal = mesh_normal_consistency(new_src_mesh)

        loss = (
            loss_silhouette
            + LAMBDA_LAPLACIAN * loss_laplacian
            + LAMBDA_EDGE * loss_edge
            + LAMBDA_NORMAL * loss_normal
        )

        loss.backward()
        optimizer.step()

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            clear_output(wait=True)

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Total Loss: {loss.item():.4f} | "
                f"Silhouette Loss: {loss_silhouette.item():.4f} | "
                f"Laplacian: {loss_laplacian.item():.4f} | "
                f"Edge: {loss_edge.item():.4f} | "
                f"Normal: {loss_normal.item():.4f}"
            )

            current_verts = new_src_mesh.verts_list()[0]
            current_faces = new_src_mesh.faces_list()[0]

            obj_path = os.path.join(OUTPUT_DIR, f"mesh_epoch_{epoch:03d}.obj")
            save_obj(obj_path, current_verts, current_faces)

            frame_path = save_compare_figure(
                target_silhouette,
                pred_silhouette,
                epoch,
                loss,
                loss_silhouette,
            )
            frame_paths.append(frame_path)

            print("[*] 当前 mesh 已保存到:", obj_path)
            print("[*] 当前对比图已保存到:", frame_path)

    final_mesh = src_mesh.offset_verts(deform_verts)
    final_verts = final_mesh.verts_list()[0]
    final_faces = final_mesh.faces_list()[0]

    final_obj_path = os.path.join(OUTPUT_DIR, "final_mesh.obj")
    save_obj(final_obj_path, final_verts, final_faces)

    gif_path = os.path.join(OUTPUT_DIR, "optimization.gif")
    make_gif(frame_paths, gif_path)

    print("优化完成。")
    print("最终模型:", final_obj_path)
    print("优化过程 GIF:", gif_path)
    print("可以把 optimization.gif 下载后放到 src/Work06/assets/demo.gif。")


if __name__ == "__main__":
    main()