import time 
import math
import copy
import torch
import matplotlib.pyplot as plt
from diff_gaussian_renderer import DiffGaussRenderer
from tqdm import tqdm
from utils import readCamerasFromTransforms
from PIL import Image
import torchvision.transforms as T

def image_path_to_tensor(image_path):
    img = Image.open(image_path)
    transform = T.ToTensor()
    img_tensor = transform(img)[:3]
    return img_tensor

def main():
    N = 10_000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    renderer = DiffGaussRenderer(N=N,)

    # Random gaussians
    bd = 5.0
    be = 0.01

    renderer.mu = bd * (torch.rand(N, 3, device=device,) - 0.5)
    renderer.scales = be * (torch.rand(N, 3, device=device,))
    d = 3
    renderer.cols = torch.rand(N, d, device=device,)

    u = torch.rand(N, 1, device=device)
    v = torch.rand(N, 1, device=device)
    w = torch.rand(N, 1, device=device)

    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        ],
        -1,
    )
    renderer.quats = quats.to(device=device)
    renderer.opcs = torch.ones((N,), device=device,)
    renderer.register_params()

    # Create GT image
    cameras = readCamerasFromTransforms(
        './examples/nerf_example_data/nerf_synthetic/lego',
        'transforms_train.json',
        False
    )

    renderer.viewmat = torch.from_numpy(cameras[0].w2c).to(dtype=torch.float).to(renderer.device)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=renderer.params, lr=1e-2)

    resize  = T.Resize(256)

    pred = renderer.render()

    plt.ion()
    figure, ax = plt.subplots()
    im1 = ax.matshow(pred.detach().cpu())

    for iter in tqdm(range(100)):
        camera = copy.copy(cameras[0])
        viewmat = torch.from_numpy(camera.w2c).to(dtype=torch.float).to(renderer.device)
        # gt_image = read_image(camera.image_path).permute(1,2,0) / 255
        gt_image = image_path_to_tensor(camera.image_path)
        gt_image = resize(gt_image).permute(1,2,0)

        optimizer.zero_grad()

        pred = renderer.render()

        im1.set_data(pred.detach().cpu())
        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)

        loss = criterion(pred, gt_image)
        
        loss.backward()

        torch.nn.utils.clip_grad_value_(renderer.params, clip_value=1.0)
        for param in renderer.params:
            param.grad[param.grad.isnan()] = 0.

        optimizer.step()

        print(f'Iter: {iter}, Loss: {loss.item()}, Grad. Norms: {[p.abs().norm().item() for p in renderer.params]}')

    plt.matshow(pred.detach().cpu())
    plt.show()

if __name__ == '__main__':
    main()