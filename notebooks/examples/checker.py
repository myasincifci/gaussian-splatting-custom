import time 
import torch
import matplotlib.pyplot as plt
from diff_gaussian_renderer import DiffGaussRenderer
from tqdm import tqdm

def main():
    renderer = DiffGaussRenderer()
    
    # Create GT image
    gt_image = torch.ones((renderer.H, renderer.W, 3)) * 1.0
    gt_image[: renderer.H // 2, : renderer.W // 2, :] = torch.tensor([1.0, 0.0, 0.0])
    gt_image[renderer.H // 2 :, renderer.W // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])


    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=renderer.params, lr=1e-2)


    pred = renderer.render()

    plt.ion()
    figure, ax = plt.subplots()
    im1 = ax.matshow(pred.detach().cpu())

    for iter in tqdm(range(100)):
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