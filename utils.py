from typing import List
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from einops import rearrange

def tensor_to_numpy(tensor, dict_size):
    """Convert a PyTorch tensor to a numpy array suitable for Matplotlib."""

    B, H, W = tensor.shape
    tensor = tensor.reshape(B, 1, H, W)

    # remove mask class
    tensor = torch.clamp(tensor - 1, 0)
    tensor = tensor * (255 // dict_size)

    nrow = int(tensor.size(0) ** 0.5)
    tensor = make_grid(tensor, nrow=nrow, pad_value=0).cpu()


    tensor = tensor.permute(1, 2, 0)  # Change to (H, W, C)
    array = tensor.numpy().astype(np.uint8)  # Scale and convert to uint8
    return array

def create_animation(tensors, output_path, duration=5, dict_size=10):
    """Create an animation from a series of tensors and save it as a video file."""

    # Choose a reasonable fps, like 25
    fps = 25

    # Calculate the number of frames to use based on the desired duration and fps
    num_frames = fps * duration

    # Sample tensors to match the desired number of frames
    sampled_indices = np.linspace(0, len(tensors) - 1, num_frames).astype(int)
    tensors = [tensors[i] for i in sampled_indices]

    # tensor of shape (T B C H W)
    frames = [tensor_to_numpy(tensor, dict_size) for tensor in tensors]
    frames += [frames[-1]] * fps

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])
    ax.axis('off')

    def update(frame):
        img.set_data(frame)
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=1/fps)
    
    # Save the animation
    Writer = animation.writers['pillow']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(output_path, dpi=150, writer=writer)


def plot_generation(xts: List[torch.Tensor], n_plots: int = 5) -> None:
    stride = len(xts) // n_plots
    xts = [xts[i * stride] for i in range(n_plots)] + [xts[-1]]
    grid = torch.stack(xts, dim=0)
    grid = torch.clamp(grid - 1, 0).cpu().numpy()
    grid = rearrange(grid, 't b h w -> (t h) (b w)')
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')