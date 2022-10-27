"""A generative dataset that creates haunted/not haunted photos"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Country211

ghost_dir = Path("ghosts")
ghost_images = ["casper.png", "jasper.png", "longboi.png"]


def gradient_2d(start, stop, width, height, horizontal):
    if horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, horizontal) in enumerate(zip(start_list, stop_list,
                                                      horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height,
                                      horizontal)
    return result.astype(np.uint8)


def paste(background, foreground, x_offset, y_offset):
    mask = foreground[..., 3].astype(bool)
    fg = foreground[..., :3]*255
    x_end = x_offset + foreground.shape[1]
    y_end = y_offset + foreground.shape[0]
    background[y_offset:y_end, x_offset:x_end][mask] = fg[mask]
    return


def load_ghost(path, scale=15):
    """Load and rescale ghost images.

    Images resized to 1/10 their normal size by default.
    """
    src = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    bgr = src[:, :, :3]  # Channels 0..2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Some sort of processing...

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    alpha = src[:, :, 3]  # Channel 3
    image = np.dstack([bgr, alpha])  # Add the alpha channel
    image_resized = resize(image, (image.shape[0] // scale,
                                   image.shape[1] // scale),
                           anti_aliasing=False)
    return image_resized


def create_characters(filename="characters.png"):
    fig, axes = plt.subplots(1, len(ghost_images))
    for ax, ghost in zip(axes, ghost_images):
        img = load_ghost(ghost_dir / ghost)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(ghost.replace('.png', '').capitalize())
    plt.savefig(filename,  transparent=True, bbox_inches='tight')
    return


def create_examples(haunted):
    classes = ["Safe", "Haunted"]
    fig, axes = plt.subplots(3, 5)
    for i, ax in enumerate(axes.ravel()):
        ghost_im, y = haunted[i]
        ax.imshow(ghost_im)
        ax.axis('off')
        ax.set_title(classes[y])
    plt.savefig("examples.png", bbox_inches="tight")


class HauntedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.ghosts = [load_ghost(ghost_dir / im) for im in ghost_images]

    def __getitem__(self, item):
        img, _ = self.dataset[item]
        p = np.random.random()
        if p < 0.1:
            img = self.add_fog(self.add_ghost(img))
        elif p < 0.3:
            # Add fog
            img = self.add_fog(img)
        elif p < 0.5:
            # Add a ghost
            img = self.add_ghost(img)
            return img, 1
        else:
            return np.array(img), 0
        return img, 1

    def __len__(self):
        return len(self.dataset)

    def add_ghost(self, image):
        # Choose a ghost
        ix = np.random.randint(0, 3)
        # print(ghost_images[ix])
        ghost = self.ghosts[ix]
        # Pad the image to fit the ghost
        a, b = ghost.shape[0], ghost.shape[1]
        background = np.array(image)
        background = np.pad(np.array(image), ((a, a), (b, b), (0, 0)))
        # Position
        x = np.random.randint(a, background.shape[0] - a)
        y = np.random.randint(b, background.shape[1] - b)
        paste(background, ghost, y, x)
        # Crop the centre again
        background = background[a:-a, b:-b]
        return background

    def add_fog(self, image):
        background = np.array(image).astype(float)
        w, h, c = background.shape
        fog = gradient_3d(w, h, (0, )*c, (255,)*c, (False,)*c)
        background = 0.6*fog + 0.4*background
        return background.astype(np.uint8)


if __name__ == "__main__":
    if not Path("characters.png").exists():
        create_characters()
    if not Path("examples.png").exists():
        transform = transforms.CenterCrop(250)
        base = Country211("./data", download=True, transform=transform)
        haunted = HauntedDataset(base)
        create_examples(haunted)

