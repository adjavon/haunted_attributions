from captum.attr import (Saliency, InputXGradient, GuidedBackprop, DeepLift,
                         GuidedGradCam, IntegratedGradients)
import numpy as np
import torch
import zarr


from data import get_haunted_dataset
from utils import Vgg2D


def create_batch(haunted, items=None):
    if items is None:
        items = [7, 13, 42]
    additions = ['ghost', 'fog', 'both']
    images = []
    target = []
    for i in items:
        img, _ = haunted.dataset[i]
        img = np.array(img)
        images.append(img)
        target.append(0)
        for addition in additions:
            if addition == 'ghost':
                ghost = haunted.add_ghost(img)
                images.append(ghost)
                target.append(1)
            elif addition == 'fog':
                fog = haunted.add_fog(img)
                images.append(fog)
                target.append(1)
            elif addition == 'both':
                both = haunted.add_fog(haunted.add_ghost(img))
                images.append(both)
                target.append(1)
    # Stack the images
    images.append(np.zeros_like(img))
    target.append(0)
    images = np.stack(images)
    return images, target

if __name__ == "__main__":
    # Create the datasets
    print("Creating and saving data")
    haunted = get_haunted_dataset(split="test")
    images, target = create_batch(haunted)
    #
    results = zarr.open("result.zarr")
    results["images"] = images
    results["target"] = target
    print("Creating model and loading weights")
    # Get the network and load weights
    model = Vgg2D(input_size=(256, 256),
                  fmaps=8,
                  output_classes=2,
                  input_fmaps=3)
    model.load_state_dict(torch.load('models/model.pth'))
    model.eval()
    # predict each
    print("Running predictions")
    tensor_img = np.stack([haunted.normalize(img) for img in images])
    tensor_img = torch.from_numpy(tensor_img).contiguous()
    with torch.no_grad():
        predictions = model(tensor_img)
        results["predictions"] = predictions.numpy()

    names = ['saliency', 'inputXgradient', 'guided_backprop', 'deeplift',
             'guided_gradcam', 'integrated_gradients']
    attributions = [Saliency(model), InputXGradient(model),
                    GuidedBackprop(model), DeepLift(model),
                    GuidedGradCam(model, model.features[24]),
                    IntegratedGradients(model)]
    for name, attr in zip(names, attributions):
        print(f"Running {name} attribution.")
        attribution = attr.attribute(tensor_img, target=target)
        results[name] = attribution.detach().numpy()
    # TODO Plot and save results


