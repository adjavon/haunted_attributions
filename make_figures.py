#!/usr/bin/env python
# coding: utf-8
import zarr
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    results = zarr.open('result.zarr', 'r')

    # Plot the predictions
    for i, pred in enumerate(results['predictions'][:]):
        x = np.array([i-0.2, i+.2])
        plt.bar(x, pred, width=0.3)
        plt.axvline(i + 0.5, color='k', linestyle=':')
        plt.xlim(-0.5, 12.5)
        plt.xticks(list(range(13)))
        plt.xlabel("Sample")
        plt.ylabel("Logits")
    plt.savefig("attribution_figures/predictions_per_sample.png", bbox_inches='tight')

    # Plot the attributions
    names = ['deeplift', 'guided_backprop', 'guided_gradcam',
             'inputXgradient', 'integrated_gradients', 'saliency']
    cmap = 'gray'
    #
    for name in names:
        print(name)
        for i in range(3):
            start = i * 4
            end = (i + 1)*4
            images = results["images"][start:end]
            ig = results[name][start:end]
            fig, (axes1, axes2, axes3) = plt.subplots(3, len(images))
            for im, attr, ax1, ax2, ax3 in zip(images, ig, axes1, axes2, axes3):
                ax1.imshow(im)
                ax1.axis('off')
                attr_plus = np.clip(attr, 0, attr.max()).sum(axis=0)
                attr_minus = -np.clip(-attr, 0, -attr.min()).sum(axis=0)
                ax2.imshow(attr_plus > np.percentile(attr_plus, 95), cmap=cmap)
                ax2.axis('off')
                ax3.imshow(attr_minus > np.percentile(attr_minus, 70), cmap=cmap)
                ax3.axis('off')
            fig.savefig(f"attribution_figures/{name}_{i}.png", bbox_inches="tight", transparent=True)
