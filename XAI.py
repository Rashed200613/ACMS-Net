import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import config

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ==========================================================
# GradCAM
# ==========================================================

def generate_gradcam(
    model,
    image_tensor,
    image_rgb,
    target_layer,
    save_path
):

    model.eval()

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(
        input_tensor=image_tensor
    )[0]

    visualization = show_cam_on_image(
        image_rgb,
        grayscale_cam,
        use_rgb=True
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )


# ==========================================================
# Feature Map Visualization
# ==========================================================

def save_feature_maps(
    model,
    image_tensor,
    target_layer,
    save_dir
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    features = []

    def hook_fn(
        module,
        input,
        output
    ):
        features.append(
            output.detach().cpu()
        )

    handle = target_layer.register_forward_hook(
        hook_fn
    )

    model.eval()

    with torch.no_grad():
        _ = model(image_tensor)

    handle.remove()

    fmap = features[0][0]

    num_maps = min(
        fmap.shape[0],
        16
    )

    for i in range(num_maps):

        plt.figure(
            figsize=(4,4)
        )

        plt.imshow(
            fmap[i],
            cmap="jet"
        )

        plt.axis("off")

        plt.savefig(
            os.path.join(
                save_dir,
                f"feature_map_{i}.png"
            ),
            bbox_inches="tight"
        )

        plt.close()


# ==========================================================
# Attention Map Visualization
# ==========================================================

def save_attention_maps(
    model,
    image_tensor,
    target_layer,
    save_dir
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    activations = []

    def hook_fn(
        module,
        input,
        output
    ):
        activations.append(
            output.detach().cpu()
        )

    handle = target_layer.register_forward_hook(
        hook_fn
    )

    model.eval()

    with torch.no_grad():
        _ = model(image_tensor)

    handle.remove()

    attention = activations[0][0]

    attention = torch.mean(
        attention,
        dim=0
    )

    plt.figure(
        figsize=(6,6)
    )

    plt.imshow(
        attention,
        cmap="jet"
    )

    plt.colorbar()

    plt.axis("off")

    plt.savefig(
        os.path.join(
            save_dir,
            "attention_map.png"
        ),
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# t-SNE
# ==========================================================

def generate_tsne(
    model,
    dataloader,
    device,
    save_path
):

    features = []
    labels = []

    model.eval()

    with torch.no_grad():

        for images, target in dataloader:

            images = images.to(device)

            x = model.conv_stem(images)

            x = model.amsfe_stage1(x)

            x = model.pool1(x)

            x = model.bridge(x)

            x = model.amsfe_stage2(x)

            x = model.pool2(x)

            x = model.conv_block(x)

            x = model.global_pool(x)

            x = torch.flatten(x, 1)

            features.append(
                x.cpu().numpy()
            )

            labels.append(
                target.numpy()
            )

    features = np.concatenate(
        features,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    tsne = TSNE(
        n_components=2,
        random_state=42
    )

    embedding = tsne.fit_transform(
        features
    )

    plt.figure(
        figsize=(8,6)
    )

    scatter = plt.scatter(
        embedding[:,0],
        embedding[:,1],
        c=labels
    )

    plt.colorbar(scatter)

    plt.title(
        "t-SNE Feature Embedding"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()