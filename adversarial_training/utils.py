# Portions of this code are adapted from LLaVA-NeXT project.
# Original code borrowed from: https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/mm_utils.py
# License: Please refer to the original repository for licensing terms.


import os
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from transformers.image_processing_utils import BatchFeature
import math

# Normalize image tensors
def encode_image(image_tensor):
    transform = Compose([
        Normalize((0.48145466, 0.4578275, 0.40821073), 
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform(image_tensor)

def preprocess_image_llava(image):
    transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(224),
        ToTensor()
    ])
    return transform(image).unsqueeze(0)

def preprocess_image_blip(image):
    transform = Compose([
        Resize((224,224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor()
    ])
    return transform(image).unsqueeze(0)

def preprocess_image_qwen(image):
    transform = Compose([
        Resize((224,224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor()
    ])
    return transform(image)

def preprocess_image_llava_next(image):
    transform = Compose([ToTensor()])
    return transform(image).unsqueeze(0)

def preprocess_image_llava_onevision(image):
    transform = Compose([ToTensor()])
    return transform(image).unsqueeze(0)

def apply_transform_qwen(image_tensor):
    transform = Compose([
        Normalize((0.48145466, 0.4578275, 0.40821073), 
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    images = []
    images.append(transform(image_tensor).squeeze(0))
    images = torch.stack(images, dim=0)
    return images

class DissimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(DissimilarityLoss, self).__init__()

    def forward(self, original_embeddings, modified_embeddings, flag):
        cosine_sim = F.cosine_similarity(original_embeddings, modified_embeddings)
        if flag == "Ascending":
            cosine_sim = - cosine_sim
        return cosine_sim.mean()

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def resize_and_pad_image_tensor(tensor, target_resolution, image):
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the tensor using torchvision transforms
    transform = Compose([
        Resize((new_height, new_width), interpolation=InterpolationMode.BICUBIC, antialias=True),
    ])
    resized_tensor = transform(tensor)

    # Create a new tensor with the target size and paste the resized tensor onto it
    new_tensor = torch.zeros((tensor.shape[0], target_height, target_width))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_tensor[:, paste_y:paste_y + new_height, paste_x:paste_x + new_width] = resized_tensor

    return new_tensor

def divide_tensor_to_patches(tensor, patch_size):
    C, H, W = tensor.shape
    patches = []

    # Ensure the tensor height and width are divisible by patch_size
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h))

    # Compute the number of patches along height and width
    n_patches_h = tensor.shape[1] // patch_size
    n_patches_w = tensor.shape[2] // patch_size

    # Divide the tensor into patches
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch = tensor[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)

    return patches

def process_anyres_image_tensor(image_tensor, processor, grid_pinpoints, img_rgb, model_name):
    # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)

    best_resolution = select_best_resolution(img_rgb.size, possible_resolutions)
    image_padded = resize_and_pad_image_tensor(image_tensor, best_resolution, img_rgb)

    patches = divide_tensor_to_patches(image_padded, processor.crop_size["height"])

    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)

    transform = Compose([
        Resize((shortest_edge, shortest_edge), interpolation=InterpolationMode.BICUBIC, antialias=True),
    ])
    image_original_resize = transform(image_tensor)
    image_original_resize = torch.clamp(image_original_resize, 0, 1)

    for i in range(len(patches)):
        patches[i] = patches[i].to(image_original_resize.device)
    image_patches = [image_original_resize] + patches
    if model_name == 'llava_next':
        image_patches = [encode_image(image_patch) for image_patch in image_patches]
    elif model_name == 'llava_onevision':
        image_patches = [encode_image_siglip(image_patch)['pixel_values'][0] for image_patch in image_patches]
    else:
        raise ValueError("Unsupported model_name. Please use 'llava_next' or 'llava_onevision'.")

    return torch.stack(image_patches, dim=0)

def process_images_tensor(images, image_processor, model_cfg, img_rgb, model_name):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "highres":
        for image in images:
            image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image_tensor(image, image_processor, model_cfg.image_grid_pinpoints, img_rgb, model_name)
            new_images.append(image)
    elif image_aspect_ratio == "crop_split":
        for image in images:
            image = process_highres_image_crop_split(image, model_cfg, image_processor)
            new_images.append(image)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def encode_image_siglip(
    images: torch.Tensor,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
    size=(384, 384),
    rescale_factor=1/255,
    return_tensors='pt',
):
    # Ensure input is a batch of tensors, even if a single tensor is provided
    if images.ndim == 3:
        images = images.unsqueeze(0)

    # Define transformations
    transforms = Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),  # Resize using torchvision's Resize
        Normalize(mean=image_mean, std=image_std),  # Normalize using torchvision's Normalize
    ])

    # Apply transformations to each image in the batch
    images = transforms(images)

    # Optionally convert to the desired tensor type (e.g., float32)
    images = images.float()
    data = {"pixel_values": images}

    return BatchFeature(data=data, tensor_type=return_tensors)