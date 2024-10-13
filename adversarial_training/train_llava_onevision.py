import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*copying from a non-meta parameter.*",
    category=UserWarning
)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import requests
import copy
import torch
import os
from utils import DissimilarityLoss, encode_image, preprocess_image_llava_onevision, process_images_tensor
from torch import optim
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import contextlib
from torch.cuda.amp import autocast
import argparse
from transformers import BitsAndBytesConfig

pretrained = "lmms-lab/llava-onevision-qwen2-7b-si"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, 
    None, 
    model_name, 
    quantization_config=quantization_config,
    device_map=device_map,
    attn_implementation=None
)
model.eval()
to_pil = ToPILImage()


def process_and_save_images(input_dir, output_dir, log_dir, epsilon, alpha, num_steps, method, hallucination_type, use_categories):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if use_categories:
        categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d != '.DS_Store']
        for category in categories:
            category_path = os.path.join(input_dir, category)
            output_category_path = os.path.join(output_dir, category)
            log_category_path = os.path.join(log_dir, f"{category}.xlsx")
            
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            logs = []
            
            for img_name in tqdm(os.listdir(category_path)):
                img_path = os.path.join(category_path, img_name)
                image = Image.open(img_path).convert("RGB")
                img_rgb = image

                ## ori
                image = preprocess_image_llava_onevision(image).squeeze(0)

                log = apply_adv_perturbations(image, img_name, os.path.join(output_dir, category), img_rgb, epsilon, alpha, num_steps, method, hallucination_type)
                logs.extend(log)
            
            # Save logs to Excel
            df = pd.DataFrame(logs, columns=["Image", "Step", "Loss"])
            df.to_excel(log_category_path, index=False)
    else:
        logs = []
        for img_name in tqdm(os.listdir(input_dir)):
            img_path = os.path.join(input_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            img_rgb = image

            ## ori
            image = preprocess_image_llava_onevision(image).squeeze(0)

            log = apply_adv_perturbations(image, img_name, output_dir, img_rgb, epsilon, alpha, num_steps, method, hallucination_type)
            logs.extend(log)
        
        # Save logs to Excel
        log_category_path = os.path.join(log_dir, "logs.xlsx")
        df = pd.DataFrame(logs, columns=["Image", "Step", "Loss"])
        df.to_excel(log_category_path, index=False)

def apply_adv_perturbations(ori_img, img_name, output_category_path, img_rgb, epsilon, alpha, num_steps, method='i-fgsm', hallucination_type='non_h'):
    ori_img = ori_img.to(model.device)
    adv_img = ori_img.clone().detach().to(model.device).requires_grad_(True)

    if hallucination_type == 'h':
        random_direction = torch.randint(0, 2, adv_img.shape, device=adv_img.device).float() * 2 - 1
        random_perturbation = random_direction * (5/255)
        adv_img = torch.clamp(adv_img + random_perturbation, min=0, max=1).detach_()
        adv_img.requires_grad_(True)
    
    log = []

    # Dummy parameter for scheduler
    dummy_param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([dummy_param], lr=alpha)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        ori_inputs = process_images([img_rgb], image_processor, model.config)
        ori_inputs = [_image.to(dtype=torch.float16, device=device) for _image in ori_inputs]
        image_sizes = [img_rgb.size]
        ori_embeddings = model.prepare_inputs_labels_for_multimodal_for_attack(images=ori_inputs, image_sizes=image_sizes)
        ori_embeddings = ori_embeddings[0]

        ## ori
        adv_inputs = process_images_tensor([adv_img], image_processor, model.config, img_rgb, 'llava_onevision')
        adv_inputs = [_image.to(dtype=torch.float16, device=device) for _image in adv_inputs]
        adv_embeddings = model.prepare_inputs_labels_for_multimodal_for_attack(images=adv_inputs, image_sizes=image_sizes)
        adv_embeddings = adv_embeddings[0]
        
        loss_flag = "Descending" if hallucination_type == 'h' else "Ascending"
        loss = loss_fn(ori_embeddings, adv_embeddings, flag=loss_flag)
        log.append([img_name, step, loss.item()])
        
        loss.backward()

        if hallucination_type == 'h' and step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item()}, LR = {scheduler.get_last_lr()[0]}")
            step_output_dir = os.path.join(output_category_path, f"step_{step}")
            if not os.path.exists(step_output_dir):
                os.makedirs(step_output_dir)
            temp_output_img_path = os.path.join(step_output_dir, os.path.splitext(img_name)[0] + '.png')
            adv_img_rgb = to_pil(adv_img.squeeze(0))
            adv_img_rgb.save(temp_output_img_path)
        elif hallucination_type == 'non_h' and step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item()}, LR = {scheduler.get_last_lr()[0]}")
            step_output_dir = os.path.join(output_category_path, f"step_{step}")
            if not os.path.exists(step_output_dir):
                os.makedirs(step_output_dir)
            temp_output_img_path = os.path.join(step_output_dir, os.path.splitext(img_name)[0] + '.png')
            adv_img_rgb = to_pil(adv_img.squeeze(0))
            adv_img_rgb.save(temp_output_img_path)
        
        if hallucination_type == 'h' and step == 100:
            break
        if hallucination_type == 'non_h' and step == num_steps:
            break

        with torch.no_grad():
            lr = scheduler.get_last_lr()[0]
            if method == 'i-fgsm':
                adv_img += lr * adv_img.grad.sign()
            elif method == 'pgd':
                adv_img += lr * adv_img.grad
            eta = torch.clamp(adv_img - ori_img, min=-epsilon, max=epsilon)
            adv_img = torch.clamp(ori_img + eta, min=0, max=1).detach_()
            adv_img.requires_grad_(True)

        optimizer.step()
        scheduler.step()
            
    return log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save results')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to the log directory to save logs')
    parser.add_argument('--epsilon', type=float, default=8/255, help='Maximum perturbation epsilon')
    parser.add_argument('--alpha', type=float, help='Step size alpha')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of steps for adversarial attack')
    parser.add_argument('--method', type=str, default='i-fgsm', choices=['i-fgsm', 'pgd'], help='Adversarial attack method to use')
    parser.add_argument('--hallucination_type', type=str, default='non_h', choices=['non_h', 'h'], help='Type of training: non_h for non-hallucinated, h for hallucinated')
    parser.add_argument('--use_categories', action='store_true', help='Whether to use categories for image processing')
    args = parser.parse_args()

    if args.alpha is None:
        if args.method == 'i-fgsm':
            alpha = 0.5 / 255
        elif args.method == 'pgd':
            alpha = 1.0
    else:
        alpha = args.alpha

    loss_fn = DissimilarityLoss()
    process_and_save_images(args.input_dir, args.output_dir, args.log_dir, args.epsilon, alpha, args.num_steps, args.method, args.hallucination_type, args.use_categories)


