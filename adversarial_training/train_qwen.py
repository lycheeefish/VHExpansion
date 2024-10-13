import torch
import os
from utils import DissimilarityLoss, preprocess_image_qwen, apply_transform_qwen
from torch import optim
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import argparse

torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda:1", trust_remote_code=True).eval()
model.requires_grad_(False)
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
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

                ## resize
                image = preprocess_image_qwen(image)

                log = apply_adv_perturbations(image, img_name, os.path.join(output_dir, category), img_path, epsilon, alpha, num_steps, method, hallucination_type)
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

            ## resize
            image = preprocess_image_qwen(image)

            log = apply_adv_perturbations(image, img_name, output_dir, img_path, epsilon, alpha, num_steps, method, hallucination_type)
            logs.extend(log)
        
        # Save logs to Excel
        log_category_path = os.path.join(log_dir, "logs.xlsx")
        df = pd.DataFrame(logs, columns=["Image", "Step", "Loss"])
        df.to_excel(log_category_path, index=False)

def apply_adv_perturbations(ori_img, img_name, output_category_path, img_path, epsilon, alpha, num_steps, method='i-fgsm', hallucination_type='non_h'):
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

        ori_embeddings = model.transformer.visual.encode([img_path])

        ## resize
        adv_inputs = apply_transform_qwen(adv_img).to(model.device)
        adv_embeddings = model.transformer.visual(adv_inputs)
        
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