from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI


def get_image_embedding(image_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
    image_embeds_np = image_embeds.detach().numpy().flatten().tolist()
    return image_embeds_np


def get_text_embedding(texts):
    # 加载预训练的CLIP模型和处理器
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 处理文本以适配模型输入
    inputs = processor(text=texts, return_tensors="pt", padding=True)

    # 获取文本嵌入
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)

    # 将嵌入转换为NumPy数组
    text_embeds_np = text_embeds.detach().numpy().flatten().tolist()
    
    return text_embeds_np





def generate_img_by_img(prompt,encoded_image):
    import requests
    import json

    # 设置API端点
    url = "http://localhost:7862/sdapi/v1/txt2img"
    
    # 构建请求体
    payload = {
        "prompt": prompt,
        "negative_prompt": "",
        "styles": [],
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "sampler_name": "",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 50,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "restore_faces": True,
        "tiling": False,
        "do_not_save_samples": False,
        "do_not_save_grid": False,
        "eta": 0,
        "denoising_strength": 0.75,
        "s_min_uncond": 0,
        "s_churn": 0,
        "s_tmax": 0,
        "s_tmin": 0,
        "s_noise": 0,
        "override_settings": {},
        "override_settings_restore_afterwards": True,
        "refiner_checkpoint": "",
        "refiner_switch_at": 0,
        "disable_extra_networks": False,
        "comments": {},
        "init_images": [encoded_image], 
        "resize_mode": 0,
        "image_cfg_scale": 0,
        "mask": "",
        "mask_blur_x": 4,
        "mask_blur_y": 4,
        "mask_blur": 0,
        "inpainting_fill": 0,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 0,
        "inpainting_mask_invert": 0,
        "initial_noise_multiplier": 0,
        "latent_mask": "",
        "sampler_index": "Euler",
        "include_init_images": False,
        "script_name": "",  # 将脚本名置空
        "script_args": [],
        "send_images": True,
        "save_images": False,
        "alwayson_scripts": {}
    }

    # 将请求体转换为JSON格式
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(payload)

    # 发送POST请求
    response = requests.post(url, headers=headers, data=data)

    # 检查响应状态码
    if response.status_code == 200:
        result = response.json()
        
        # 获取返回的Base64编码图像
        image_base64 = result['images'][0]
        return image_base64
        
    else:
        print(f"Failed to generate image. Status code: {response.status_code}")
        print(response.json())

if __name__ == '__main__':
    #res = get_image_embedding('./0014.jpg')
    #res = get_text_embedding('黑猫警长')
    #print(res,type(res))
    #download_coco_part()
    generate_img_by_img('Generate a knight image','./coco_dataset/val2017/00000000000.jpg','./result.jpg')