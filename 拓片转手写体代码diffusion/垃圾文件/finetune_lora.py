import os
import shutil
import subprocess

def main(
    raw_data_dir='diffusion转换模型训练集',
    train_dir='lora-output/train',
    output_dir='lora-output',
    prompt='handwriting',
    pretrained_model='runwayml/stable-diffusion-v1-5',
    resolution=100,
    train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_train_steps=2000
):
    # 1. 数据整理
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir, exist_ok=True)

    images = []
    for fname in os.listdir(raw_data_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            src = os.path.join(raw_data_dir, fname)
            dst = os.path.join(train_dir, fname)
            shutil.copy(src, dst)
            images.append(fname)
    print(f"共找到图片: {len(images)} 张")
    print(f"训练集已整理到: {train_dir}")
    # 2. 生成 captions.txt（每张图片同样描述）
    captions_path = os.path.join(train_dir, 'captions.txt')
    with open(captions_path, 'w', encoding='utf-8') as f:
        for fname in images:
            f.write(f"{fname}\t{prompt}\n")
    print(f"每张图片的描述均为: {prompt}")

    # 3. 调用 LoRA 训练脚本
    print("开始LoRA微调训练...")
    command = [
        "accelerate", "launch", "train_dreambooth_lora.py",
        f"--pretrained_model_name_or_path={pretrained_model}",
        f"--instance_data_dir={train_dir}",
        f"--output_dir={output_dir}",
        f"--instance_prompt={prompt}",
        f"--resolution={resolution}",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--learning_rate={learning_rate}",
        f"--max_train_steps={max_train_steps}"
    ]
    print(" ".join(command))
    subprocess.run(command)
    print(f"LoRA训练完成，权重保存在: {output_dir}")

if __name__ == '__main__':
    main()