import argparse
import cv2
import os
import numpy as np
import random
import tqdm
from torchvision import transforms
from PIL import Image, ImageFilter


def add_salt_and_pepper_noise(image, prob):
    """添加椒盐噪声，适应单通道和三通道图像"""
    output = np.copy(image)
    total_pixels = image.size
    num_salt = int(prob * total_pixels)
    num_pepper = int(prob * total_pixels)

    # 对图像进行处理，检查通道数
    if len(image.shape) == 3:  # 彩色图像（3通道）
        for _ in range(num_salt):
            # 随机选取位置
            x, y = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
            # 给每个通道添加盐噪声
            output[x, y] = [255, 255, 255]  # 白色噪声

        for _ in range(num_pepper):
            # 随机选取位置
            x, y = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
            # 给每个通道添加椒噪声
            output[x, y] = [0, 0, 0]  # 黑色噪声
    
    else:  # 灰度图像（1通道）
        for _ in range(num_salt):
            x, y = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
            output[x, y] = 255  # 白色噪声

        for _ in range(num_pepper):
            x, y = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
            output[x, y] = 0  # 黑色噪声
    
    return output

def add_gaussian_noise(image, mean=0, sigma=0.05):
    """添加高斯噪声，适应单通道和三通道图像"""
    if len(image.shape) == 3:  # 彩色图像（3通道）
        row, col, channels = image.shape
        noisy_image = np.zeros_like(image, dtype=np.float32)  # 创建一个同样大小的噪声图像
        
        for channel in range(channels):  # 对每个通道添加噪声
            gaussian = np.random.normal(mean, sigma, (row, col))
            noisy_image[:, :, channel] = image[:, :, channel] + gaussian * 255
        
        # 保证值在0到255之间，并转换为uint8
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)
    
    else:  # 灰度图像（1通道）
        row, col = image.shape
        gaussian = np.random.normal(mean, sigma, (row, col))
        noisy_image = image + gaussian * 255  # 加上高斯噪声
        noisy_image = np.clip(noisy_image, 0, 255)  # 限制值在0到255之间
        return noisy_image.astype(np.uint8)


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """调整图像亮度和对比度"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def random_affine_transform(image):
    """仿射变换，自动适应通道数"""
    # 获取图像的高度、宽度和通道数
    if len(image.shape) == 3:  # 彩色图像
        rows, cols, channels = image.shape
    else:  # 灰度图像
        rows, cols = image.shape
        channels = 1
    
    # 随机生成缩放因子（0.7 到 1.3）
    scale = random.uniform(0.7, 1.3)
    
    # 随机生成错切因子，范围是 -0.3 到 0.3，增加错切幅度
    shear_x = random.uniform(-0.3, 0.3)
    shear_y = random.uniform(-0.3, 0.3)  # 对y轴也增加错切
    
    # 计算仿射变换矩阵
    matrix = np.array([[1, shear_x, 0],  # x轴错切
                       [shear_y, scale, 0]], dtype=np.float32)
    
    # 计算仿射变换后图像的尺寸
    new_cols = int(cols * (1 + abs(shear_x)))  # 根据错切因子调整宽度
    new_rows = int(rows * scale)  # 根据缩放因子调整高度

    # 对图像进行仿射变换
    transformed_image = cv2.warpAffine(image, matrix, (new_cols, new_rows), borderValue=255)
    
    return transformed_image


def random_rotation(image):
    """旋转图像"""
    angle = random.randint(-20, 20)
    rows, cols = image.shape[:2]  # 获取图像的行数和列数
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (cols, rows), borderValue=255)


def random_blur(image):
    """应用高斯模糊"""
    if random.random() < 0.3:
        return cv2.GaussianBlur(image, (7, 7), 0.15)
    return image

def random_grayscale(image):
    """将图像转换为灰度，如果原图是灰度图则不做处理"""
    if len(image.shape) == 3 and image.shape[2] == 3:  # 彩色图像
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # 如果已经是灰度图，直接返回

def apply_data_augmentation(image):
    """应用所有的增广方法"""
    if random.random() < 0.5:  # 水平翻转概率
        image = cv2.flip(image, 1)
    if random.random() < 1.0:  # 旋转概率
        image = random_rotation(image)
    if random.random() < 1.0:  # 仿射变换概率
        image = random_affine_transform(image)
    if random.random() < 0.3:  # 椒盐噪声概率
        image = add_salt_and_pepper_noise(image, 0.03)
    if random.random() < 0.3:  # 高斯噪声概率
        image = add_gaussian_noise(image)
    if random.random() < 1.0:  # 明暗变化概率
        image = adjust_brightness_contrast(image, random.uniform(0.8, 1.2), random.randint(-20, 20))
    # if random.random() < 0.5:  # 灰度处理概率
    #     image = random_grayscale(image)
    if random.random() < 0.3:  # 高斯模糊概率
        image = random_blur(image)
    return image

def save_augmented_images(input_dir, output_dir):
    for folder in ['train', 'test']:
        input_folder = os.path.join(input_dir, folder)
        output_folder = os.path.join(output_dir, folder)
        os.makedirs(output_folder, exist_ok=True)

        for class_folder in os.listdir(input_folder):
            class_input_folder = os.path.join(input_folder, class_folder)
            class_output_folder = os.path.join(output_folder, class_folder)
            os.makedirs(class_output_folder, exist_ok=True)

            if os.path.isdir(class_input_folder):
                for img_name in tqdm.tqdm(os.listdir(class_input_folder)):
                    img_path = os.path.join(class_input_folder, img_name)
                    
                    # 读取转换并转换为灰度图像
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    orig_shape = img.shape

                    #调整图片大小为原本大小
                    img = cv2.resize(img, (orig_shape[1], orig_shape[0]))
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # 保存原图
                    original_img_path = os.path.join(class_output_folder, f"{img_name}")
                    cv2.imwrite(original_img_path, img)
                    print(f"Saved original image: {original_img_path}")

                    # 扩充8次
                    for i in range(8):
                        augmented_img = apply_data_augmentation(img)
                        augmented_img = cv2.resize(augmented_img, (orig_shape[1], orig_shape[0]))
                        augmented_img_path = os.path.join(class_output_folder, f"aug_{i}_{img_name}")
                        cv2.imwrite(augmented_img_path, augmented_img)
                        print(f"Saved augmented image: {augmented_img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Augmentation for Handwriting Dataset")
    parser.add_argument('--input_dir', type=str, default='金文手写+金文拓片转手写', help="Path to the input dataset directory")
    parser.add_argument('--output_dir', type=str, default='金文手写+金文拓片转手写_增广', help="Path to the output augmented dataset directory")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 执行数据增广
    save_augmented_images(args.input_dir, args.output_dir)
