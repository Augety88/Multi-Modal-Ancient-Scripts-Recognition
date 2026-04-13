# 古文字数据增强与风格转换项目

本项目用于甲骨文、金文等古文字手写体和拓片图像的数据增强与风格转换，包含多种数据增强方法和深度学习模型相关脚本，配套学术研究论文使用。

## 支撑论文
本项目为以下学术论文的官方实现代码：
> **论文标题**：Multi-modal ancient scripts recognition via deep learning with data homogenization and augmentation
> 作者：Nan Wang, Weichen Wang, Bang Li 等
> 发表期刊：npj Heritage Science
> DOI：10.1038/s40494-025-02095-x

## 目录结构

```
Data_augmentation/
    数据增广.py           # 主数据增强脚本（批量处理，支持多种方法）
    test.py              # 增强方法测试脚本
    test/
        train/
        val/
    test_aug/
        train/
        val/
Jinwen/
    金文手写/
    金文手写+金文拓片/
    金文手写+金文拓片_增广/
    金文手写+金文拓片转手写/
    金文手写+金文拓片转手写_增广/
    金文拓片/
    金文拓片转手写/
Oracle/
    （最终）甲骨文拓片转手写_增广+甲骨文手写体_增广/
    甲骨文手写体/
    甲骨文手写体+甲骨文拓片/
    甲骨文手写体+甲骨文拓片_增广/
    甲骨文拓片/
    甲骨文拓片转手写/
    甲骨文拓片转手写+甲骨文手写体/
Rubbing_to_handwriting/
    拓片转手写体.py
    convert4.py
    models_adafunet.py
    models.py
    netG_t2m_11.pth
    non_local.py
    input/
    output/
```

## 环境依赖

- Python 3.x
- OpenCV (`cv2`)
- numpy
- tqdm
- torchvision
- Pillow

安装依赖：
```sh
pip install opencv-python numpy tqdm torchvision pillow
```

## 数据增强使用说明

### 1. 增强金文/甲骨文等数据集

进入 `Data_augmentation` 目录，运行主增强脚本：

```sh
python 数据增广.py --input_dir <原始数据集目录> --output_dir <增强后输出目录>
```

- `--input_dir`：原始数据集路径（如 `金文手写+金文拓片转手写`）
- `--output_dir`：增强后数据集保存路径（如 `金文手写+金文拓片转手写_增广`）

脚本会自动遍历 `train` 和 `test` 子文件夹，对每张图片进行多种增强（旋转、仿射、噪声、亮度、模糊等），每张图片扩充8倍。

### 2. 增强测试集样例

使用 `test.py` 脚本增强测试集：

```sh
python test.py --input_dir test --output_dir test_aug
```

- 默认增强 `test/train` 和 `test/val` 下的图片，输出到 `test_aug` 目录。

## 拓片转手写体风格转换

`Rubbing_to_handwriting` 目录下包含模型定义和风格转换脚本。  
如需使用模型进行图片风格转换，请参考该目录下脚本的注释说明。

## 主要功能

- 支持多种图像增强方式：仿射、旋转、椒盐噪声、高斯噪声、亮度对比度、模糊等
- 批量处理，自动扩充数据集
- 支持灰度和彩色图像
- 适用于古文字手写体和拓片数据集的增强与风格迁移任务

## 主要脚本

- 数据增强主脚本：`Data_augmentation/数据增广.py`
- 增强测试脚本：`Data_augmentation/test.py`
- 风格转换相关脚本：`Rubbing_to_handwriting/`

---


