# Datasets Preparation

We recommend placing all datasets under the same folder (e.g., `$DATA`) for easier management. Follow the instructions below to organize each dataset.

The expected file structure:
```
$DATA/
|-- caltech-101/
|-- oxford_pets/
|-- stanford_cars/
|-- ...
```

If you already have some datasets installed elsewhere, you can create symbolic links in `$DATA/dataset_name` pointing to the original data to avoid duplicate downloads.

---

## Table of Contents

### Single-Domain Datasets
- [OxfordPets](#oxfordpets)
- [Flowers102](#flowers102)
- [DTD](#dtd)
- [Caltech101](#caltech101)
- [Food101](#food101)
- [Stanford Cars](#stanford-cars)
- [FGVC Aircraft](#fgvc-aircraft)
- [UCF101](#ucf101)
- [SUN397](#sun397)
- [EuroSAT](#eurosat)
- [CIFAR10/100](#cifar10100)
- [ImageNet](#imagenet)

### Multi-Domain Datasets
- [DomainNet](#domainnet)
- [Office-Caltech10](#office-caltech10)

---

## Single-Domain Datasets

> **Acknowledgement**: This guide for dataset preparation is adapted from the official [CoOp repository](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

### OxfordPets

1. Create a folder named `oxford_pets/` under `$DATA`.
2. Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
3. Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
4. Download `split_zhou_OxfordPets.json` from [Google Drive](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing).

The directory structure should look like:
```
oxford_pets/
|-- images/
|-- annotations/
|-- split_zhou_OxfordPets.json
```

### Flowers102

1. Create a folder named `oxford_flowers/` under `$DATA`.
2. Download the images from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz.
3. Download the labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat.
4. Download `cat_to_name.json` from [Google Drive](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing).
5. Download `split_zhou_OxfordFlowers.json` from [Google Drive](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing).

The directory structure should look like:
```
oxford_flowers/
|-- cat_to_name.json
|-- imagelabels.mat
|-- jpg/
|-- split_zhou_OxfordFlowers.json
```

### DTD

1. Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `$DATA`. This creates `$DATA/dtd/`.
2. Download `split_zhou_DescribableTextures.json` from [Google Drive](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing).

The directory structure should look like:
```
dtd/
|-- images/
|-- imdb/
|-- labels/
|-- split_zhou_DescribableTextures.json
```

### Caltech101

1. Create a folder named `caltech-101/` under `$DATA`.
2. Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract it under `$DATA/caltech-101`.
3. Download `split_zhou_Caltech101.json` from [Google Drive](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and place it under `$DATA/caltech-101`.

The directory structure should look like:
```
caltech-101/
|-- 101_ObjectCategories/
|-- split_zhou_Caltech101.json
```

### Food101

1. Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract `food-101.tar.gz` under `$DATA`, resulting in `$DATA/food-101/`.
2. Download `split_zhou_Food101.json` from [Google Drive](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like:
```
food-101/
|-- images/
|-- license_agreement.txt
|-- meta/
|-- README.txt
|-- split_zhou_Food101.json
```

### Stanford Cars

1. Create a folder named `stanford_cars/` under `$DATA`.
2. Download the train images from http://ai.stanford.edu/~jkrause/car196/cars_train.tgz.
3. Download the test images from http://ai.stanford.edu/~jkrause/car196/cars_test.tgz.
4. Download the train labels from https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz.
5. Download the test labels from http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat.
6. Download `split_zhou_StanfordCars.json` from [Google Drive](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing).

The directory structure should look like:
```
stanford_cars/
|-- cars_test/
|-- cars_test_annos_withlabels.mat
|-- cars_train/
|-- devkit/
|-- split_zhou_StanfordCars.json
```

### FGVC Aircraft

1. Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
2. Extract `fgvc-aircraft-2013b.tar.gz` and keep only the `data/` folder.
3. Move `data/` to `$DATA` and rename the folder to `fgvc_aircraft/`.

The directory structure should look like:
```
fgvc_aircraft/
|-- images/
|-- ... # a bunch of .txt files
```

### UCF101

1. Create a folder named `ucf101/` under `$DATA`.
2. Download the zip file `UCF-101-midframes.zip` from [Google Drive](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `$DATA/ucf101/`. This zip file contains the extracted middle video frames.
3. Download `split_zhou_UCF101.json` from [Google Drive](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing).

The directory structure should look like:
```
ucf101/
|-- UCF-101-midframes/
|-- split_zhou_UCF101.json
```

### SUN397

1. Create a folder named `sun397/` under `$DATA`.
2. Download the images from http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
3. Download the partitions from https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
4. Extract these files under `$DATA/sun397/`.
5. Download `split_zhou_SUN397.json` from [Google Drive](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing).

The directory structure should look like:
```
sun397/
|-- SUN397/
|-- split_zhou_SUN397.json
|-- ... # a bunch of .txt files
```

### EuroSAT

1. Create a folder named `eurosat/` under `$DATA`.
2. Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `$DATA/eurosat/`.
3. Download `split_zhou_EuroSAT.json` from [Google Drive](https://drive.google.com/file/d/1hYs1oapLOMGKR1WZm0bEutD4KcQyT1R_/view?usp=sharing).

The directory structure should look like:
```
eurosat/
|-- 2750/
|-- split_zhou_EuroSAT.json
```

### CIFAR10/100

For **CIFAR10** and **CIFAR100** datasets, simply run experiments with the dataset - the program will download the data automatically to `$DATA/`.

Alternatively, you can manually download and extract:
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

### ImageNet

1. Create a folder named `imagenet/` under `$DATA`.
2. Download the dataset from http://image-net.org/ (requires registration).
3. Extract the training and validation sets to `$DATA/imagenet/`.

The directory structure should look like:
```
imagenet/
|-- train/  # contains 1000 class folders
|-- val/    # contains 1000 class folders
```

---

## Multi-Domain Datasets

### DomainNet

1. Download the data split files from [Hugging Face](https://huggingface.co/datasets/Jemary/FedBN_Dataset/blob/main/domainnet_dataset.zip) and extract under `$DATA`:
   ```bash
   cd $DATA
   unzip domainnet_dataset.zip
   ```

2. Download the domain-specific datasets and extract them under `$DATA/DomainNet/`:
   - [Clipart](http://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip)
   - [Infograph](http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip)
   - [Painting](http://csr.bu.edu/ftp/visda/2019/multi-source/painting.zip)
   - [Quickdraw](http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip)
   - [Real](http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip)
   - [Sketch](http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip)

   ```bash
   cd $DATA/DomainNet
   unzip clipart.zip
   unzip infograph.zip
   # ... repeat for other domains
   ```

The directory structure should look like:
```
DomainNet/
|-- clipart/
|-- infograph/
|-- painting/
|-- quickdraw/
|-- real/
|-- sketch/
|-- splits/  # split files
```

### Office-Caltech10

Download the pre-processed dataset from [Hugging Face](https://huggingface.co/datasets/Jemary/FedBN_Dataset/blob/main/office_caltech_10_dataset.zip) and extract under `$DATA`:

```bash
cd $DATA
unzip office_caltech_10_dataset.zip
```

The directory structure should look like:
```
office_caltech_10/
|-- amazon/
|-- caltech/
|-- dslr/
|-- webcam/
```

---

## Quick Verification

After preparing the datasets, you can verify your setup by running:

```bash
# Set the dataset path
export COOP_DATASET=/path/to/your/datasets

# Test with a simple dataset (e.g., CIFAR-10 which auto-downloads)
python federated_main.py \
  --root $COOP_DATASET \
  --config-file configs/trainers/FedMGP/base2novel_vit_b16.yaml \
  --dataset-config-file configs/datasets/cifar10.yaml \
  --model FedMGP \
  --trainer FedMGP \
  --debug-mode true
```
