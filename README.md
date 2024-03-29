# The code of Detection Network for multi-size and multi-target tea bud leaves in the field of view

## Introduction
We have provided PyTorch-based YOLOv7 code along with modules for multi-object, multi-scale detection. The SKAttention module, multi-feature fusion module, and loss function module have been designed in a more user-friendly format and placed within the "modules" folder. These can be inserted into the code files in the "nets" folder, allowing for flexible experimentation with more possibilities in other similar research endeavors.

## Prerequisites
To reference the functionality library, please consult the 'requirements.txt' file.

## Yolov7 pre-training weight
The weight required for pre-training can be downloaded in Baidu web disk.  
link: https://pan.baidu.com/s/1uYpjWC1uOo3Q-klpUEy9LQ     
code: pmua    

## Training model
The original model is YOLOv7, and the functional modules are in Tea_detection/modules. You can incorporate improvements into the respective backbone，yolo and yolo_training in "nets" folder.
The file used for training is 'train.py'

## Video demo
```
https://www.bilibili.com/video/BV1A4421A72f/?spm_id_from=333.999.0.0&vd_source=d787a1c2da2e295508a37e23b2000370
```

## Cite tea detection network
```
@article{
  title = {Detection Network for multi-size and multi-target tea bud leaves in the field of view via improved YOLOv7},
  journal = {Computers and Electronics in Agriculture},
  pages = {108700},
  volume = {218},
  year = {2024},
  issn = {0168-1699},
  doi = {https://doi.org/10.1016/j.compag.2024.108700},
  author = {Tianci Chen, Haoxin Li, Jiazheng Chen, Zhiheng Zeng, Chongyang Han, Weibin Wu}
}
```

## Reference
https://github.com/WongKinYiu/yolov7  
https://github.com/augmentedstartups/yolov7  
https://github.com/AshesBen/citrus-detection-localization  
https://github.com/augmentedstartups  
