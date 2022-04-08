# SCAUHDM

## Training 

1.Prepare dataset.

    # your dataset structure should be like this
    dataset/
        SCAUHDM/  
            JPEGImages/
                *.jpg
                ...
            Annotations/
                *.xml
                ...
                
2. Run'VOCdevkit/VOC2007/voc2efficientdet.py' to generate the index of data.
    ```
    xmlfilepath=r'your_path\dataset\SCAUHDM\Annotations'
    saveBasePath=r"your_path\dataset\SCAUHDM\ImageSets\Main"
    ```
3. Run'voc_annotation.py' to generate a txt file about label informations.
    ```
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id), encoding='utf-8')
    ...
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
    ```
4. Run 'train'
```
    phi = 2
    ...
    classes_path = 'model_data/helmet_detection_classes.txt'
    ...
    model_path = "model_data/efficientdet-d2.pth"
```

## Test
1.Prepare dataset.

    # your dataset structure should be like this
    VOCdevkit/
        VOC2007/  
            JPEGImages_val/
                *.jpg
                ...
            Annotations_val/
                *.xml
                ...

2. Run'VOCdevkit/VOC2007/voc2efficientdet.py' to generate the index of data.
    ```
    xmlfilepath=r'.\VOCdevkit\VOC2007\Annotations_val'
    saveBasePath=r".\VOCdevkit\VOC2007\ImageSets\Main"
    ```
3. Run'voc_annotation.py' to generate a txt file about label informations.
    ```
    in_file = open('VOCdevkit/VOC%s/Annotations_val/%s.xml'%(year, image_id), encoding='utf-8')
    ...
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages_val/%s.jpg'%(wd, year, image_id))
    
4. Run 'get_gt_txt.py' to generate ground truth information.

5. the file: efficientdet.py
    ```
    _defaults = {
    "model_path": 'model_data/weights.pth',
    "classes_path": 'model_data/helmet_detection_classes.txt',
    "phi": 2,
    "confidence": 0.01,
    "cuda": True
}
    ```

6. Run 'get_dr_txt.py' to generate prediction results.

7. Run 'get_map_coco.py' to evalute by COCO style.

## Reference
https://github.com/bubbliiiing/efficientdet-pytorch
