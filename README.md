# SCAUHDM

## Training 

1.Prepare dataset.

    # your dataset structure should be like this
    VOCdevkit/
        VOC2007/  
            JPEGImages/
                *.jpg
                ...
            Annotations/
                *.xml
                ...
                
2. Run'VOCdevkit/VOC2007/voc2efficientdet.py' to generate the index of data.
    ```
    xmlfilepath=r'.\VOCdevkit\VOC2007\Annotations'
    ...
    saveBasePath=r".\VOCdevkit\VOC2007\ImageSets\Main"
    ```
3. Run'voc_annotation.py' to generate a txt file about label informations.
    ```
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id), encoding='utf-8')
    ...
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages_val/%s.jpg'%(wd, year, image_id))
    ```
4. Run 'train'
```
    phi = 2
    ...
    classes_path = 'model_data/helmet_detection_classes.txt'
    ...
    model_path = "model_data/efficientdet-d2.pth"
```

## Reference
https://github.com/bubbliiiing/efficientdet-pytorch
