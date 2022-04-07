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

3. Run'voc_annotation.py' to generate a txt file about label informations.

4. Run 'train'


## Reference
https://github.com/bubbliiiing/efficientdet-pytorch
