# Vehicle Detection

## Detection multiple vehicle in the image context using R-CNN and Faster R-CNN.

### Getting started

#### Installing required packages
'pip3 install -r requirements.txt'

#### Training RCNN model
1. By default it used provided [annotated_dataset](annotated_dataset/ideal/). You can also provide path to your own dataset.

'python3 vehicle_detection.py train --subset=ideal'

2. By default it used provided [annotated_dataset](annotated_dataset/non_ideal/). You can also provide path to your own dataset. 

'python3 vehicle_detection.py train --subset=non_ideal'

#### Testing trained model on an image
'python3 vehicle_detection.py run --image=/path/to/image.jpg'