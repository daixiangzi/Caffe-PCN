# Caffe-PCN
Detect multi angle face by pcn ,and Crop detected face  
Paper:https://arxiv.org/pdf/1804.06039.pdf  
Original repo:https://github.com/Jack-CV/FaceKit/tree/master/PCN
# Install 
caffe  
python-opencv  
numpy
# Usage
cd Caffe-PCN  
python pcn_caffe.py  
# Result
on CPU(i5):7-9 fps  
on GPU(1080ti):28-31 fps  
test image:  
![](https://github.com/daixiangzi/Caffe-PCN/tree/master/results/test.jpg)  
Crop face:  
![](https://github.com/daixiangzi/Caffe-PCN/tree/master/results/crop.jpg)

