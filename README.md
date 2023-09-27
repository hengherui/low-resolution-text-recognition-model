# low-resolution-text-recognition-model


### Contributions

* We put forward an efficient multi-task network that can jointly handle low-resolution text recognition. We propose a multi-task learning approach for scene text recognition, which includes a text recognition branch and a super-resolution branch. The proposed super-resolution branch, incorporating residual super-resolution units, effectively captures rich information from low-resolution features. It is utilized to generate high-resolution text features by contrastive learning between high-resolution and low-resolution features.

* We conduct large number of experiments on real logistics express sheet images, which are not represented on our paper, here, we show these recognition results, and we mosaicked the sensitive information such as name and telephone num for preventing disclosure of private customer information during the presentation. We show different logistics scenes, including the blurred, tilted, dark background, low-resolution images, all these scenes can prove that our method have robustness on multi scenes of logistics express image text recognition. The details have shown as followed.

## training enviroment
```setup
Tesla V100 32G memory, 8 cards.
```
```setup
install required package "pip install -r requirments"
```
```setup
You shoul prepare GPU equipment and the CUDA envoirment and the tensorrt for speed acceleration.
```
```setup
Then loding the training dataset including 11 million text image data set.
```
```setup
Running the code in docker environment or visual envoirment for steady training
```

## Data preparation

We give an example to construct your own datasets. Details please refer to `tools/create_svtp_lmdb.py`.
We provide datasets for [training](https://pan.baidu.com/s/1BMYb93u4gW_3GJdjBWSCSw&shfl=sharepset) (password: wi05) and [testing](https://drive.google.com/open?id=1U4mGLlsm9Ade1-gQOyd6He5R0yiaafYJ).

### Logistics text recognition Result

#### Example 1, the area inside the blue box is the detected text region in the left side, the left side is the detected logistics sheet image, and the right side is the recognition result corresponding to the text area one-to-one. Although the image is blurred, but our model gives the correct results.
<img src="https://github.com/hengherui/Low-resolution-Text-Recognition/blob/master/Results/1.jpg" width="500px">

#### Example 2, due to the dark light in the distribution centre, the background of the image is dark and the picture is in a tilted position, but our model can also perform well.
<img src="https://github.com/hengherui/Low-resolution-Text-Recognition/blob/master/Results/5.jpg" width="500px">

#### Example 3, the image is very dirty and  is in low-resolution condition, but our model gives correctly results after distinguishing carefully. 
<img src="https://github.com/hengherui/Low-resolution-Text-Recognition/blob/master/Results/12.jpg" width="500px">

#### Example 4, the image is in low-resolution condition, but our method performs well and predicts correctly. 
<img src="https://github.com/hengherui/Low-resolution-Text-Recognition/blob/master/Results/7.jpg" width="500px">



### Requirement

This codebase has been developed with python version 3.7, PyTorch 1.7+ and torchvision 0.8+:
```setup
pip install -r requirements.txt
```
### Train

bash scripts/stn_att_rec.sh

### Test

bash scripts/main_test_image.sh
