# Visual-Explanations-from-Spiking-Neural-Networks-using-Interspike-Intervals
Kim, Y. & Panda, P., Visual explanations from spiking neural networks using inter-spike intervals. Sci Rep 11, 19037 (2021). https://doi.org/10.1038/s41598-021-98448-0    

### Training

*  Download tinyimagenet dataset to ```PATH/TO/DATASET```  
*  Train a model using BNTT (https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time).

### Testing (on pretrained model)

* As a first step, download pretrained parameters ([link][e]) to ```PATH/TO/MODEL.pth.tar```   

[e]: https://drive.google.com/file/d/11ybFXqRB3edxsFMUrPrwwH5_HlAAHNUp/view?usp=sharing

* The above pretrained model is for TinyImageNet / VGG11 architecture

* Run the following command

```
python main.py --pretrainedmodel_pth 'PATH/TO/MODEL' --dataset_pth 'PATH/TO/DATASET' --target_layer 6
```

*  Heatmaps (across timesteps) are visualized in folder ```figuresave```
*  In order to change the target images, please change the list in line 32. 

   Example) visualize 10th and 52th images in TinyImageNet validation dataset.  

```
img_nums = [10, 52]
```


