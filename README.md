# Pose-SPI

Model for human keypoint detection based on grayscale images.
2 variations of the model are presented:
 - model with the simple keypoint detection head, consisting of 2 convolutions, 2 average poolings and non-linearities
 - model with the [efficientnet](https://arxiv.org/abs/1905.11946) pretrained backbone as feature extractor

For the rest, both models consist of
 * encoder,
 * FSRCNN model for upsampling,
 * keypoint detector head in 2 variations as described above.

## Requirements
  for installing required packages run
  
` pip install -r requirements.txt`

If the tensorboard is missing from the install packages, run separately

`pip install tensorboard`

## Training
To run training of simple keypoint detector, run

`python train.py`

it will run the training with the default arguments (64x64 images, 1000 epochs...) on the toydataset.

To run training of the efficientnet-based model, run

`python train_keypointrcnn.py`

Again, it will run with the default arguments, which can be further changed.

To see training performance in real-time, a tensorboard can be used. During    the training, in a separate terminal run

`tensorboard --logdir tensorboard_logs`

## Additional Points
Below are some recommendations and code-related notes to be taken into account:

### Notes
 - hyperparameters in the default settings are chosen to show good results on the current dataset. When changing the data, the learning rate, number of epochs and batch size should be adjusted accordingly.
 - both models contain fully connected layers in which the number of input features depends on the input image size and should be corrected if the size of the input will be different from 64x64.
 - coco-evaluation contains empirically calculated scalars in **kpt_oks_sigmas**, so the correct result for the evaluation can be achieved with the original 17 keypoint setting or when only the correct values in **kpt_oks_sigmas** are kept.
### Recommendations
- training of the small models usually is better on small batch sizes. Small batch size adds additional noise during the training and make the model more robust.
- when changing the architecture or dataset, first it is better to try to overfit the model on small subset of data to be sure that there is no bugs in code and that the transformation that we want to teach the model can be learned.
   

