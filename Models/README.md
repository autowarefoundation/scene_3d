![Scene3D GIF](../Media/Scene3D_GIF.gif)

## [visualizations](./visualizations/)
If you want to try out one of the models, then please visit the [visualizations folder](./visualizations/), where you will find examples of how to run model inference on a single image as well as a video

## [model_components](./model_components/)
To understand the architectural layers which comprise each of our models, please enter the [model components folder](./model_components/), in which there are definitions of the various layers and modular blocks which we have used to construct our custom models

## [training](./training/)
In the [training folder](./training/), we share our open-source model training pipelines which are configured to work with open-source data. You retrain our models on open-source data or modify the training pipelines to use your custom data. Each model has a "trainer" class which contains all of the main training implementation such as model loading, loss function definition, applying augmentations, loss calculation, and logging to TensorBoard. Each model also has a "train" script which calls functions from the "trainer" class and is responsible for orchestrating the training using various datasets.

## [data_parsing](./data_parsing/)
We utilize open-source datasets to train our models. Since each open-source dataset may have a different ground truth format, we provide various dataset parsing scripts which are used to prepare our custom ground truth datasets specific for the learning task of each model. You can learn more by visiting the [data parsing folder](./data_parsing/)

## [data_utils](./data_utils/)
The [data utils folder](./data_utils/) provides data loaders for each model which help parse and load the specific open-source datasets used to train that model. This folder also contains an augmentations class which applies various image noise transforms to make models more robust and generalizable during training. It also provides other simple data helper classes.

## [exports](./exports/)
Within the [exports folder](./exports/), you can find more information about the 'Quanty' open-source model quantization framework which can compress large model files into smaller models while preserving model inference accuracy. This folder also contains simple utilities for converting model formats.

## [inference](./inference/)
In the [inference folder](./inference/), there are helper classes which are used to perform inference with pre-trained models.





