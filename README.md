# ATT-UNET: DBT breast lesions

## Introduction:
Trained Attention Unet for DBT semi-automatic segmentation on a dataset of 73 women over the age of 40, according to the P.I.N.K study protocol.

The model requires 2D images cropped around the lesion and provides 5 different predictions (one for each trained model) of the lesion.
The predictions are coupled with the corresponding 2D attention map.

An average segmentation is provided for each crop as a combined (more stable) result.

## Run inference:
Inference can be run from console by calling:
python att_unet_dbt_inference.py test_images models prediction

## Trained models:
The model structure can be found in *model.py*.
Model weights can be found in the *models* folder.
