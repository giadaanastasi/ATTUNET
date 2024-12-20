'''
Inference script that applies the trained pytorch Attention-UNet models to the 
images. The script returns the prediction (and corresponding attention map) for 
each input image and model. 

20-12-2024

Giada Anastasi, Giulio Del Corso, Daniela Gasperini
'''


#%% Import libraries
import os
import sys
import argparse
import numpy as np
from PIL import Image
from model_structure import AttentionUNet
import torch
from torchvision import transforms
import matplotlib
import shutil



#%% Image pre processing:
transform = transforms.Compose([
        transforms.Resize((256, 256)),                  # Resize to 256x256
        transforms.ToTensor(),                          # Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])     # Normalization
    ])



#%% Define the inference device (CPU)
device = torch.device("cpu")



#%% Call the parser to receive the input path to the images/trained models
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_img", help="path to the image dataset folder",
                type=str)
    parser.add_argument("path_models", help="path to the models dataset folder",
                        type=str)
    parser.add_argument("path_prediction", help="path to the prediction folder",
                        type=str)
    args = parser.parse_args()
except:
    e = sys.exc_info()[0]



#%% Define the path to the images and the models
path_img = os.path.join(os.getcwd(),str(sys.argv[1]))
path_model = os.path.join(os.getcwd(),str(sys.argv[2]))
path_prediction = os.path.join(os.getcwd(),str(sys.argv[3]))
    


#%% Check if save folder is empty and create it
if os.path.isdir(path_prediction):
    shutil.rmtree(path_prediction)
    os.mkdir(path_prediction)
else:    
    os.mkdir(path_prediction)



#%% Define the list of all images to be evaluated
list_path_images = [os.path.join(path_img, i) for i in os.listdir(path_img)]
list_images = [transform(Image.open(i)).unsqueeze(0) for i in list_path_images]

print("\nFound "+str(len(list_path_images))+" unique images.")



#%% Define the list of possible models
list_path_models = [os.path.join(path_model, i) for i in os.listdir(path_model)]
list_model = [AttentionUNet().to(device) for i in list_path_models]

print("\nFound "+str(len(list_path_models))+" different models.\n")



#%% Load the trained models
for i,path_i in enumerate(list_path_models):
    try:
        list_model[i].load_state_dict(torch.load(path_i, map_location={'cuda:0':'cpu'}))
    except:
        list_model[i].load_state_dict(torch.load(path_i, map_location="cpu"))


#%% Iterate on each image
print("\nStart inference:")
for i_count, tmp_img in enumerate(list_images):
    print("Percentage: " +str(100*i_count/len(list_images))+"%")
    
    filename = os.listdir(path_img)[i_count].split(".")[0]
    
    pred_avg = np.zeros((256,256))          # Initialize the average prediction
    
    # Iterate on each model
    for round,m in enumerate(list_model):
        
        m.eval()                            # Switch to eval()
        
        with torch.no_grad():
            pred, attention = m.forward(tmp_img)  # Prediction and attention map
            
        pred_avg = pred_avg + np.array(pred)# Update average
        
        pred = (pred > 0.5).float()         # Binarization
        pred = pred.cpu().numpy()[0,0,:,:]
        pred = (pred * 255).astype(np.uint8)# Scale binary values to uint8
        
        
        attention_img = attention['att1'] 
        attention_mask_resized = torch.nn.functional.interpolate(attention_img, 
                                                               size=(256, 256),
                                          mode='bilinear', align_corners=False)
        attention_mask_np = attention_mask_resized[0][0].cpu().detach().numpy()
        attention_mask_np = (attention_mask_np - attention_mask_np.min()) / (
                attention_mask_np.max() - attention_mask_np.min())
        jet_colormap = matplotlib.colormaps['jet']     # Get the 'jet' colormap

        colored_image = jet_colormap(attention_mask_np)# Apply the colormap 
        
        # Remove alpha and scale to uint8
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  
        image = Image.fromarray(np.array(pred), mode='L')
        attention_img_pil = Image.fromarray(colored_image)

        tmp_img_save_path_prediction = os.path.join(path_prediction, 
                                           str(filename)+"_"+str(round)+".png")
        tmp_att_save_path_prediction = os.path.join(path_prediction, 
                                 str(filename)+"_attention_"+str(round)+".png")
        
        image.save(tmp_img_save_path_prediction)
        attention_img_pil.save(tmp_att_save_path_prediction)
        
    pred_avg = pred_avg/len(list_model)
    pred_avg = torch.from_numpy(pred_avg)
    pred_avg = (pred_avg > 0.5).float()
    pred_avg = pred_avg.cpu().numpy()[0, 0, :, :]

    pred_avg = (pred_avg * 255).astype(np.uint8)    
    
    tmp_img_avg_save_path_prediction = os.path.join(path_prediction, 
                                       str(filename)+"_avg_"+str(round)+".png")
    
    pred_avg_image = Image.fromarray(np.array(pred_avg), mode='L')
    pred_avg_image.save(tmp_img_avg_save_path_prediction)

print("Percentage: 100%")