import os
import sys
import argparse
import numpy as np
from PIL import Image
from model import AttentionUNet, ModelWrapper
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#%% Call the parser to receive the input path

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_img", help="path to the image dataset folder",
                type=str)
    parser.add_argument("path_models", help="path to the models dataset folder",
                        type=str)
    args = parser.parse_args()
except:
    e = sys.exc_info()[0]

path_img = os.path.join(os.getcwd(),str(sys.argv[1]))
path_model = os.path.join(os.getcwd(),str(sys.argv[2]))

'''
path_img = os.path.join(os.getcwd(), "test_images")
path_model = os.path.join(os.getcwd(),"models")
'''
print(path_img)
list_path_images = [os.path.join(path_img, i) for i in os.listdir(path_img)]

transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]

    ])

list_images = [transform(Image.open(i)).unsqueeze(0) for i in list_path_images]


list_path_models = [os.path.join(path_model, i) for i in os.listdir(path_model)]

device = torch.device("cpu")
list_model = [AttentionUNet().to(device) for i in list_path_models]


for i,path_i in enumerate(list_path_models):
    try:
        list_model[i].load_state_dict(torch.load(path_i, map_location={'cuda:0':'cpu'}))
    except:
        list_model[i].load_state_dict(torch.load(path_i, map_location="cpu"))

## MANCA PREPROCESSING

for file_path, i in enumerate(list_images):
    filename = os.listdir(path_img)[file_path].split(".")[0]

    pred_avg = np.zeros((256,256))
    for round,m in enumerate(list_model):
        m.eval()
        with torch.no_grad():
            pred, attention = m.forward(i)
        pred_avg = pred_avg + np.array(pred)
        pred = (pred > 0.5).float()
        pred = pred.cpu().numpy()[0,0,:,:]

        pred = (pred * 255).astype(np.uint8)  # Scale binary values to uint8
        
        #attention_img = transforms.ToPILImage()(attention['att1'])
        #attention['att1'] = (attention['att1'] > 0.5).float()

        attention_img = attention['att1'] # Select the first time step

        attention_mask_resized = torch.nn.functional.interpolate(attention_img, size=(256, 256),
                                               mode='bilinear', align_corners=False)

        # Example usage:
        # Single channel
        #plot_attention(images[0], attention_mask_resized[0], original_image=images[0], mode='single', channel=0)

        attention_mask_np = attention_mask_resized[0][0].cpu().detach().numpy()

        attention_mask_np = (attention_mask_np - attention_mask_np.min()) / (
                attention_mask_np.max() - attention_mask_np.min())

        #for idx in range(attention_img.shape[0]):
        #    attention_img_single = attention_img[idx]
        #    attention_img_pil = Image.fromarray(np.array(attention_img_single), mode='RGB')
        #    attention_img_pil.save(f'test_PREDS/{filename}_attention_layer_{idx}.png')

        jet_colormap = cm.get_cmap('jet')  # Get the 'jet' colormap
        colored_image = jet_colormap(attention_mask_np)  # Apply the colormap to the data

        # Convert the colormap output (RGBA) to an 8-bit image
        # The colormap result has RGBA format (values between 0 and 1), so we scale to 255
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # Remove alpha and scale

        image = Image.fromarray(np.array(pred), mode='L')
        attention_img_pil = Image.fromarray(colored_image)
        image.save("test_PREDS/"+str(filename)+"_"+str(round)+".png")
        attention_img_pil.save("test_PREDS/" + str(filename) + "_attention_" + str(round) + ".png")

    pred_avg = pred_avg/5
    pred_avg = torch.from_numpy(pred_avg)
    pred_avg = (pred_avg > 0.5).float()
    pred_avg = pred_avg.cpu().numpy()[0, 0, :, :]

    pred_avg = (pred_avg * 255).astype(np.uint8)

    pred_avg_image = Image.fromarray(np.array(pred_avg), mode='L')
    pred_avg_image.save("test_PREDS/" + str(filename) + "_avg.png")