#%%  
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchaudio
#%%
import cv2
#print(cv2.getBuildInformation())
#%%
print("torch version:", torch.__version__)
print(torch.cuda.is_available())
print("torchvision version:", torchvision.__version__)
print("torchaudio version:", torchaudio.__version__)
#%%

def visualize_images(image: np.ndarray=None, image_type: str = "RGB"):
    
    
    fig1, axs = plt.subplots(1, 4, figsize=(10, 4))
    axs[0].imshow(image)
    axs[0].set_title(image_type)
    axs[1].imshow(image[:,:,0], cmap='gray')
    axs[1].set_title(image_type[0])
    axs[2].imshow(image[:, :, 1], cmap='gray')
    axs[2].set_title(image_type[1])
    axs[3].imshow(image[:, :, 2], cmap='gray')
    axs[3].set_title(image_type[2])

    for ax in axs:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show() 
#%%
img = cv.imread("./Images/s1808.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
mask = (img[:,:,0] > 0) & (img[:,:,1] > 0)
print(f"R mean : {img[:,:,0][mask].mean()}")
print(f"G mean : {img[:,:,1][mask].mean()}")
visualize_images(img, 'RGB')

#%%
img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
print("hsv type : ", img_hsv.dtype)
visualize_images(img_hsv, 'HSV')

#%%

# Affichage de l'image en RGB et Lab
img_Lab = cv.cvtColor(img, cv.COLOR_RGB2Lab)
print("lab_type : ", img_Lab.dtype)
visualize_images(img_Lab, 'Lab')

#%% 
# Égalisation d'histogramme simple
img_hsv_eq = img_hsv.copy()
img_hsv_eq [:,:,2] = cv.equalizeHist(img_hsv_eq [:,:,2])
img_eq = cv.cvtColor(img_hsv_eq , cv.COLOR_HSV2RGB)
mask = (img_eq[:,:,0] > 0) & (img_eq[:,:,1] > 0)
print(f"R mean : {img_eq[:,:,0][mask].mean()}")
print(f"G mean : {img_eq[:,:,1][mask].mean()}")
visualize_images(img_eq, 'RGB Equalized')

#%%
# Méthode CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))  # Paramètres ajustables
img_hsv_eq = img_hsv.copy()  
img_hsv_eq[:,:,2] = clahe.apply(img_hsv_eq[:,:,2])  # Appliquer CLAHE sur V
img_eq2 = cv.cvtColor(img_hsv_eq, cv.COLOR_HSV2RGB)  # Convertir en RGB
mask = (img_eq2[:,:,0] > 3) & (img_eq2[:,:,1] > 3)
print(f"R mean : {img_eq2[:,:,0][mask].mean()}")
print(f"G mean : {img_eq2[:,:,1][mask].mean()}")
visualize_images(img_eq2, 'RGB CLAHE')