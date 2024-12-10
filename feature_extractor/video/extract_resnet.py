import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as tfn

def get_featsdim(variant='resnet50'):
    variant='resnet50'
    if variant == 'resnet18':
        return 512 
    elif variant == 'resnet50':
        return 2048 
        
def instantiate_resnet(variant='resnet50'):
    # Load the pretrained model
    if variant == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif variant == 'resnet50':
        model = models.resnet50(pretrained=True)
    # Strip the last layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    # Set model to evaluation mode
    model.eval()
    return model, feature_extractor

def preproc_image_for_resnet(input_image):
    input_image = tfn.to_pil_image(input_image)
    # 1
    preprocess = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
    # 2
    # preprocess = transforms.Compose([
    #                                     transforms.Scale((224, 224)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                                 ])
    
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch

def extract_resnet_feats(model, image, preproc=False):
    if preproc:
        # 1. preprocess image for resnet, image expected as [channel, height, width]
        image = preproc_image_for_resnet(image)
    embed = model(image).squeeze()
    return embed