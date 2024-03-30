import torch
from skimage import io

from model import CNN

def inference(PATH=""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()
    model.load_state_dict(torch.load("Sleeping-Pose-Detector/state_dict_model.pt"))

    model.to(device)

    with torch.no_grad():
        image = io.imread(PATH)
        logits = model.forward(image)
        pred = torch.argmax(logits).item()
    
    return pred
