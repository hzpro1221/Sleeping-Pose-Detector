import torch
from skimage import io

from model import CNN
from args import parse_arguments

def inference(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()
    model.load_state_dict(torch.load("Sleeping-Pose-Detector/state_dict_model.pt"))

    model.to(device)

    with torch.no_grad():
        image = io.imread(args.PATH)
        logits = model.forward(image)
        pred = torch.argmax(logits).item()
    
    return pred

if __name__ == '__main__':
    args = parse_arguments()
    pred = inference(args)
    if (pred==0):
        print("Tư thế ngủ không xấu")
    if (pred==1):
        print("Tư thế ngủ bào thai")
    if (pred==2):
        print("Nằm ngửa hai tay giơ lên đầu")
    if (pred==3):
        print("Ngủ sấp")
    if (pred==4):
        print("Ngủ tựa đầu lên cánh tay")