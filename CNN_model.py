import torch
from torch import nn
from PIL import Image
from torchvision import transforms
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.flatten = nn.Flatten()
        self.dense_layers =  nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self,inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x
def predict(model,image_path,class_mapping):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
  
    image_tensor = transform(img).unsqueeze(0)    
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        return predicted

if __name__ == "__main__":
    model = ConvNet()
    state_dict = torch.load('CONV_NET_v2.pth',map_location=torch.device('cpu'))
    # print(model.load_state_dict(state_dict))
    # print(model)
    img_path = "testing_images/dog.jpg"
    prediction = predict(model,img_path,classes)
    print(prediction)
    