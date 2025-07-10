from CNN_model import ConvNet,classes
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms

def predict(model,img,class_mapping):
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

model = ConvNet()
state_dict = torch.load("CONV_NET_v2.pth",map_location=torch.device('cpu'))
model.load_state_dict(state_dict)


app = FastAPI()

@app.get("/")
def root():
    return {"Ferrari":"Shit"}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    prediction = predict(model,image,classes)
    return JSONResponse(content={"Prediction": prediction})
