import torch
from torchvision import models, transforms
from PIL import Image

#lets load pretrained model

model = models.resnet50(pretrained = True)
model.eval


with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]


transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

    ]
)

image_path = "fire.jpg"
image = Image.open(image_path).convert("RGB")
img_tensor = transform(image).unsqueeze(0)



# Predict
with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top3_prob, top3_idx = torch.topk(probabilities, 3)

# Print top 3 predictions
for i in range(3):
    print(f"Prediction {i+1}: {classes[top3_idx[i]]} ({top3_prob[i].item()*100:.2f}%)")