import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
img = 'https://www.mcall.com/resizer/sf69RLNG2NL-uYWrPzcD5NvmR_g=/415x233/top/cloudfront-us-east-1.images.arcpublishing.com/tronc/4SHUTDKWUJBQXALP7LBBUUGSXM.jpg'

# Inference
results = model(img)

# Results
results.print()