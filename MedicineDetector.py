#MedicineDetector with YOLOv5
#clone YOLOv5 and
"""
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow
"""
import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")

# set up environment
os.environ["DATASET_DIRECTORY"] = "/content/datasets"

#Get the dataset from Roboflow
rf = Roboflow(api_key="Owb7wL0INQDuAuAz9gth")
project = rf.workspace().project("medicine-detector")
dataset = project.version(1).download("yolov5")

#Train the Yolo Model
#Run
"""
!python train.py --img 416 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
"""

#Show Training Results
# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
"""
%load_ext tensorboard
%tensorboard --logdir runs

"""

#Prediction
"""
!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.2 --source {dataset.location}/valid/images
"""

#display inference on ALL test images

import glob
from IPython.display import Image, display

i = 0
# Choose the correct exp folder - see prev output block
for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    i += 1

    if i < 8:
      display(Image(filename=imageName))
      print("\n")


#Save Model
#export your model's weights for future use
from google.colab import files
files.download('./runs/train/exp/weights/best.pt')
