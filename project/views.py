import glob
import torch
import pathlib
import datetime
import subprocess

from django.views import View
from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render, reverse, redirect, get_object_or_404

from .net import *
from .models import ImageModel
from .forms import ImageForm

class ProjectView(View):
    def get(self, request):
        images = ImageModel.objects.all()
        form = ImageForm()

        context = {
            "images": images,
            "form": form
        }
        return render(request, "index.html", context=context)

    def post(self, request):
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            new_image = ImageModel(image=request.FILES['imagefile'])
            new_image.save()

            image_path = ["media/" + str(new_image.image)]

            test_dataset = PlantDataset(image_path)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

            output, pred = test(model, test_loader)
            pred = "healty" if pred.cpu().numpy()[0][0] == 0 else "infected"

            outputType, predType = test(modelType, test_loader)
            predType = leaf_types_map_inv[predType.cpu().numpy()[0][0]]

            result = subprocess.Popen(
                [
                    'python', 'project/static/yolov5/detect.py',
                    '--weights', 'project/static/best.pt',
                    '--project', 'media/image',
                    '--img', '256',
                    '--conf', '0.4',
                    '--source', image_path[0]
                ],
                shell=True,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            output, error = result.communicate()
            
            f1 = "Results saved to "
            i1 = output.find(f1)
            i2 = output.find("\nDone. (")
            bbox_dir = output[i1 + len(f1):i2]
            
            f1 = "256x256 "
            i1 = output.find(f1)
            i2 = output.find(", Done. (")
            YOLOleafType = output[i1 + len(f1):i2].split(" ")[1]

            images = ImageModel.objects.all()
            form = ImageForm()

            context = {
                "images": ImageModel.objects.filter(image=new_image.image),
                "form": form,
                "pred": pred,
                "predType": predType,
                "bbox": glob.glob(bbox_dir + "/*.*")[0],
                "YOLOleafType": YOLOleafType
            }
            return render(request, "index.html", context=context)
        else:
            form = ImageForm()
