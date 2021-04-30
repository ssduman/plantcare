import json
import torch
import datetime

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

        # return JsonResponse(list(vehicles), safe=False)

    def post(self, request):
        # data = json.loads(request.body)

        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            new_image = ImageModel(image=request.FILES['imagefile'])
            new_image.save()

            image_path = ["media/" + str(new_image.image)]
            
            test_dataset = PlantDataset(image_path)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            output, pred = test(model, test_loader)
            pred = "healty" if pred.cpu().numpy()[0][0] == 0 else "infected"

            images = ImageModel.objects.all()
            form = ImageForm()

            context = {
                "images": ImageModel.objects.filter(image=new_image.image),
                "form": form,
                "pred": pred
            }
            return render(request, "index.html", context=context)

            # return HttpResponseRedirect(reverse('project'))
        else:
            form = ImageForm()

        # return JsonResponse(list(data), safe=False)
