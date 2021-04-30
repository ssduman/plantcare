from django.db import models

# https://stackoverflow.com/questions/5871730/how-to-upload-a-file-in-django
class ImageModel(models.Model):
    image = models.FileField(upload_to='image/%Y/%m/%d')
