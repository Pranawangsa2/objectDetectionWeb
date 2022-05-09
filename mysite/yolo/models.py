from django.db import models

# Create your models here.
def validatorValue(value):
        if value < 0.1: value = 0.1
        elif value > 1.0: value = 1.0
        else: pass
        return value

class UploadImage(models.Model):  
    image = models.ImageField(upload_to='images')
    confValue = models.FloatField(verbose_name="Confidence", default=0.3, validators=[validatorValue])
    iouValue = models.FloatField(verbose_name="IoU", default=0.3, validators=[validatorValue])  



             