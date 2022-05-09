from django.shortcuts import render  
from yolo.forms import UserImage as UserImageForm
from yolo.yolov5 import yolov5
from django.core.files.storage import FileSystemStorage
  
def image_request(request):  
    if request.method == 'POST':  
        form = UserImageForm(request.POST, request.FILES)  
        if form.is_valid():  
            form.save()  
            imgObject = form.instance 
            path = imgObject.image.path
            conf = imgObject.confValue
            iou = imgObject.iouValue
            imgLabel = yolov5.imageProcess(path, conf, iou)
            imgLabel.index += 1
            imgLabel = imgLabel.to_html(classes=["table-bordered", "table-striped", "table-hover"], justify="center")
            fss = FileSystemStorage()
            imgPathNew = fss.url('/images/processed/processed.png')
            return render(request, 'index.html', {'form': form, 'imgOriginal': imgObject, 'imgProcessed': imgPathNew, 'imgLabel': imgLabel})  
    else:  
        form = UserImageForm()  
    return render(request, 'index.html', {'form': form})  