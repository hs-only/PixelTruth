import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from .forms import VideoUploadForm
from django.conf import settings
import cv2
import face_recognition
from torchvision import transforms
import torch
from .models import Model3
from .prediction import classify_video

@csrf_exempt
def check(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data['video']
            
            # Save the video file to a temporary location
            video_path = save_uploaded_video(video_file)
            
            # Process the video using your deepfake detection model
            model_path = os.path.join(settings.BASE_DIR,'check', 'lstm-model','model3.pt')
            result = make_prediction(video_path,model_path)
            
            try:
                os.unlink(video_path)
                print(f"File '{video_path}' has been unlinked successfully.")
            except FileNotFoundError:
                print(f"File '{video_path}' does not exist.")

            # Return the result as JSON
            return JsonResponse({'result': result})
        else:
            return JsonResponse({'error': 'Invalid form data'}, status=400)
    else:
        form = VideoUploadForm()
        
    template = loader.get_template('check_page.html')
    context = {'form': form}
    return HttpResponse(template.render(context, request))

def save_uploaded_video(file):
    # Define the path to save the uploaded video
    video_path = os.path.join(settings.MEDIA_ROOT, 'temp_videos', file.name)
    
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Save the video file
    with open(video_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    return video_path

def make_prediction(video_path,model_path, num_classes=2,device='cuda'):

    predicted_class, probabilities= classify_video(video_path, model_path, num_classes, device='cpu')
    target ='REAL' if predicted_class==1 else 'FAKE'
    if target=='REAL':
        probabilities=probabilities[0][1]
    else:
        probabilities=probabilities[0][0]
    probabilities=round(probabilities,4)
    return {'probability': probabilities, 'target': target}

