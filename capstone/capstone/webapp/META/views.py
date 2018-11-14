from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from webapp.META.models import Document
from webapp.META.forms import DocumentForm


def home(request):
    documents = Document.objects.all()
    return render(request, 'META/home.html', { 'documents': documents })

'''
def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'META/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'META/simple_upload.html')
'''

def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'META/model_form_upload.html', {
        'form': form
    })

def analyse_document(request):
    if request.method == "POST":
        document = request.FILES
        return render(request, 'META/temp.html', {
            'document': document
        })
