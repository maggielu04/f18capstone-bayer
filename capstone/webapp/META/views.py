from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from webapp.META.models import Document
from webapp.META.forms import DocumentForm

from .models import Document
from webapp.META.test import readJSON
from webapp.META.test import writeJSON
import json
#from .forms import UploadFileForm


def home(request):
    documents = Document.objects.all()
    return render(request, 'META/home.html', { 'documents': documents })

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
        print("here")
        document_name = request.POST.get('name')
        #document = Document.objects.get(document = document_name).
        author2doc = readJSON("media/" + document_name)
        #print(len(author2doc))
        test_output = writeJSON(author2doc, "media/" + document_name)
        print(test_output)
        with open(test_output, encoding='utf-8') as data_file:
            data = json.load(data_file)
        print(type(data))
        print(len(data))
        print(data[0])
        #title = request.POST.get("title", "")
        #print(title)
        #document = open(title, 'w')
        return render(request, 'META/analyse.html', {'document': data})
    else:
        documents = Document.objects.all()
        return render(request, 'META/home.html', { 'documents': documents})