from django.conf.urls import url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

from webapp.META import views


urlpatterns = [
    url(r'^$', views.home, name='home'),
    #url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
    url(r'^META/form/$', views.model_form_upload, name='model_form_upload'),
    url(r'^META/analyse/$', views.analyse_document, name='analyse'),
    url(r'^admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
