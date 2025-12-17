from django.urls import path
from . import views

urlpatterns = [
    path("", views.upload_page, name="upload-page"),
    path("upload/", views.upload_api, name="upload-api"),
    path("forecast/", views.index, name="forecast-index"),
    path("api/products/", views.products_api, name="api-products"),
]
