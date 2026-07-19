from django.urls import path
from . import views

urlpatterns = [
    # API endpoints
    path('detect/', views.detect, name='detect'),
    path('add-to-cart/', views.add_to_cart, name='add_to_cart'),
    path('cart/<int:transaction_id>/', views.cart, name='cart'),
    path('latest-transaction/', views.latest_transaction, name='latest_transaction'),
    path('checkout-complete/', views.checkout_complete, name='checkout_complete'),
    
    # Video feed
    path('video-feed/', views.video_feed, name='video_feed'),
    
    # Pages
    path('', views.checkout, name='checkout'),
]
