from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils.timezone import now
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import json
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime

from .models import Product, Transaction, TransactionItem


@csrf_exempt
@require_http_methods(["POST"])
def detect(request):
    """
    API endpoint: /api/detect/
    Receives detected label and confidence from detector.py
    Returns product information if found
    """
    try:
        data = json.loads(request.body)
        label = data.get('label')
        confidence = data.get('confidence', 0)
        
        if not label:
            return JsonResponse(
                {'error': 'Missing label'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Find product by model_label
        product = Product.objects.filter(
            model_label=label,
            active=True
        ).first()
        
        if not product:
            return JsonResponse(
                {'error': f'Product not found for label: {label}'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        return JsonResponse({
            'id': product.id,
            'name': product.name,
            'price': float(product.price),
            'label': product.model_label,
            'confidence': confidence
        }, status=status.HTTP_200_OK)
        
    except json.JSONDecodeError:
        return JsonResponse(
            {'error': 'Invalid JSON'},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return JsonResponse(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@csrf_exempt
@require_http_methods(["POST"])
def add_to_cart(request):
    """
    API endpoint: /api/add-to-cart/
    Adds an item to the current transaction (cart)
    """
    try:
        data = json.loads(request.body)
        transaction_id = data.get('transaction_id')
        product_id = data.get('product_id')
        quantity = data.get('quantity', 1)
        unit_price = data.get('unit_price')
        
        # Validate inputs
        if not transaction_id or not product_id:
            return JsonResponse(
                {'error': 'Missing transaction_id or product_id'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get transaction
        transaction = Transaction.objects.get(id=transaction_id)
        
        # Get product
        product = Product.objects.get(id=product_id)
        
        # Create or update cart item
        cart_item, created = TransactionItem.objects.get_or_create(
            transaction=transaction,
            product=product,
            defaults={
                'quantity': quantity,
                'unit_price': unit_price or product.price
            }
        )
        
        if not created:
            # Item already in cart, increment quantity
            cart_item.quantity += quantity
            cart_item.save()
        
        # Update transaction total
        update_transaction_total(transaction)
        
        return JsonResponse({
            'success': True,
            'item_id': cart_item.id,
            'product_name': product.name,
            'quantity': cart_item.quantity,
            'unit_price': float(cart_item.unit_price),
            'total_amount': float(transaction.total_amount),
            'item_count': transaction.items.count()
        }, status=status.HTTP_201_CREATED)
        
    except Transaction.DoesNotExist:
        return JsonResponse(
            {'error': 'Transaction not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Product.DoesNotExist:
        return JsonResponse(
            {'error': 'Product not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except json.JSONDecodeError:
        return JsonResponse(
            {'error': 'Invalid JSON'},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return JsonResponse(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@require_http_methods(["GET"])
def cart(request, transaction_id):
    """
    API endpoint: /api/cart/<transaction_id>/
    Returns cart contents for a transaction
    """
    try:
        transaction = Transaction.objects.get(id=transaction_id)
        
        items = TransactionItem.objects.filter(transaction=transaction).values()
        items_list = []
        
        for item in items:
            product = Product.objects.get(id=item['product_id'])
            items_list.append({
                'id': item['id'],
                'product_id': item['product_id'],
                'product_name': product.name,
                'quantity': item['quantity'],
                'unit_price': float(item['unit_price']),
                'total': float(item['quantity'] * item['unit_price'])
            })
        
        return JsonResponse({
            'transaction_id': transaction.id,
            'status': transaction.status,
            'items': items_list,
            'total_amount': float(transaction.total_amount),
            'item_count': len(items_list)
        }, status=status.HTTP_200_OK)
        
    except Transaction.DoesNotExist:
        return JsonResponse(
            {'error': 'Transaction not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return JsonResponse(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@require_http_methods(["GET"])
def latest_transaction(request):
    """
    API endpoint: /api/latest-transaction/
    Returns the latest open transaction (active cart)
    Creates a new one if none exists
    """
    try:
        # Try to get existing open transaction
        transaction = Transaction.objects.filter(status='open').last()
        
        if not transaction:
            # Create new transaction if none exists
            transaction = Transaction.objects.create(status='open')
            print(f"[API] Created new transaction: {transaction.id}")
        
        return JsonResponse({
            'id': transaction.id,
            'status': transaction.status,
            'total_amount': float(transaction.total_amount),
            'created_at': transaction.created_at.isoformat(),
            'item_count': transaction.items.count()
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return JsonResponse(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@csrf_exempt
@require_http_methods(["POST"])
def checkout_complete(request):
    """
    API endpoint: /api/checkout-complete/
    Mark transaction as paid and complete checkout
    """
    try:
        data = json.loads(request.body)
        transaction_id = data.get('transaction_id')
        payment_method = data.get('payment_method', 'card')
        
        transaction = Transaction.objects.get(id=transaction_id)
        transaction.status = 'paid'
        transaction.save()
        
        print(f"[CHECKOUT] Transaction {transaction_id} marked as paid")
        
        return JsonResponse({
            'success': True,
            'transaction_id': transaction.id,
            'status': transaction.status,
            'total_amount': float(transaction.total_amount),
            'payment_method': payment_method
        }, status=status.HTTP_200_OK)
        
    except Transaction.DoesNotExist:
        return JsonResponse(
            {'error': 'Transaction not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return JsonResponse(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def update_transaction_total(transaction):
    """
    Recalculate and update transaction total amount
    """
    total = 0
    for item in transaction.items.all():
        total += item.quantity * item.unit_price
    
    transaction.total_amount = total
    transaction.save()


def video_feed(request):
    """
    View: /video-feed/
    Returns MJPEG stream with detection overlay
    """
    def generate():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    break
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # Yield MJPEG boundary and frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                       + frame_bytes + b'\r\n')
        finally:
            camera.release()
    
    return StreamingHttpResponse(
        generate(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def checkout(request):
    """
    View: /checkout/
    Main checkout page with live camera feed and cart
    """
    return render(request, 'checkout/checkout.html')
