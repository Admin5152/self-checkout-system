from rest_framework import serializers
from django.contrib.auth.models import User
from checkout.models import (
    Product, CheckoutSession, CartItem, Detection, CameraStream, Scale
)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id']


class ProductSerializer(serializers.ModelSerializer):
    is_in_stock = serializers.SerializerMethodField()
    
    class Meta:
        model = Product
        fields = [
            'id', 'sku', 'barcode', 'name', 'description', 'category',
            'price', 'weight', 'image', 'stock', 'is_in_stock',
            'reorder_level', 'is_active', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
        lookup_field = 'sku'

    def get_is_in_stock(self, obj):
        return obj.is_in_stock()


class CartItemSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source='product.name', read_only=True)
    product_details = ProductSerializer(source='product', read_only=True)
    
    class Meta:
        model = CartItem
        fields = [
            'id', 'session', 'product', 'product_name', 'product_details',
            'quantity', 'unit_price', 'total_price', 'detected_weight',
            'expected_weight', 'weight_verified', 'detected_by', 'added_at'
        ]
        read_only_fields = ['id', 'total_price', 'added_at']

    def create(self, validated_data):
        item = CartItem(**validated_data)
        item.total_price = item.calculate_total()
        item.save()
        return item


class DetectionSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source='product.name', read_only=True)
    
    class Meta:
        model = Detection
        fields = [
            'id', 'session', 'product', 'product_name', 'detection_type',
            'confidence', 'detected_data', 'frame_image', 'bounding_box',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class CheckoutSessionSerializer(serializers.ModelSerializer):
    items = CartItemSerializer(many=True, read_only=True)
    customer_name = serializers.CharField(source='customer.get_full_name', read_only=True)
    total = serializers.SerializerMethodField()
    
    class Meta:
        model = CheckoutSession
        fields = [
            'id', 'customer', 'customer_name', 'status', 'total_amount',
            'tax_amount', 'discount_amount', 'total', 'payment_method',
            'transaction_id', 'notes', 'started_at', 'completed_at',
            'updated_at', 'items'
        ]
        read_only_fields = ['id', 'started_at', 'completed_at', 'updated_at', 'items']

    def get_total(self, obj):
        return float(obj.get_total())


class CameraStreamSerializer(serializers.ModelSerializer):
    class Meta:
        model = CameraStream
        fields = [
            'id', 'name', 'source', 'location', 'is_active',
            'frame_rate', 'resolution_width', 'resolution_height',
            'last_frame_at', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'last_frame_at', 'created_at', 'updated_at']


class ScaleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scale
        fields = [
            'id', 'name', 'serial_port', 'baudrate', 'is_active',
            'calibration_factor', 'last_reading', 'last_reading_at',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'last_reading', 'last_reading_at', 'created_at', 'updated_at']
