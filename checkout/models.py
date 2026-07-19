from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid


class Product(models.Model):
    """Product inventory model"""
    CATEGORY_CHOICES = [
        ('produce', 'Produce'),
        ('dairy', 'Dairy'),
        ('meat', 'Meat'),
        ('bakery', 'Bakery'),
        ('frozen', 'Frozen'),
        ('pantry', 'Pantry'),
        ('beverage', 'Beverage'),
        ('personal', 'Personal Care'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sku = models.CharField(max_length=50, unique=True, db_index=True)
    barcode = models.CharField(max_length=100, unique=True, null=True, blank=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    price = models.DecimalField(max_digits=8, decimal_places=2, validators=[MinValueValidator(0)])
    weight = models.FloatField(help_text="Weight in kg", null=True, blank=True)
    image = models.ImageField(upload_to='products/', null=True, blank=True)
    stock = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    reorder_level = models.IntegerField(default=10)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['sku']),
            models.Index(fields=['barcode']),
            models.Index(fields=['category']),
        ]

    def __str__(self):
        return f"{self.sku} - {self.name}"

    def is_in_stock(self):
        return self.stock > 0


class CheckoutSession(models.Model):
    """Active or completed checkout session"""
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('abandoned', 'Abandoned'),
        ('cancelled', 'Cancelled'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    customer = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    tax_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    discount_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    payment_method = models.CharField(max_length=50, null=True, blank=True)  # card, cash, paypal, etc.
    transaction_id = models.CharField(max_length=255, null=True, blank=True, unique=True)
    notes = models.TextField(blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['status', '-started_at']),
            models.Index(fields=['customer', '-started_at']),
        ]

    def __str__(self):
        return f"Checkout {self.id} - {self.status}"

    def get_total(self):
        return self.total_amount + self.tax_amount - self.discount_amount

    def mark_completed(self, payment_method=None, transaction_id=None):
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.payment_method = payment_method or 'unknown'
        self.transaction_id = transaction_id
        self.save()


class CartItem(models.Model):
    """Items in a checkout session"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(CheckoutSession, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True)
    quantity = models.IntegerField(default=1, validators=[MinValueValidator(1)])
    unit_price = models.DecimalField(max_digits=8, decimal_places=2)  # Price at time of scan
    total_price = models.DecimalField(max_digits=10, decimal_places=2)
    detected_weight = models.FloatField(null=True, blank=True, help_text="Weight detected by scale in kg")
    expected_weight = models.FloatField(null=True, blank=True, help_text="Expected weight in kg")
    weight_verified = models.BooleanField(default=False)
    detected_by = models.CharField(max_length=50, default='barcode')  # barcode, image, manual
    added_at = models.DateTimeField(auto_now_add=True)
    scanned_image = models.ImageField(upload_to='scans/', null=True, blank=True)

    class Meta:
        ordering = ['added_at']

    def __str__(self):
        return f"{self.product.name} x {self.quantity}"

    def calculate_total(self):
        self.total_price = self.unit_price * self.quantity
        return self.total_price

    def verify_weight(self):
        """Check if detected weight matches expected weight within tolerance"""
        if not self.expected_weight or self.detected_weight is None:
            return True
        
        from django.conf import settings
        tolerance = settings.WEIGHT_TOLERANCE
        
        expected = self.expected_weight * self.quantity
        detected = self.detected_weight
        
        min_weight = expected * (1 - tolerance)
        max_weight = expected * (1 + tolerance)
        
        verified = min_weight <= detected <= max_weight
        self.weight_verified = verified
        return verified


class Detection(models.Model):
    """Logs of detected items via computer vision"""
    DETECTION_TYPE_CHOICES = [
        ('yolo', 'YOLO Detection'),
        ('barcode', 'Barcode Scan'),
        ('manual', 'Manual Entry'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(CheckoutSession, on_delete=models.CASCADE, related_name='detections')
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
    detection_type = models.CharField(max_length=20, choices=DETECTION_TYPE_CHOICES)
    confidence = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    detected_data = models.JSONField()  # Store raw detection data
    frame_image = models.ImageField(upload_to='frames/')
    bounding_box = models.JSONField()  # {x1, y1, x2, y2}
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.detection_type} - {self.product} ({self.confidence:.2f})"


class CameraStream(models.Model):
    """Camera configuration and status"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    source = models.CharField(max_length=255, help_text="Camera source URL or device path (0 for webcam)")
    location = models.CharField(max_length=200, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    frame_rate = models.IntegerField(default=30)
    resolution_width = models.IntegerField(default=1920)
    resolution_height = models.IntegerField(default=1080)
    last_frame_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class Scale(models.Model):
    """Weight scale sensor configuration"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    serial_port = models.CharField(max_length=100)
    baudrate = models.IntegerField(default=9600)
    is_active = models.BooleanField(default=True)
    calibration_factor = models.FloatField(default=1.0)
    last_reading = models.FloatField(null=True, blank=True)
    last_reading_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name
