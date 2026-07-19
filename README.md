# Self-Checkout System

A Django-based self-checkout system using OpenCV for real-time product detection and recognition.

## Features

- **Real-time Product Detection**: Uses YOLO v8 for fast object detection
- **Item Recognition**: Identifies products via barcode or image recognition
- **Weight Verification**: Validates items against expected weights using a scale sensor
- **Transaction Management**: Django ORM for checkout history and inventory tracking
- **Payment Integration**: Ready for Stripe/PayPal integration
- **Live Camera Feed**: WebSocket streaming with real-time frame processing
- **User Authentication**: Session management for customer and staff
- **REST API**: Full-featured API for frontend integration

## Tech Stack

- **Backend**: Django 4.2 + Django REST Framework
- **Computer Vision**: OpenCV, YOLO v8 (ultralytics)
- **Real-time Communication**: Django Channels (WebSockets)
- **Database**: PostgreSQL
- **Deployment**: Docker + Docker Compose

## Project Structure

```
self-checkout-system/
├── checkout/              # Main Django app
│   ├── models.py         # Database models
│   ├── views.py          # API views
│   ├── serializers.py    # DRF serializers
│   ├── urls.py           # URL routing
│   └── consumers.py      # WebSocket consumers
├── vision/               # Computer vision module
│   ├── detector.py       # YOLO detection logic
│   ├── processor.py      # Frame processing
│   └── utils.py          # Vision utilities
├── config/               # Django settings
├── static/               # Frontend assets
├── templates/            # HTML templates
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Container orchestration
└── manage.py            # Django CLI
```

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Webcam or video source

### Setup

1. Clone the repository
```bash
git clone https://github.com/Admin5152/self-checkout-system.git
cd self-checkout-system
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Setup database
```bash
python manage.py migrate
python manage.py createsuperuser
```

5. Run development server
```bash
python manage.py runserver
```

6. Start Celery for background tasks (in another terminal)
```bash
celery -A config worker -l info
```

7. Access the application
- Admin: http://localhost:8000/admin
- API: http://localhost:8000/api/
- Frontend: http://localhost:8000/

## API Endpoints

### Authentication
- `POST /api/auth/login/` - User login
- `POST /api/auth/logout/` - User logout
- `POST /api/auth/register/` - Customer registration

### Checkout
- `POST /api/checkout/start/` - Start new checkout session
- `GET /api/checkout/session/<id>/` - Get session details
- `POST /api/checkout/add-item/` - Add item to cart
- `POST /api/checkout/complete/` - Complete transaction

### Products
- `GET /api/products/` - List all products
- `GET /api/products/<sku>/` - Get product details
- `POST /api/products/search/` - Search products

### Camera
- `WebSocket /ws/camera/` - Real-time camera feed and detection

## Configuration

Create a `.env` file in the root directory:

```env
DEBUG=False
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:password@db:5432/checkout_db
ALLOWED_HOSTS=localhost,127.0.0.1
YOLO_CONF_THRESHOLD=0.5
WEIGHT_SENSOR_PORT=/dev/ttyUSB0
STRIPE_API_KEY=your-stripe-key
STRIPE_PUBLIC_KEY=your-public-key
```

## Docker Deployment

```bash
docker-compose up -d
```

This will start:
- Django application (port 8000)
- PostgreSQL database
- Redis cache
- Celery worker

## WebSocket Connection Example

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/camera/');

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Process frame with detected items
  console.log(data.detections);
};

socket.send(JSON.stringify({
  action: 'start_detection',
  camera_id: 1
}));
```

## Testing

```bash
python manage.py test
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions, please open a GitHub issue or contact support@example.com
