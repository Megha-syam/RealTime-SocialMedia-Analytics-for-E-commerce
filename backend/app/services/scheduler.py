from apscheduler.schedulers.background import BackgroundScheduler
from app.models import Product
from app.services.ingestion_service import ingest_product_data


scheduler = BackgroundScheduler()
_app = None


def _scheduled_ingestion():
    if _app is None:
        return
    with _app.app_context():
        products = Product.query.limit(30).all()
        for product in products:
            ingest_product_data(product.display_name, include_ai=False)


def start_scheduler(app):
    global _app
    if scheduler.running:
        return

    _app = app
    interval = app.config.get("INGESTION_INTERVAL_SECONDS", 60)
    scheduler.add_job(_scheduled_ingestion, "interval", seconds=interval, id="periodic_ingestion")
    scheduler.start()


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
