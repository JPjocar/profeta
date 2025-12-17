from django.db import models

class Dataset(models.Model):
    session_key = models.CharField(max_length=64, db_index=True)
    original_filename = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

class Product(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="products")
    name = models.CharField(max_length=255, db_index=True)

    class Meta:
        unique_together = ("dataset", "name")

class SaleDaily(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="sales_daily")
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="sales_daily")
    date = models.DateField(db_index=True)
    qty = models.FloatField()

    class Meta:
        unique_together = ("dataset", "product", "date")
        indexes = [models.Index(fields=["dataset", "product", "date"])]
