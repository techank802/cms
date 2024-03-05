from django.db import models

class Prediction(models.Model):
    """
    Represents a prediction for sales.
    """
    date = models.DateTimeField(max_length=50)
    price = models.FloatField(max_length=50)

    def __str__(self):
        """
        String representation of the sales prediction.
        """
        return (
            f"Date: {self.date}"
            f"Price: {self.price}"
            )
    class Meta:
        verbose_name_plural = 'Predictions'
        app_label = 'dibba'