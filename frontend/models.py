from django.db import models
# Create your models here.

class RateBDD(models.Model):
    movie_id = models.IntegerField()
    ratings = models.IntegerField()
    def __str__(self):
        return "movie : "+str(self.movie_id) +" => Rate : "+ str(self.ratings)
