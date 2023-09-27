from django import forms
from .models import RateBDD

class RateDBForm(forms.ModelForm):
    class Meta:
        model=RateBDD
        fields=('ratings','movie_id')
        
class sear(forms.Form):
    Search = forms.CharField()

class deleteRating(forms.Form):
    id = forms.CharField()

class Evaluate(forms.Form):
    lr = forms.CharField()
    opt = forms.CharField()
    epochs = forms.CharField()
    embd_ = forms.CharField()
    default = forms.CharField()

class GenreSearch(forms.Form):
    Search = forms.CharField()