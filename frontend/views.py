from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
import frontend.Grapper as Grapper
from .models import RateBDD
from .forms import RateDBForm,sear,deleteRating,Evaluate,GenreSearch
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
# Create your views here.

@csrf_exempt
def index(request):
    if request.method == 'POST':
        form = RateDBForm(request.POST)
        former = sear(request.POST)

        if form.is_valid():            
            form.save()

        if former.is_valid():            
            Results = Grapper.Get_Title_Researche(former['Search'].value())
            return render(request,'Search.html',{'Data':Results})
    print(request.GET.get('Category'))
    if request.GET.get('Category'):
        print('Category Searched')
        Results = Grapper.Get_Search_Category(request.GET.get('Category').lower())
        return render(request,'Search.html',{'Data':Results})
        
    trending_ =  Grapper.get_trends()
    Best_Sellers = Grapper.get_best_sellers()
    recommand = Grapper.Recommand()
    return render(request,'home.html',{'Data':trending_,'Sells':Best_Sellers,'recommand':recommand})

@csrf_exempt    
def ratel(request):
    if request.method == 'POST':
        form = deleteRating(request.POST)
        if form.is_valid():   
            id = form['id'].value()
            RateBDD.objects.filter(movie_id=id).delete()
    res = Grapper.Get_Ratings_list()
    return render(request,'Rating_list.html',{'Data':res})

def ind(request):
    return render(request,'Index.html')

@csrf_exempt   
def eva(request):
    x = {'mae':'','mse':'','rmse':''}
    if request.method == 'POST':
        form = Evaluate(request.POST)
        if form.is_valid():   
            lr = float(form['lr'].value())
            opt = (form['opt'].value())
            embd_ = int(form['embd_'].value())
            epoch = int(form['epochs'].value())
            default = form['default'].value()

            x = Grapper.Evaluation_Finale(lr,opt,embd_,epoch,default)
    return render(request,'Evalution.html',{'Data':x})
    