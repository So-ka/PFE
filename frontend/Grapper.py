from django.forms import NullBooleanField
import pandas as pd
import numpy as np
import urllib.request, json 
import os 
from .models import RateBDD
from ast import literal_eval
import frontend.Recommander as Recommander


os.chdir('..')
os.chdir('datasets')

dataset = pd.read_csv('movies_metadata.csv',index_col=8)
dataset['genres'] = dataset['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'].lower() for i in x] if isinstance(x, list) else [])


def construct_poster_path(id):
    try:
        url = urllib.request.urlopen("https://api.themoviedb.org/3/movie/"+id+"?api_key=6a8e3b0fe6893390b625222b7e968f16")
        data = json.loads(url.read().decode())
        if data['poster_path'] == None:
            return None
        return "https://image.tmdb.org/t/p/w200"+data['poster_path']
    except Exception:
        pass 
    

def get_trends():
    trending_data = []
    Trends = dataset[['popularity','id','title','vote_average']].sort_values('popularity',ascending=False).head(10)
    for i in Trends.values :
        trending_data.append({'id':i[1],'title':i[2],'vote_average':i[3]*5/10,'poster_path':"files\cache\\"+(str(i[1])+".jpg")})
    return trending_data;

def get_best_sellers():
    best_seller = []
    sellers = dataset[['revenue','id','title','vote_average']].sort_values('revenue',ascending=False).head(25)
    for i in sellers.values :
        best_seller.append({'id':i[1],'title':i[2],'vote_average':i[3]*5/10,'poster_path':"files\cache\\"+(str(i[1])+".jpg")})
    return best_seller;

def Get_Title_Researche(title):
    x = title.lower().split(' ')
    results = (dataset[['id','title','vote_average']].dropna()[(dataset['title'].dropna().str.lower().str.contains('|'.join(x))).values])
    res = []
    for i in results[:20].values:
        print(i[0])
        res.append({'id':i[0],'title':i[1],'vote_average':i[2]*5/10,'poster_path':construct_poster_path(str(i[0]))})
    return res
    
def Get_Search_Category(category):
    l = []
    for i in range(0,dataset.genres.shape[0]):
        l.append(category.lower())
    x = np.array(l)
    res=[]
    for a,b in zip(x,dataset.genres.values):
        if a in b :
            res.append(True)
        else:
            res.append(False)
    results = dataset[['id','title','vote_average']][res]
    
    res = []
    for i in results[:20].values:
        print(i[0])
        res.append({'id':i[0],'title':i[1],'vote_average':i[2]*5/10,'poster_path':construct_poster_path(str(i[0]))})
    return res

def Get_Ratings_list():
    data = RateBDD.objects.all()
    res =[]
    for i in list(data):
        res.append({'id':i.movie_id,'ratings':i.ratings*5/10,'poster_path':construct_poster_path(str(i.movie_id))})
    return res

def Recommand():
    data = RateBDD.objects.all()
    data = RateBDD.objects.values('movie_id','ratings')
    data = list(data)
    Target_User = Recommander.Create_Profile(data)
    print(Target_User[0])

    CF_User_ratings,CF_User_id = Recommander.CF_UserUser(Target_User,Recommander.user_movie,Recommander.kmeans)
    print(CF_User_ratings)

    Backup = np.array(CF_User_ratings)
    Final = Recommander.Neural_Predictions(Recommander.RNN_,CF_User_ratings,CF_User_id)
    res = []
    #subset = Final[:100]
    #np.random.shuffle(subset)
    f = 0
    cached = [73290,3030,7459,1860,309,1859,52767,139757,50740,55069,26587]
    for i in Final[:10]:
        if str(int(i[1])) not in cached :
            ppath = construct_poster_path(str(int(i[1])))
        else :
            ppath="files\cache\\"+str(i[1])+".jpg"

        if ppath != None :
            print(str(int(i[1])))
            res.append({'id':str(int(i[1])),'ratings':str(int(i[0])),'poster_path':ppath})
        f = len(res)
        if(f > 10):
            break
    return res

def Evaluation_Finale(lr,opt,embd,epch,default):
    if default == 'on':
        return Recommander.Final_Evaluation(Recommander.UM_Test,Recommander.kmeans)
    else :
        model_trained = Recommander.RNN_init(Recommander.rates,Recommander.rates,lr=lr,opt=opt,embd=embd,epochs=epch)
        return Recommander.Final_Evaluation(Recommander.UM_Test,Recommander.kmeans,model=model_trained)
