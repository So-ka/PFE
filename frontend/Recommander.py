#pip install pygad
#pip install tensorflow
#pip install keras
#pip install numpy
#pip install pandas
#pip install sklearn
#pip install matplotlib

from keras.metrics import accuracy
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate,Dropout
from keras.models import Model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow import keras
import pygad


os.chdir('datasets')

rates = pd.read_csv('ratings_small.csv')
rates_mm = rates.pivot(index='userId',columns='movieId',values="rating").fillna(2.5)
user_movie = rates.pivot(index='userId',columns='movieId',values="rating")
rates.columns = [['userId','movieId','rating','']]
rates = rates[['userId','movieId','rating']]

MMScaler = MinMaxScaler()
MMScaler.fit(rates_mm)
rates_mm = MMScaler.transform(rates_mm)

user_movie = user_movie.dropna(thresh=1).fillna(2.5)
user_movie.columns.name=None
user_movie.index.name=None

train, test = train_test_split(rates, test_size=0.3, random_state=42)
train_, test_ = train_test_split(user_movie, test_size=0.3, random_state=42)

GA_Centroids=[]
inertia_min = 999999

#Initiating Neural Network
def RNN_init(train,rates,opt='Adam',epochs=10,lr=0.001,embd=5):
  n_users = max(rates.userId.values)[0]
  n_Movie = max(rates.movieId.values)[0]
  Movie_input = Input(shape=[1], name="Movie-Input")
  Movie_embedding = Embedding(n_Movie+1, embd, name="Movie-Embedding")(Movie_input)
  Movie_vec = Flatten(name="Flatten-Movie")(Movie_embedding)
  user_input = Input(shape=[1], name="User-Input")
  user_embedding = Embedding(n_users+1, embd, name="User-Embedding")(user_input)
  user_vec = Flatten(name="Flatten-Users")(user_embedding)
  conc = Concatenate()([Movie_vec, user_vec])
  fc1 = Dense(128, activation='relu')(conc)
  fc2 = Dense(32, activation='relu')(fc1)
  out = Dense(1)(fc2)
  model = Model([Movie_input,user_input ], out)
  if opt=='Adam' :
    opt = keras.optimizers.Adam(learning_rate=lr)
  elif opt == 'SGD':
    opt = keras.optimizers.SGD(learning_rate=lr)
  elif opt == 'RMSprop':
    opt = keras.optimizers.RMSprop(learning_rate=lr)
  elif opt == 'Adadelta':
    opt = keras.optimizers.Adadelta(learning_rate=lr)
  elif opt == 'Adagrad':
    opt = keras.optimizers.Adagrad(learning_rate=lr)
  elif opt == 'Adamax':
    opt = keras.optimizers.Adamax(learning_rate=lr)
  elif opt == 'Nadam':
    opt = keras.optimizers.Nadam(learning_rate=lr)
  model.compile(loss='mean_squared_error', optimizer=opt)
  model.fit([train.movieId, train.userId], train.rating, epochs=epochs)
  return model
#Centroid_Fitness Function
def Centroid_Fitness(solution, solution_idx):
  global GA_Centroids , inertia_min
  bef = KMeans(n_clusters=34,n_init=2)
  bef.fit(rates_mm,solution.T.reshape(34,9066)[solution_idx])
  inertia = bef.inertia_
  if inertia < inertia_min :
    inertia_min = inertia
    GA_Centroids = bef.cluster_centers_
  fitness = 1.0 / ( inertia + 0.00001)
  return fitness
#GA Optimization Function
def GA_Optimization(rates=rates,features=9066):
  num_genes = 34 * 9066
  num_clusters = 34
  feature_vector_length=9066
  data = rates
  ga_instance = pygad.GA(num_generations=1,
                       sol_per_pop=10,
                       num_parents_mating=7,
                       keep_parents=2,
                       num_genes=num_genes,
                       fitness_func=Centroid_Fitness,)
  ga_instance.run()
#kmeans Initiating
def Cluster_init(user_movie,GA_BestSolution='random'):
  kmeans_kwargs = {
    "init":GA_BestSolution,
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
  }
  kmeans = KMeans(n_clusters=34, **kmeans_kwargs)
  kmeans.fit(user_movie)
  return kmeans
#Collaborative Filtering Function
def CF_UserUser(user,usermovie,classifier):
  prediction = kmeans.predict(user_movie)
  User_Cluster = kmeans.predict(user)
  list_of_similiar_people = user_movie.index[prediction==User_Cluster]
  #list_of_rating = user_movie.values[list_of_similiar_people]
  list_of_rating = user_movie[np.isin(user_movie.index.values,list_of_similiar_people)].values
  list_of_rating = np.insert(list_of_rating,0,user,axis=0)
  
  new_l= []
  for i in list_of_rating:
    new_l.append(i)
  res = cosine_similarity(list_of_rating)[0]

  return new_l[np.argmax(res[1:])],list_of_similiar_people[np.argmax(res[1:])-1]
#Neural Network Prediction Function
def Neural_Predictions(model,User_,ID_):
  c1 = (np.zeros(len(user_movie.columns.values))+ID_).astype(int)
  c2 = user_movie.columns.astype(int)
  y_ = model.predict([c2,c1])
  dd = np.array([y_.T[0],c2]).T
  dd = dd[dd[:, 0].argsort()[::-1]]
  return dd
  #return y_
#Web Data To Python
def Create_Profile(Web_Ratings,user_movie=user_movie):
  CP = np.zeros(9066)+2.5
  for x in Web_Ratings : 
    CP[user_movie.columns==x['movie_id']] = x['ratings']
  return CP.reshape(1,-1)
#Evaluating Model
def Evaluation(Origin,Pred):
  d = Origin[Origin!=2.5] - Pred[Origin!=2.5]
  mse_f = np.mean(d**2)
  mae_f = np.mean(abs(d))
  rmse_f = np.sqrt(mse_f)
  return {'mae':mae_f,'mse':mse_f,'rmse':rmse_f}
#Evaluating Test Model

try:
  RNN_ = keras.models.load_model('Models\XMainDataset_TrainedModel_.h5')
except:
  RNN_ = RNN_init(rates,rates)
  RNN_.save('Models\XMainDataset_TrainedModel_.h5')

def Final_Evaluation(UM_Test,kmeans,model=RNN_):
  x=[]
  y=[]
  for id,a in zip(UM_Test.index,UM_Test.values):
    cf,cf_id = CF_UserUser(np.array([a]),id,kmeans)
    x.append(cf)
    r = Neural_Predictions(model,np.array(cf)[0],cf_id)
    y.append(r.T[0])
  return Evaluation(x,y)

# To use GA_kmeans :
#GA_Optimization(rates_mm)
kmeans = Cluster_init(rates_mm)#,GA_BestSolution=GA_Centroids)

#Final is Results where C1 is Ratings and C2 is Movie_ID Sorted in Descending Order

UM_Test = user_movie[np.isin(user_movie.index,test_.index)]

#Final_Evaluation(UM_Test,kmeans)

###Evaluation(Backup,Final.T[0])