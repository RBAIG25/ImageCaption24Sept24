from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
app = FastAPI()
templates = Jinja2Templates(directory = 'templates/')
from sklearn.datasets import load_iris
iris = load_iris()
import pandas as pd
df = pd.DataFrame(data = iris.data,columns = iris.feature_names)
df['Species'] = iris.target
#print(df.head())
X = df.iloc[:,0:4]
y = df.iloc[:,4]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

#from sklearn.metrics import accuracy_score
#print('Accuracy' ,accuracy_score(y_test,y_pred))
@app.get('/')
def read_form():
    return 'hello world'
@app.get('/test')
def form_post(request:Request):
    res = 'Please enter Data'
    return templates.TemplateResponse('test.html',context  = {'request':request,'result':res})
@app.post('/test')
def form_post(request:Request,num1:float=Form(...),num2:float=Form(...),num3:float=Form(...),num4:float=Form(...)):
        result = dt.predict([[num1,num2,num3,num4]])
        if result == 0:
            t = 'Versicolor'
            pic = 'https://cdn.pixabay.com/photo/2017/05/24/08/22/iris-2339883_1280.jpg'
        elif result == 1:
            t = 'setosa'
            pic = 'https://cdn.pixabay.com/photo/2015/05/26/13/57/flower-784688__340.jpg'
        else:
            t= 'virginica'
            pic = 'https://s3.amazonaws.com/eit-planttoolbox-prod/media/images/Iris-virginica--Jenny-Evans--CC-BY-NC.jpg'
        return templates.TemplateResponse('test.html',context = {'request':request,'result':t,'num1':num1,'num2':num2,'num3':num3,'num4':num4,'pic':pic})

   



