import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix
import os


from os import listdir


import warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

def scores(clf_name, prediction, metodo, target_test, file, split_number, iteracao, output):
    with open(output, 'at') as out_file:
        line = f"\"{file} , {clf_name} , {metodo} , Split # {split_number} , Treino # {iteracao}\","
        line += f"{accuracy_score(target_test, prediction)},"
        line += f"{matthews_corrcoef(target_test, prediction)},"
        line += f"{f1_score(target_test, prediction,average='macro')},"
        line += f"{recall_score(target_test, prediction, average='macro')},"
        line += f"{precision_score(target_test, prediction, average='macro')}\n"
        out_file.writelines(line)
dir = 'datasets/'
output = 'output.csv'
with open(output, 'wt') as out_file: 
        out_file.writelines('\"Descrição\",\"Acurácia\",\"F1-Score\",\"Recall\",\"Precisão\",\"MCC\"\n')
    

names=[]
for file in listdir(dir):
    fileName = os.path.splitext(file)[0]
    names.clear()
    print(f"---{dir + file}---")
    with open(dir +file, 'rt') as in_file:
        for line in in_file:
            if line.startswith("@inputs"):
                for word in line.split(" "):
                    if word != '@inputs':
                        names.append(word.replace('\n', ''))
                names.append("classes")
            if line.startswith("@data"):
                break
    data = pd.read_csv(dir + file, comment = '@', header=None)
    encoder = LabelEncoder()
    data = data.apply(encoder.fit_transform)
    ultimaColuna = len(names) - 1
    stateValue = 42
    ft = data.iloc[:, 0:ultimaColuna]
    tg = data.iloc[:,-1]  
    ft_train, ft_test, tg_train, tg_test = train_test_split(ft, tg,train_size=0.75, stratify =tg, random_state=stateValue)
    ft_train, ft_valid, tg_train, tg_valid = train_test_split(ft_train, tg_train,train_size=0.9,stratify =tg_train,random_state=stateValue)
        
    s = StandardScaler()
    padr_ft_train = s.fit_transform(ft_train)
    padr_ft_test = s.transform(ft_test)
        
    n = Normalizer()
    norm_ft_train = n.fit_transform(ft_train)       
    norm_ft_test = n.transform(ft_test)
    print("iniciando PERCEPTRON")
    percep = Perceptron(max_iter=10, random_state=0 ,eta0=0.1, n_jobs=-1)
    percep.fit(norm_ft_train, tg_train)
    norm_prediction = percep.predict(norm_ft_test)
    matriz = confusion_matrix(norm_ft_test, norm_prediction)
    matriz.save_csv("MatrizesConfusao/" + fileName + "Normalizado.csv")
    scores("Perceptron", norm_prediction, "Normalizado", tg_test, file, stateValue, 0, output)
            
    percep = Perceptron(max_iter=10, random_state=0 ,eta0=0.1, n_jobs=-1)
    percep.fit(padr_ft_train, tg_train)
    padr_prediction = percep.predict(padr_ft_test)
    matriz = confusion_matrix(padr_ft_test, padr_prediction)
    matriz.save_csv("MatrizesConfusao/" + fileName + "Padronizado.csv")
    scores("Perceptron", padr_prediction, "Padronizado", tg_test, file, stateValue, 0, output)
    print("saindo do PERCEPTRON")
            
    print("iniciando ARVORE DE DECISOES")
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(norm_ft_train, tg_train)
    norm_prediction = dt.predict(norm_ft_test)
    matriz = confusion_matrix(norm_ft_test, norm_prediction)
    matriz.save_csv("MatrizesConfusao/" + fileName + "Normalizado.csv")
    scores("Decision Tree", norm_prediction, "Normalizado", tg_test, file, stateValue, 0, output)


    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(padr_ft_train, tg_train)
    padr_prediction = dt.predict(padr_ft_test)
    matriz = confusion_matrix(padr_ft_test, padr_prediction)
    matriz.save_csv("MatrizesConfusao/" + fileName + "Padronizado.csv")
    scores("Decision Tree", padr_prediction, "Padronizado", tg_test, file, stateValue, 0, output)
    print("saindo da arvore de decisoes")
    

print("fim")