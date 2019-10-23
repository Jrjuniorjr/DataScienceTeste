import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import StandardScaler
from os import listdir
from sklearn.metrics import precision_score
from sklearn.linear_model import Perceptron
#from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

dir = 'datasets/'

def scores(clf_name, metodo, acc, file, split_number, iteracao, output, tp, tn, fp, fn, recall, precision, mco):
    with open(output, 'at') as out_file:
        line = f"\"{file} , {clf_name} , {metodo} , Split: {split_number} , Treino: {iteracao}\"," 
        line+= f"ACC: {acc}, True_Positive: {tp}, True_Negative: {tn}, False_Negative: {fp}, False_Negative: {fn}, Recall: {recall}, Precision: {precision}, Matthews_Corrcoef: {mco}\n"
        #line += f"{matthews_corrcoef(target_test, prediction)},"
       # line += f"{recall_score(target_test, prediction, average='macro')},"
       # line += f"{precision_score(target_test, prediction, average='macro')}\n"
        out_file.writelines(line)

output = 'output_menores_datasets.csv'
#with open(output, 'wt') as out_file: 
        #out_file.writelines('\"Descrição\",\"Acurácia\",\"Matriz de Confusao\",\"Recall\",\"Precisão\",\"MCC\"\n')

names=[]
for file in listdir(dir):
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
    
    dados = data.iloc[:, 0:ultimaColuna]
    resposta = data.iloc[:,-1]
    
    maior_norm = 0
    melhor_depth_norm = 0
    melhor_state_norm = 0
    
    maior_s = 0
    melhor_depth_s = 0
    melhor_state_s = 0
    
    
    
    for i in range(10):
        ft_train, ft_test, tg_train, tg_test = train_test_split(dados, resposta, train_size=0.7+(i/100), random_state=i)
        
        
        n = Normalizer()
        ft_train_norm = n.fit_transform(ft_train)
        ft_test_norm = n.transform(ft_test)
        
        
        s = StandardScaler()
        ft_train_s = s.fit_transform(ft_train)
        ft_test_s = s.transform(ft_test)
        
        
        j = 2
    
        while j<50:            
            
            
            #PERCEPTRON NORMALIZADO
            
            percep_norm = Perceptron(random_state=i)
            percep_norm = percep_norm.fit(ft_train_norm, tg_train)
            prediction_norm = percep_norm.predict(ft_test_norm)
            
            acc = accuracy_score(prediction_norm, tg_test)
            tn,fp,fn,tp = confusion_matrix(tg_test,prediction_norm).ravel()
            #print(acc_norm)
            recall = recall_score(tg_test, prediction_norm, average='macro')
            precision = precision_score(tg_test, prediction_norm, average='macro')
            #f1_score = f1_score(tg_test, prediction_norm, average='macro')
            #f1_score = np.arrang
            mco = matthews_corrcoef(tg_test, prediction_norm)  
            scores("Perceptron", "Normalizado", acc, file, i, j, output, tp, tn, fp, fn, recall, precision, mco)


            
            #PERCEPTRON PADRONIZADO
            
            
            percep_s = Perceptron(random_state=i)
            percep_s = percep_s.fit(ft_train_s, tg_train)
            prediction_s = percep_s.predict(ft_test_s)
            
            acc = accuracy_score(prediction_s, tg_test)
            tn,fp,fn,tp = confusion_matrix(tg_test,prediction_s).ravel()
            #print(acc_norm)
            recall = recall_score(tg_test, prediction_s, average='macro')
            precision = precision_score(tg_test, prediction_s, average='macro')
            #f1_score = f1_score(tg_test, prediction_norm, average='macro')
            #f1_score = np.arrang
            mco = matthews_corrcoef(tg_test, prediction_norm)  
            scores("Perceptron", "Padronizado", acc, file, i, j, output, tp, tn, fp, fn, recall, precision, mco)

            
            
            
            
            
            #DT NORMALIZADO
            
            
            
            decision_tree_norm = DecisionTreeClassifier(random_state = i, max_depth= j)
            decision_tree_norm = decision_tree_norm.fit(ft_train_norm, tg_train)
            prediction_norm = decision_tree_norm.predict(ft_test_norm)
            
            acc = accuracy_score(prediction_norm, tg_test)
            tn,fp,fn,tp = confusion_matrix(tg_test,prediction_norm).ravel()
            #print(acc_norm)
            recall = recall_score(tg_test, prediction_norm, average='macro')
            precision = precision_score(tg_test, prediction_norm, average='macro')
            #f1_score = f1_score(tg_test, prediction_norm, average='macro')
            #f1_score = np.arrang
            mco = matthews_corrcoef(tg_test, prediction_norm)  
            #if acc_norm > maior_norm:
            #    maior_norm = acc_norm
             #   maior_depth_norm = j
             #   maior_state_norm = i
                
            scores("DT", "Normalizado", acc, file, i, j, output, tp, tn, fp, fn, recall, precision, mco)
            
            
            
            #DT PADRONIZADO            
            
            
            decision_tree_s = DecisionTreeClassifier(random_state = i, max_depth= j)
            decision_tree_s = decision_tree_s.fit(ft_train_s, tg_train)
            prediction_s = decision_tree_s.predict(ft_test_s)
            
            acc= accuracy_score(prediction_s, tg_test)
            tn,fp,fn,tp = confusion_matrix(tg_test,prediction_s).ravel()
            recall = recall_score(tg_test, prediction_s, average='macro')
            precision = precision_score(tg_test, prediction_s, average='macro') 
           # f1_score = f1_score(tg_test, prediction_s, average='macro')  
            mco = matthews_corrcoef(tg_test, prediction_s)  
            
            ##print(acc_s)
            ##if acc_s > maior_s:
             ##   maior_s = acc_s
                #maior_depth_s = j
                #maior_state_s = i
               
            scores("DT", "Padronizado", acc, file, i, j, output, tp, tn, fp, fn, recall, precision, mco)
            
            j = j + 1
            
            
print("fim")
