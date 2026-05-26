
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
)


dataset = pd.read_csv(
        r"C:\Users\Karen\Desktop\Fraude\creditcard_2023.csv")

print("\nShape:")
print(dataset.shape)

print("\nHead:")
print(dataset.head())

print("\nDescribe:")
print(dataset.describe())

print("\nnulos:")
print(dataset.isnull().sum())

print("\nDistribuição da classe:")
print(dataset["Class"].value_counts())

sns.countplot(x=dataset["Class"])


x_dataset = dataset.iloc[:, 1:30].values
y_dataset = dataset.iloc[:, 30].values


scaler = StandardScaler()
x_dataset = scaler.fit_transform(x_dataset)


x_dataset_treinamento, x_dataset_teste, y_dataset_treinamento, y_dataset_teste = train_test_split(
    x_dataset,
        y_dataset,
        test_size=0.25,
        random_state=0,
        stratify=y_dataset
    )

print("\nShape treino:")
print(x_dataset_treinamento.shape)
print(y_dataset_treinamento.shape)

print("\nShape teste:")
print(x_dataset_teste.shape)
print(y_dataset_teste.shape)


dataset_naive = GaussianNB()
dataset_naive.fit(x_dataset_treinamento, y_dataset_treinamento)


prob_fraude = dataset_naive.predict_proba(x_dataset_teste)[:, 1]

threshold = 0.5
previsao = (prob_fraude >= threshold).astype(int)


taxa_acerto = accuracy_score(y_dataset_teste, previsao)
matriz = confusion_matrix(y_dataset_teste, previsao)

print("\nAccuracy:")
print(taxa_acerto)

print("\nMatriz de confusão:")
print(matriz)

print("\nClassification Report:")
print(classification_report(y_dataset_teste, previsao))

print("\nROC-AUC:")
print(roc_auc_score(y_dataset_teste, prob_fraude))

print("\nScore teste:")
print(dataset_naive.score(x_dataset_teste, y_dataset_teste))

print("\nScore treinamento:")
print(dataset_naive.score(x_dataset_treinamento, y_dataset_treinamento))



