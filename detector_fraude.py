import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# 1. Carregar os dados
df = pd.read_csv("transacoes.csv")

# 2. Variáveis independentes (X) e alvo (y)
X = df.drop("fraude", axis=1)
y = df["fraude"]

# 3. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Balanceamento de classes (fraudes são geralmente <5%)
classes = [0,1]
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
weights = dict(zip(classes, class_weights))

# 5. Treinar modelo
modelo = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=weights)
modelo.fit(X_train, y_train)

# 6. Predição
y_pred = modelo.predict(X_test)

# 7. Relatório de classificação
print("=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred))

# 8. Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Fraude"],
            yticklabels=["Normal","Fraude"])
plt.title("Matriz de Confusão - Detecção de Fraudes")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# 9. Importância das variáveis
importancias = modelo.feature_importances_
indices = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importancias, y=indices)
plt.title("Importância das Variáveis no Modelo")
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.show()
