# ModeloRedeNeuralAntiFraude.py

"""
Rede Neural para Classificação de Fraude
Este script define e plota a arquitetura de uma rede neural simples usando Keras.
Gera um arquivo PNG com a visualização da estrutura da rede.
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

# Definição do modelo
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))  # Supondo 20 variáveis de entrada
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Saída binária: fraude (1) ou não fraude (0)

# Geração da imagem da arquitetura
plot_model(
    model,
    to_file='ModeloRedeNeuralAntiFraude.png',
    show_shapes=True,
    show_layer_names=True
)

print("Arquivo 'ModeloRedeNeuralAntiFraude.png' gerado com sucesso.")
