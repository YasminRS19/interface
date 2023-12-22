from flask import Flask,request,render_template
import pickle
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import joblib

app = Flask(__name__)

url='https://drive.google.com/uc?id=1EO2kECD4idaQhkS_XESX_Q1hQNnQRU8q'


df=pd.read_csv(url)

columns_to_drop = ['Escola', 'Sexo', 'idade', 'TP_Moradia', 'Tamanho_Familia',
       'Situacao_Pais', 'Educacao_Mae', 'Educacao_Pai', 'Trabalho_Mae',
       'Trabalho_Pai', 'Motivo_Escolha_Escolar', 'Responsavel_Legal',
       'Tempo_ida_Escola', 'Tempo_Estudo_Semanal', 'Apoio_Educacao_Extra',
       'Apoio_Educacao_Pais', 'Aulas_Particulares', 'Ativ_Extracurricular',
       'Frequentou_Creche', 'Tem_Internet',
       'Esta_Namorando', 'Boa_Convivencia_Familia', 'Tempo_Livre_Apos_Escola',
       'Tempo_com_Amigos', 'Alcool_Dia_Util', 'Alcool_Fim_Semana',
       'Estado_Saude', 'Falta_Escolar']

df = df.drop(columns=columns_to_drop, errors='ignore')

df = df.dropna()


le = LabelEncoder()
df['Quer_Fazer_Graduacao'] = le.fit_transform(df['Quer_Fazer_Graduacao'])

X = df.drop('Nota_2Semestre', axis=1, errors='ignore')
y = df['Nota_2Semestre']

modelo = BaggingClassifier()
modelo.fit(X, y)


# Salvar o modelo
joblib.dump(modelo, 'interface_de_classifica.pkl')


@app.route('/',methods=['GET','POST'])
def index():
    prediction = None
    if request.method == 'POST':
        Quer_Fazer_Graduacao = int(request.form.get('Quer_Fazer_Graduacao'))
        Nota_1Semestre = int(request.form.get('Nota_1Semestre'))

        # Criar DataFrame para a predição
        input_data = pd.DataFrame({'Quer_Fazer_Graduacao': [Quer_Fazer_Graduacao], 'Nota_1Semestre': [Nota_1Semestre]})

        # Realizar a predição
        predicted_nota_2_semestre = modelo.predict(input_data)[0]

        return render_template('results.html', Quer_Fazer_Graduacao=Quer_Fazer_Graduacao,
                               Nota_1Semestre=Nota_1Semestre,
                               predicted_nota_2_semestre=predicted_nota_2_semestre)

    return render_template('form.html', prediction=prediction)


@app.route('/results',methods=['GET'])
def results():
    return render_template('results.html')

if __name__=='__main__':
    app.run(debug=True)
