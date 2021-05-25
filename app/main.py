import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64

st.set_option('deprecation.showfileUploaderEncoding', False)

def main():
    st.sidebar.image('churn_bunner.png', width=310)
    # funçãp para carregar o dataset
    #@st.cache(allow_output_mutation=True) # coloca o streamlit em cache para acelerar o processo
    def get_data():
        return pd.read_csv("../dados/test.csv")

    # função para importar o modelo treinado
    def load_model():
        return pickle.load(open('../notebook/RF_model_trained.sav', 'rb'))

    # função para fazer as previsões
    def predict_chur():
        data_new = data.drop(['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes'], axis=1)
        return loaded_model.predict(data_new)


    # titulo do app
    st.title("Previsão de Churn")

    # subtitulo
    st.markdown("""
    DEFINIÇÃO: Churn é uma métrica para saber o quanto houve de perda de clientes em uma empresa, ou seja, calcula o índice de
    cancelamento de clientes em um determinado período.""")
    st.markdown("""
    Este app tem o objetivo de prever a probabilidade se o CHURN será positivo ou negativo para determinado cliente, com base nas suas características.
    """)


    # importando o modelo treinado
    loaded_model = load_model()

    st.sidebar.title("File uploader")

    menu = ["Selecione", "Arrastar arquivo", 'Demonstração']
    choice = st.sidebar.selectbox("Selecione", menu)


    if choice == "Arrastar arquivo":
        st.sidebar.subheader("Arraste aqui!")
        file_dragged = st.sidebar.file_uploader("Upload csv", type=["csv"])
        if file_dragged is not None:
            data = pd.read_csv(file_dragged)

            # exibindo os top 10 primeiros registos do dataframe
            st.dataframe(data.head(10))

    if choice == "Demonstração":
        data = get_data()

        # criando um dataframe
        #data = get_data()

        # exibindo os top 10 primeiros registos do dataframe
        # st.dataframe(data.head(10))

        st.sidebar.header("O que deseja fazer?")

        menu_1 = ["Selecione", "Analisar", "Previsões", "Simulação"]
        choice = st.sidebar.selectbox("Selecione", menu_1)

        if choice == "Analisar":
            rows_exib = st.sidebar.number_input("Total de linhas exibidas", min_value=4, max_value=data.shape[0])

            st.header("Visualize o arquivo")
            st.subheader("As primeiras linhas")
            st.dataframe(data.head(rows_exib))
            st.subheader("As últimas linhas")
            st.dataframe(data.tail(rows_exib))
            st.subheader("Há faltas, os tipos estão corretos, e quantos valores únicos possui?")
            st.dataframe(pd.DataFrame({"dados_nulos": data.isna().mean(),
                                       "tipo_dados": data.dtypes,
                                       "valores_unicos": data.nunique()}))
            st.subheader("Resumo das estatísticas para dados numéricos")
            st.write(data.describe())
            st.subheader("Resumo das estatísticas para dados categóricos")
            st.write(data.describe(include="O"))

            st.sidebar.text("As dimensões em linhas e colunas:")
            st.sidebar.text(data.shape)

            st.sidebar.text("Que tipo de dados quer analisar?")
            tipo_dados = st.sidebar.radio('Selecione', ['Categóricos', 'Numéricos'])

            if tipo_dados == "Numéricos":
                st.header("Visualize os gráficos do tipo numérico")
                data_num = data.select_dtypes(include=np.number)

                # removendo algumas colunas
                colunas = data_num.columns

                # contando a quantidade de features para o loop
                # num_plots = len(colunas)

                # definindo a área de plotagem
                nrow = 4
                ncol = 4
                fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 20))
                fig.subplots_adjust(hspace=1, wspace=1)

                # plotando gráfico de densidade
                idx = 0
                for col in colunas:
                    idx += 1
                    plt.subplot(nrow, ncol, idx)
                    sns.kdeplot(data_num[col], label="No", shade=True, color='blue')
                    plt.title(col, fontsize=14, fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)

            if tipo_dados == "Categóricos":
                st.header("Visualize os gráficos do tipo categórico")
                data_cat = data.select_dtypes(include='object')
                # definindo a área de plotagem
                fig = plt.figure(figsize=(20, 12))

                # plotando o gráfico por area_code
                ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3, rowspan=1)
                #sns.barplot(x='state', y='%churn', data=state_churn, palette='dark:salmon')
                sns.countplot(x='state', data=data_cat, ax=ax1,
                              palette=["blue"], alpha=0.6)
                _, xlabels = plt.xticks()
                ax1.set_xticklabels(xlabels, size=12, fontweight='bold')
                ax1.set(xlabel=None)
                ax1 = plt.title("% churn de Clientes por estado", fontsize=25, fontweight='bold')

                ax2 = plt.subplot2grid((2, 3), (1, 0))
                area_code_plot = sns.countplot(x='area_code', data=data_cat, ax=ax2,
                                               palette=["blue"], alpha=0.6)
                _, xlabels = plt.xticks()
                ax2.set_xticklabels(xlabels, size=15, fontweight='bold')
                ax2.set(xlabel=None)
                ax2 = plt.title("Clientes por código da área", fontsize=25, fontweight='bold')

                # plotando o gráfico por international_plan
                ax3 = plt.subplot2grid((2, 3), (1, 1))
                inter_plan = sns.countplot(x='international_plan', data=data_cat, ax=ax3,
                                           palette=["blue"], alpha=0.6)
                _, xlabels = plt.xticks()
                ax3.set_xticklabels(xlabels, size=15, fontweight='bold')
                ax3.set(xlabel=None)
                ax3 = plt.title("Clientes por plano internacional", fontsize=25, fontweight='bold')

                # plotando o gráfico por voice_mail_plan
                ax4 = plt.subplot2grid((2, 3), (1, 2))
                voice_mail = sns.countplot(x='voice_mail_plan', data=data_cat, ax=ax4,
                                           palette=["blue"], alpha=0.6)
                _, xlabels = plt.xticks()
                ax4.set_xticklabels(xlabels, size=15, fontweight='bold')
                ax4.set(xlabel=None)
                ax4 = plt.title("Clientes por plano de voice-mail", fontsize=25, fontweight='bold')

                # exibindo os gráficos
                plt.tight_layout()
                plt.show()
                st.pyplot(fig)

        if choice == "Previsões":
            st.header('Arquivo importado com as características dos clientes')
            st.dataframe(data.head(10))
            st.write("linhas, colunas", data.shape)

            data_df = data.copy()
            st.header('Arquivo com a previsões do modelo para cada cliente')
            data_new = data.drop(['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes'], axis=1)
            data_df["Previsão_Churn"] = loaded_model.predict(data_new)
            data_df["Proba_no"] = loaded_model.predict_proba(data_new)[:, 0]
            data_df["Proba_yes"] = loaded_model.predict_proba(data_new)[:, 1]
            data_df['Previsão_Churn'] = data_df['Previsão_Churn'].map({0:"Não", 1:"Sim"})
            st.dataframe(data_df.head(10))
            st.write("linhas, colunas", data_df.shape)
            st.write("Foram inseridas mais três colunas, previsão, probabilidade de sim e probabilidade de não churn.")

            st.title('Downloader')

            download = st.button('Clique para baixar')

            if download:
                # 'Download Started!'
                csv = data_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings
                linko = f'<a href="data:file/csv;base64,{b64}" download="Churn_Predict.csv">Download csv file</a>'
                st.markdown(linko, unsafe_allow_html=True)


        if choice == "Simulação":
            st.sidebar.subheader("Defina abaixo:")
            state_sigla = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
                           'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
                           'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
                           'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA',
                           'WI', 'WV', 'WY']
            state = st.sidebar.selectbox('state', state_sigla)
            account_length = st.sidebar.number_input("account_length", value=round(data.account_length.mean(), 2))
            area_code = st.sidebar.selectbox('area_code', ['area_code_408', 'area_code_415', 'area_code_510'])
            international_plan = st.sidebar.radio('international_plan', ['Sim', 'Não'])
            voice_mail_plan = st.sidebar.radio('voice_mail_plan', ['Sim', 'Não'])
            number_vmail_messages = st.sidebar.number_input("number_vmail_messages", value=data.number_vmail_messages.mean())
            total_day_calls = st.sidebar.number_input("total_day_calls", value=data.total_day_calls.mean())
            total_day_charge = st.sidebar.number_input("total_day_charge", value=data.total_day_charge.mean())
            total_eve_calls = st.sidebar.number_input("total_eve_calls", value=data.total_eve_calls.mean())
            total_eve_charge = st.sidebar.number_input("total_eve_charge", value=data.total_eve_charge.mean())
            total_night_calls = st.sidebar.number_input("total_night_calls", value=data.total_night_calls.mean())
            total_night_charge = st.sidebar.number_input("total_night_charge", value=data.total_night_charge.mean())
            total_intl_calls = st.sidebar.number_input("total_intl_calls", value=data.total_intl_calls.mean())
            total_intl_charge = st.sidebar.number_input("total_intl_charge", value=data.total_intl_charge.mean())
            number_customer_service_calls = st.sidebar.number_input("number_customer_service_calls", value=data.number_customer_service_calls.mean())

            #df_const = {state: state}
            local_df = pd.DataFrame({'state': state, 'account_length': account_length, 'area_code': area_code,
                                     'international_plan': international_plan, 'voice_mail_plan': voice_mail_plan,
                                     'number_vmail_messages': number_vmail_messages,
                                     'total_day_calls': total_day_calls, 'total_day_charge': total_day_charge,
                                     'total_eve_calls': total_eve_calls, 'total_eve_charge': total_eve_charge,
                                     'total_night_calls': total_night_calls,
                                     'total_night_charge': total_night_charge,
                                     'total_intl_calls': total_intl_calls, 'total_intl_charge': total_intl_charge,
                                     'number_customer_service_calls': number_customer_service_calls}, index=[0])

            # inserindo um botão na tela
            btn_predict = st.sidebar.button("Realizar predição")

            st.dataframe(local_df)

            local_df["Previsoes"] = loaded_model.predict(local_df)
            local_df["Proba_no"] = loaded_model.predict_proba(local_df)[:, 0]
            local_df["Proba_yes"] = loaded_model.predict_proba(local_df)[:, 1]

            if btn_predict:
                str_pred = {0: "Não", 1: "Sim"}
                st.header("O Cliente irá cancelar o contrado?")

                result_pred = str(local_df["Previsoes"].map(str_pred)[0])
                st.write(result_pred)

                result_prob_no = "A probabilidade estimada é de " + str(round(local_df["Proba_no"][0]*100, 2)) + "%"
                result_prob_yes = "A probabilidade estimada é de " + str(round(local_df["Proba_yes"][0]*100, 2)) + "%"
                st.write(str(np.where(local_df["Previsoes"] == 0, result_prob_no, result_prob_yes)))



if __name__ == '__main__':
    main()