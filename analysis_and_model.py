import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Предобработка данных
        st.header("Предобработка данных")
        data = data.drop(columns=['UDI', 'Product ID'])
        data['Type'] = LabelEncoder().fit_transform(data['Type'])
        
        # Масштабирование числовых признаков
        numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        
        st.write("Первые 5 строк после предобработки:")
        st.write(data.head())
        
        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Обучение моделей
        st.header("Обучение моделей")
        
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        results = {}
        fig, ax = plt.subplots()
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Сохранение результатов
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "conf_matrix": confusion_matrix(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba)
            }
            
            # ROC кривая
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.2f})")
        
        # Отображение результатов
        st.subheader("Результаты моделей")
        for name, res in results.items():
            st.write(f"#### {name}")
            st.write(f"Accuracy: {res['accuracy']:.4f}")
            st.write(f"ROC-AUC: {res['roc_auc']:.4f}")
            
            st.write("Confusion Matrix:")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(res['conf_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            st.pyplot(fig_cm)
        
        # ROC кривые
        st.subheader("ROC кривые")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC кривые')
        ax.legend()
        st.pyplot(fig)
        
        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            product_type = st.selectbox("Тип продукта", ["L", "M", "H"])
            air_temp = st.number_input("Температура окружающей среды [K]", value=300.0)
            process_temp = st.number_input("Рабочая температура [K]", value=310.0)
            rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
            torque = st.number_input("Крутящий момент [Nm]", value=40.0)
            tool_wear = st.number_input("Износ инструмента [min]", value=0)
            
            submit_button = st.form_submit_button("Предсказать")
        
        if submit_button:
            # Преобразование введенных данных
            input_data = pd.DataFrame({
                'Type': [0 if product_type == 'L' else 1 if product_type == 'M' else 2],
                'Air temperature [K]': [(air_temp - 300) / 2],  # Примерное масштабирование
                'Process temperature [K]': [(process_temp - 310) / 1],
                'Rotational speed [rpm]': [(rotational_speed - 1500) / 100],
                'Torque [Nm]': [(torque - 40) / 10],
                'Tool wear [min]': [tool_wear / 10]
            })
            
            # Выбор лучшей модели
            best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
            best_model = models[best_model_name]
            
            # Предсказание
            prediction = best_model.predict(input_data)
            prediction_proba = best_model.predict_proba(input_data)[:, 1]
            
            st.write(f"Лучшая модель: {best_model_name}")
            st.write(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
            st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")