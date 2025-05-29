```markdown
# Predictive Maintenance Project

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 О проекте

Streamlit-приложение для прогнозирования отказов промышленного оборудования с использованием методов машинного обучения. Проект решает задачу бинарной классификации (отказ/исправность) на основе данных датчиков оборудования.

## 🛠 Технологии

- **Python 3.8+**
- **Streamlit** - веб-интерфейс
- **Scikit-learn** - модели машинного обучения
- **XGBoost** - градиентный бустинг
- **Pandas** - обработка данных
- **Matplotlib/Seaborn** - визуализация


## 📂 Структура проекта
predictive_maintenance_project/
├── app.py                 # Основной файл приложения
├── analysis_and_model.py  # Логика анализа и модели
├── presentation.py        # Презентация проекта
├── requirements.txt       # Зависимости
├── README.md              # Этот файл
├── data/                  # Данные
│   └── predictive_maintenance.csv
└── video/                 # Демонстрация
    └── demo.mp4


## 🚀 Быстрый старт
```
1. Клонируйте репозиторий:
```
git clone https://github.com/ваш_username/predictive_maintenance_project.git
cd predictive_maintenance_project
```

2. Установите зависимости:
```
pip install -r requirements.txt

```
3. Запустите приложение:
```
streamlit run app.py
```

Приложение будет доступно по адресу: [http://localhost:8501](http://localhost:8501)

## 🔍 Функционал приложения

### 📊 Страница анализа
- Загрузка CSV-файлов с данными
- Предобработка данных (масштабирование, кодирование)
- Обучение трех моделей:
  - Логистическая регрессия
  - Случайный лес
  - XGBoost
- Визуализация метрик:
  - Accuracy
  - ROC-AUC
  - Confusion Matrix

### 🎤 Страница презентации
- Описание проекта
- Описание датасета
- Этапы работы
- Результаты

## 📈 Используемые метрики

- **Accuracy** - общая точность модели
- **ROC-AUC** - площадь под ROC-кривой
- **Confusion Matrix** - матрица ошибок

## 📚 Датасет

Используется синтетический датасет "AI4I 2020 Predictive Maintenance Dataset":
- 10,000 записей
- 14 признаков
- 5 типов отказов оборудования

Подробнее: [UCI Repository](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)

