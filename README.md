# salary-prediction
Salary prediction task for Rabota.ru

## Demo

![Demo](data/demo.gif)

## Technologies

- Python
- Streamlit for a web-service
- Pandas and Jupyter notebooks for data preprocessing and analysis
- DVC for dataset versioning

### Methodology

## Data Preprocessing

## Training Pipeline

### Dataset

В качестве разбиения на `train` `test` `validation` была выбрана пропорция 80:10:10.
Из-за того, что датасет изначально 
был сбалансирован проводилась случайная выборка без дополнительной стратификации. В процессе обучения, модель видела
только `train` множество, после каждой эпохи считались метрики на `validation` множестве, чтобы предотврадить
переобучение (где применимо) и обученная модель валидировалась на `test` множестве

### Metrics

## Testing and Validation

### Experimental Setup

В качестве экспериментов были обучены и провалидированы декартово множество всех моделей и всех эмбедингов на всех
обозначенных выше метриках. Результат считался положительным, если при кросс-валидации на тестовом датасете метрика
превышала 0.5, так как
данный результат может
считаться не случайным.

### Experiments

| Model                     | Embedding                      | Brier score | ROC-AUC | 
|---------------------------|--------------------------------|-------------|---------|
| **Linear Regression**     | **RuBERT-tiny**                | 0.35        | 0.36    |
| **Linear Regression**     | **RuBERT**                     | 0.21        | 0.29    |
| **Linear Regression**     | **FastText**                   | 0.36        | 0.33    |
| **Linear Regression**     | **Sentence RuBERT**            | 0.41        | 0.36    |
| **Linear Regression**     | **Sentence Multilingual BERT** | 0.24        | 0.22    |
| **CatBoost Regressor**    | **RuBERT-tiny**                | 0.63        | 0.51    |
| **CatBoost Regressor**    | **RuBERT**                     | 0.62        | 0.69    |
| **CatBoost Regressor**    | **FastText**                   | 0.62        | 0.58    |
| **CatBoost Regressor**    | **Sentence RuBERT**            | 0.63        | 0.59    |
| **CatBoost Regressor**    | **Sentence Multilingual BERT** | 0.61        | 0.69    |
| **Two Layers Perceptron** | **RuBERT-tiny**                | 0.66        | 0.74    |
| **Two Layers Perceptron** | **RuBERT**                     | 0.74        | 0.63    |
| **Two Layers Perceptron** | **FastText**                   | 0.61        | 0.62    |
| **Two Layers Perceptron** | **Sentence RuBERT**            | 0.6         | 0.61    |
| **Two Layers Perceptron** | **Sentence Multilingual BERT** | 0.7         | 0.61    |

## Deployment

Все модели были реализованы с использованием абстрактных интерфейсов, что позволяет единообразно использовать их в
сервисе и при необходимости добавлять новые модели. Взаимодействие моделей с пользователем происходит через streamlit
web-service. При загрузке своего резюме пользователю 

## Product Details

### Context

### Interface

Изначально, пользователю предлагается загрузить свое резюме -- мы реализовали подгрузку резюме в формате Word, так как
это самый популярный формат резюме в 2023 году. После того, как пользователь загрузил свое резюме оно отобразится справа
на главной странице, 

### Scaling

Для того, чтобы улучшить пользовательский опыт пользования нашим сервисом, мы выбрали использование продвинутных
эмбеддингов, а не тяжелых моделей. Данное решение позволяет посчитать эмбеддинги датасета отдельно в оффлайн режиме, а в
онлайне обрабатывать только пользовательские запросы, что положительно сказалось на общей производительности.

Помимо этого, мы реализовали сервис и бэкэнд, используя парадигму ООП и реализовали соответствующие интерфейсы, таким
образом масштабировать нас сервис и добавлять функциональность очень удобно.

## How to run

Для того, чтобы запустить web сервис изначально нужно восстановить виртуальное окружение:

```shell
poetry shell
poetry install --with web
```

И запустить сервис

```shell
poetry run streamlit run web/app.py
```

PS Первый раз запуск может быть более длительным из-за скачивания необходимых эмбеддингов
