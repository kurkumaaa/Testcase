# DataPreprocessor

Класс для базовой обработки и трансформации табличных данных в формате pandas DataFrame.

## Описание

`DataPreprocessor` — это Python-класс, который выполняет основные операции по очистке и подготовке данных для машинного обучения:

- **Удаление столбцов с высокой долей пропусков** и заполнение оставшихся пропусков
- **One-Hot кодирование** категориальных (строковых) переменных
- **Нормализация и стандартизация** числовых признаков
- **Сохранение параметров** для применения одинакового pipeline к новым данным

## Особенности

✅ Полная поддержка обработки ошибок и валидации параметров  
✅ Сохранение информации о всех применённых преобразованиях  
✅ Возможность применения соохранённого pipeline к новым данным (метод `transform`)  
✅ Поддержка различных методов заполнения пропусков (mean, median, mode)  
✅ Два метода нормализации: Min-Max и стандартизация (Z-score)  
✅ Подробное логирование всех операций

## Установка

```bash
pip install pandas numpy scikit-learn
```

## Быстрый старт

### Базовое использование

```python
from data_preprocessor import DataPreprocessor
import pandas as pd

# Загрузить данные
df = pd.read_csv('data.csv')

# Создать preprocessor и применить все преобразования
preprocessor = DataPreprocessor(df)
df_processed = preprocessor.fit_transform(
    remove_missing_threshold=0.5,
    fill_method='mean',
    normalize_method='minmax'
)
```

### Применение к новым данным

```python
# Применить сохранённый pipeline к тестовым данным
df_test_processed = preprocessor.transform(df_test)
```

## Методы класса

### `__init__(df: pd.DataFrame)`
Инициализация класса с DataFrame.

**Параметры:**
- `df` (pd.DataFrame): Входные данные

**Исключения:**
- `TypeError`: Если аргумент не является DataFrame
- `ValueError`: Если DataFrame пуст

---

### `remove_missing(threshold: float = 0.5, fill_method: str = 'mean') → DataPreprocessor`

Удаляет столбцы с высокой долей пропусков и заполняет оставшиеся.

**Параметры:**
- `threshold` (float, 0-1): Доля пропусков, при превышении столбец удаляется
- `fill_method` (str): Метод заполнения ('mean', 'median', 'mode')

**Возвращает:** `self` для цепочки методов

---

### `encode_categorical() → DataPreprocessor`

Выполняет One-Hot кодирование всех строковых (категориальных) столбцов.

**Возвращает:** `self` для цепочки методов

---

### `normalize_numeric(method: str = 'minmax') → DataPreprocessor`

Нормализует числовые столбцы выбранным методом.

**Параметры:**
- `method` (str): Метод нормализации ('minmax' или 'std')

**Возвращает:** `self` для цепочки методов

---

### `fit_transform(remove_missing_threshold: float = 0.5, fill_method: str = 'mean', normalize_method: str = 'minmax') → pd.DataFrame`

Последовательно применяет все три преобразования.

**Параметры:**
- `remove_missing_threshold` (float): Порог удаления столбцов
- `fill_method` (str): Метод заполнения пропусков
- `normalize_method` (str): Метод нормализации

**Возвращает:** Обработанный DataFrame

---

### `transform(df_new: pd.DataFrame) → pd.DataFrame`

Применяет сохранённые преобразования к новому DataFrame.

**Параметры:**
- `df_new` (pd.DataFrame): Новый DataFrame для преобразования

**Возвращает:** Преобразованный DataFrame

---

### `get_preprocessing_info() → Dict`

Возвращает информацию о всех применённых преобразованиях.

**Возвращает:** Словарь с параметрами и результатами

---

### `get_feature_names() → List[str]`

Возвращает список имён признаков после преобразований.

**Возвращает:** Список имён столбцов

## Примеры использования

Полные примеры использования с различными датасетами можно найти в файле `demonstration.ipynb`.

### Пример 1: Базовая обработка

```python
from data_preprocessor import DataPreprocessor
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Загрузить датасет
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Создать и применить preprocessor
preprocessor = DataPreprocessor(df)
df_clean = preprocessor.fit_transform(
    remove_missing_threshold=0.5,
    fill_method='mean',
    normalize_method='minmax'
)

print(df_clean.head())
```

### Пример 2: Разделение на train/test

```python
from sklearn.model_selection import train_test_split

# Разделить данные
X_train, X_test = train_test_split(df, test_size=0.2)

# Обучить preprocessor на тренировочных данных
preprocessor = DataPreprocessor(X_train)
X_train_processed = preprocessor.fit_transform()

# Применить к тестовым данным
X_test_processed = preprocessor.transform(X_test)
```

### Пример 3: Получение информации о преобразованиях

```python
preprocessor = DataPreprocessor(df)
preprocessor.fit_transform()

# Получить информацию
info = preprocessor.get_preprocessing_info()

print(f"Удалено столбцов: {info['removed_columns']}")
print(f"Категориальные столбцы: {info['categorical_columns']}")
print(f"Числовые столбцы: {info['numeric_columns']}")
print(f"Итоговая форма: {info['data_shape']}")
```

## Требования

- Python 3.7+
- pandas
- numpy
- scikit-learn

## Файлы проекта

- `data_preprocessor.py` — Основной модуль с классом DataPreprocessor
- `demonstration.ipynb` — Jupyter notebook с примерами использования
- `README.md` — Этот файл

## Лицензия

MIT License

## Автор

GitHub Copilot

## Дополнительные замечания

Класс разработан для демонстрационных целей и примеров использования. Для производства рекомендуется использовать специализированные библиотеки, такие как `scikit-learn` Pipeline или `feature-engine`.
