"""
DataPreprocessor - класс для базовой обработки и трансформации табличных данных.
Выполняет очистку, кодирование категориальных переменных и нормализацию признаков.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings


class DataPreprocessor:
    """
    Класс для выполнения базовых операций по очистке и трансформации табличных данных.
    
    Поддерживает:
    - Удаление столбцов с высокой долей пропусков и заполнение оставшихся пропусков
    - One-hot кодирование категориальных столбцов
    - Нормализацию/стандартизацию числовых столбцов
    - Сохранение параметров трансформации для применения к новым данным
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация класса.
        
        Args:
            df: pandas DataFrame для обработки
            
        Raises:
            TypeError: Если аргумент не является DataFrame
            ValueError: Если DataFrame пуст
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Ожидается pandas DataFrame, получен {type(df)}")
        if df.empty:
            raise ValueError("DataFrame не может быть пустым")
        
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
        
        # Сохранение параметров трансформации
        self.removed_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.ohe_columns: List[str] = []  # новые столбцы после one-hot encoding
        self.new_column_names: Dict[str, List[str]] = {}  # маппинг исходных столбцов на новые
        
        # Скейлеры для применения к новым данным
        self.scalers: Dict[str, Union[MinMaxScaler, StandardScaler]] = {}
        self.scaler_method: Optional[str] = None
        
        # Средние значения для заполнения пропусков
        self.fill_values: Dict[str, float] = {}
    
    def remove_missing(self, threshold: float = 0.5, fill_method: str = 'mean') -> 'DataPreprocessor':
        """
        Удаляет столбцы с высокой долей пропусков и заполняет оставшиеся.
        
        Args:
            threshold: Доля пропусков (0-1), при превышении столбец удаляется. По умолчанию 0.5
            fill_method: Метод заполнения пропусков ('mean', 'median', 'mode'). По умолчанию 'mean'
            
        Returns:
            self для цепочки методов
            
        Raises:
            ValueError: Если threshold вне диапазона [0, 1] или неправильный fill_method
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold должен быть в диапазоне [0, 1], получен {threshold}")
        
        if fill_method not in ['mean', 'median', 'mode']:
            raise ValueError(f"fill_method должен быть одним из ['mean', 'median', 'mode'], получен {fill_method}")
        
        # Определение процента пропусков для каждого столбца
        missing_percent = self.df.isnull().sum() / len(self.df)
        
        # Столбцы, которые нужно удалить
        columns_to_drop = missing_percent[missing_percent > threshold].index.tolist()
        self.removed_columns = columns_to_drop
        
        if columns_to_drop:
            print(f"Удаляются столбцы с долей пропусков > {threshold}: {columns_to_drop}")
            self.df = self.df.drop(columns=columns_to_drop)
        
        # Заполнение оставшихся пропусков
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                if fill_method == 'mean' and pd.api.types.is_numeric_dtype(self.df[column]):
                    fill_value = self.df[column].mean()
                elif fill_method == 'median' and pd.api.types.is_numeric_dtype(self.df[column]):
                    fill_value = self.df[column].median()
                elif fill_method == 'mode':
                    mode_values = self.df[column].mode()
                    if len(mode_values) > 0:
                        fill_value = mode_values[0]
                    else:
                        fill_value = self.df[column].mean() if pd.api.types.is_numeric_dtype(self.df[column]) else 'unknown'
                else:
                    # Для категориальных столбцов используем mode
                    mode_values = self.df[column].mode()
                    fill_value = mode_values[0] if len(mode_values) > 0 else 'unknown'
                
                self.fill_values[column] = fill_value
                self.df[column].fillna(fill_value, inplace=True)
                print(f"Столбец '{column}' заполнен {fill_method} = {fill_value}")
        
        return self
    
    def encode_categorical(self) -> 'DataPreprocessor':
        """
        Выполняет one-hot encoding всех строковых (категориальных) столбцов.
        
        Returns:
            self для цепочки методов
        """
        # Определение категориальных столбцов
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            print("Нет категориальных столбцов для кодирования")
            return self
        
        self.categorical_columns = categorical_cols
        original_cols = set(self.df.columns)
        
        # One-hot encoding
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=False, dtype=int)
        
        # Определение новых столбцов
        new_cols = set(self.df.columns) - original_cols
        self.ohe_columns = sorted(list(new_cols))
        
        # Маппинг исходных столбцов на новые
        for cat_col in categorical_cols:
            matching_cols = [col for col in self.ohe_columns if col.startswith(cat_col + '_')]
            self.new_column_names[cat_col] = matching_cols
        
        print(f"Выполнен one-hot encoding для столбцов: {categorical_cols}")
        print(f"Создано новых столбцов: {len(self.ohe_columns)}")
        
        return self
    
    def normalize_numeric(self, method: str = 'minmax') -> 'DataPreprocessor':
        """
        Нормализует числовые столбцы выбранным методом.
        
        Args:
            method: Метод нормализации ('minmax' - Min-Max, 'std' - стандартизация)
            
        Returns:
            self для цепочки методов
            
        Raises:
            ValueError: Если method не является 'minmax' или 'std'
        """
        if method not in ['minmax', 'std']:
            raise ValueError(f"method должен быть 'minmax' или 'std', получен {method}")
        
        # Определение числовых столбцов
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("Нет числовых столбцов для нормализации")
            return self
        
        self.numeric_columns = numeric_cols
        self.scaler_method = method
        
        # Выбор скейлера
        if method == 'minmax':
            scaler = MinMaxScaler()
            print(f"Применяется Min-Max нормализация к столбцам: {numeric_cols}")
        else:  # 'std'
            scaler = StandardScaler()
            print(f"Применяется стандартизация к столбцам: {numeric_cols}")
        
        # Применение скейлера
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        
        # Сохранение скейлера для использования на новых данных
        for col in numeric_cols:
            self.scalers[col] = scaler
        
        return self
    
    def fit_transform(self,
                      remove_missing_threshold: float = 0.5,
                      fill_method: str = 'mean',
                      normalize_method: str = 'minmax') -> pd.DataFrame:
        """
        Последовательно применяет все три преобразования и возвращает обработанный DataFrame.
        
        Args:
            remove_missing_threshold: Порог удаления столбцов с пропусками (0-1)
            fill_method: Метод заполнения пропусков ('mean', 'median', 'mode')
            normalize_method: Метод нормализации ('minmax', 'std')
            
        Returns:
            DataFrame с применёнными преобразованиями
            
        Raises:
            ValueError: Если параметры некорректны
        """
        print("=" * 60)
        print("Начало обработки данных")
        print("=" * 60)
        
        self.remove_missing(threshold=remove_missing_threshold, fill_method=fill_method)
        print()
        
        self.encode_categorical()
        print()
        
        self.normalize_numeric(method=normalize_method)
        print()
        
        print("=" * 60)
        print("Обработка завершена")
        print(f"Исходный размер: {len(self.original_columns)} столбцов, {len(self.df)} строк")
        print(f"Итоговый размер: {len(self.df.columns)} столбцов, {len(self.df)} строк")
        print("=" * 60)
        
        return self.df.copy()
    
    def transform(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет сохранённые преобразования к новому DataFrame.
        Используется для применения одного и того же pipeline к тестовым данным.
        
        Args:
            df_new: Новый DataFrame для преобразования
            
        Returns:
            Преобразованный DataFrame
            
        Raises:
            TypeError: Если аргумент не является DataFrame
            ValueError: Если столбцы не совпадают с исходными
        """
        if not isinstance(df_new, pd.DataFrame):
            raise TypeError(f"Ожидается pandas DataFrame, получен {type(df_new)}")
        
        df_transformed = df_new.copy()
        
        # Удаление столбцов, которые были удалены при обучении
        if self.removed_columns:
            df_transformed = df_transformed.drop(columns=self.removed_columns, errors='ignore')
        
        # Заполнение пропусков теми же значениями
        for column, fill_value in self.fill_values.items():
            if column in df_transformed.columns and df_transformed[column].isnull().sum() > 0:
                df_transformed[column].fillna(fill_value, inplace=True)
        
        # One-hot encoding категориальных столбцов
        if self.categorical_columns:
            df_transformed = pd.get_dummies(df_transformed, columns=self.categorical_columns, 
                                          drop_first=False, dtype=int)
            
            # Выравнивание столбцов (возможны липшие или недостающие столбцы)
            expected_cols = set(self.original_columns) - set(self.removed_columns)
            expected_cols.update(self.ohe_columns)
            
            # Добавление недостающих столбцов
            for col in expected_cols:
                if col not in df_transformed.columns and col in self.ohe_columns:
                    df_transformed[col] = 0
            
            # Удаление лишних столбцов
            cols_to_keep = [col for col in df_transformed.columns if col in expected_cols]
            df_transformed = df_transformed[cols_to_keep]
        
        # Нормализация числовых столбцов
        if self.numeric_columns and self.scalers:
            scaler = next(iter(self.scalers.values()))
            numeric_cols_present = [col for col in self.numeric_columns if col in df_transformed.columns]
            if numeric_cols_present:
                df_transformed[numeric_cols_present] = scaler.transform(df_transformed[numeric_cols_present])
        
        return df_transformed
    
    def get_feature_names(self) -> List[str]:
        """
        Возвращает список имён признаков после всех преобразований.
        
        Returns:
            Список имён столбцов обработанного DataFrame
        """
        return self.df.columns.tolist()
    
    def get_preprocessing_info(self) -> Dict:
        """
        Возвращает информацию о применённых преобразованиях.
        
        Returns:
            Словарь с параметрами и результатами преобразований
        """
        return {
            'original_columns': self.original_columns,
            'removed_columns': self.removed_columns,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'ohe_columns': self.ohe_columns,
            'final_columns': self.df.columns.tolist(),
            'fill_values': self.fill_values,
            'scaler_method': self.scaler_method,
            'data_shape': self.df.shape
        }
