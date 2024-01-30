import pandas as pd
import numpy as np
import dill

from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')


# Удаляем фичи с большим количеством пропусков
def filter_data(df):
    columns_to_drop = ['utm_keyword',
                       'device_os',
                       'device_model']

    return df.drop(columns_to_drop, axis=1)


# Сокращаем редкие категории в фичах
def df_cut(df):
    df_cuts = pd.DataFrame(df.copy())

    for col in df_cuts.columns:
        tmp_dict = dict(df_cuts[col].value_counts(normalize=True, dropna=False).sort_values(ascending=False) * 100)
        df_cuts[col] = list(map(lambda x: x if tmp_dict[x] > 0.05 else 'other', df_cuts[col]))
    return df_cuts


def print_hi(name):
    print('Avto event action predict pipeline')

    # Загружаем данные для обучения
    df_session = pd.read_csv('data/ga_sessions.csv')
    df_hits = pd.read_csv('data/ga_hits.csv')

    # Выбираем целевые действия
    action_list = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                   'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                   'sub_submit_success', 'sub_car_request_submit_click']

    # Создаем целевой признак
    df_hits['binar_action'] = np.where(df_hits['event_action'].isin(action_list), 1, 0)

    # Убираем лишнее из df_hits
    df_hits_clear = df_hits.loc[:, ['session_id', 'binar_action']].groupby('session_id').sum().sort_values(
        by='binar_action')

    # Делаем целевой признак бинарным
    df_hits_clear['binar_action'] = np.where(df_hits_clear['binar_action'] > 0, 1, 0)

    # Join df_session и df_hits_clear и убираем лишние признаки
    df_session = df_session.set_index('session_id')
    df_joined = df_session.join(df_hits_clear, how='inner')
    df_total = df_joined.drop(['client_id', 'visit_date', 'visit_time', 'visit_number'],
                              axis=1).reset_index(drop=True)

    # Убираем дубликаты
    df_total_dd = df_total[~df_total.duplicated()]
    df_total_dd.reset_index(drop=True, inplace=True)

    X = df_total_dd.drop('binar_action', axis=1).copy()
    Y = df_total_dd['binar_action'].copy()

    # Датафрейм готов, начинаем писать pipeline обработки для предикта.
    categorical_features = make_column_selector(dtype_include=object)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='other')),
        ('cut_feature', FunctionTransformer(df_cut)),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    cat_transformer = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = [
        LogisticRegression(solver='saga', n_jobs=-1, random_state=42),
        RandomForestClassifier(n_jobs=-1, random_state=42)
    ]

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('filter', FunctionTransformer(filter_data)),
            ('preprocessor', cat_transformer),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, Y, cv=3, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    # Обучаем лучшую модель на всём датафрейме
    best_pipe.fit(X, Y)

    # Создаем переменную, в которой топ 20 фич по влиянию на целевую фичу
    coefs = pd.Series(
        data=best_pipe.named_steps['classifier'].coef_[0],
        index=best_pipe.named_steps['preprocessor']\
            .transformers_[0][1]\
            .named_steps['encoder']\
            .get_feature_names_out()
    ).sort_values(ascending=False)
    top_20_features = coefs.head(20)

    # Сохраняем модель
    with open('models/pipline_model.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Car event_action predict model',
                'author': 'Dmitriy Dmitriev',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps['classifier']).__name__,
                'roc_auc': best_score.mean()
            },
            'top_features': top_20_features,
        }, file)


if __name__ == '__main__':
    print_hi('PyCharm')
