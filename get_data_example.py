import pandas as pd
import json

df_session = pd.read_csv('data/ga_sessions.csv')

cols_to_drop = ['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number']

df_clear = df_session.drop(cols_to_drop,
                           axis=1).reset_index(drop=True)

data = df_clear.sample().to_dict('records')[0]
file_name = df_clear.sample().index[0]

with open(f'test/{file_name}.json', 'w', encoding="utf-8") as file:
    json.dump(data, file)
