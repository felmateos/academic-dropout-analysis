import pandas as pd
from itertools import combinations


def remove_unused(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Remove colunas que nao serao usadas.
    
    :param df: Dataframe alvo da transformacao
    :type df: pd.DataFrame
    :return: Dataframe transformado
    :rtype: pd.DataFrame
    """
    df_cp = df.copy()
    df_cp = df.drop(cols, axis=1)
    return df_cp


def one_hot_encoding(df: pd.DataFrame, categorical_cols: list, drop_first=False) -> pd.DataFrame:
    """
    Executa o One Hot Encoding em cada coluna fornecida de um DataFrame.

    :param df: DataFrame alvo das transformacoes
    :df type: pd.DataFrame
    :param cols: Lista de colunas que devem ser transformadas
    :cols type: list
    :return: DataFrame transformado
    :rtype: pd.Dataframe
    """

    missing_cols = [col for col in categorical_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas categóricas ausentes no DataFrame: {missing_cols}")
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df_encoded


def save_transformed_data(df: pd.DataFrame, transformations: dict, filename: str):
    """
    
    """

    transformed_df = df.copy()
    for transformation, params in transformations:
        transformed_df = transformation(transformed_df, **params)
    transformed_df.to_csv(filename, index=False)
    print(f"Arquivo salvo: {filename}")



def preprocess_data(df: pd.DataFrame, transformations: dict, filename: str):
    """
    
    """
    
    transformation_names = list(transformations.keys())
    for r in range(1, len(transformation_names) + 1):
        for combo in combinations(transformation_names, r):
            filename_combo = f"{filename}_{'_'.join(combo)}.csv"
            transformation_funcs = [(transformations[name][0], transformations[name][1]) for name in combo]
            save_transformed_data(df, transformation_funcs, filename_combo)