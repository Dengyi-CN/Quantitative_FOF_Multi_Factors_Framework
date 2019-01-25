import pandas as pd
import numpy as np
from Quantitative_FOF_Multi_Factors_Framework.DataBase import read_data_from_oracle_db


def get_factor_info():
    factor_library = read_data_from_oracle_db('select * from lyzs_tinysoft.factor_library')
    # factor_library = pd.read_excel('/Users/yi.deng/Desktop/file/database/因子列表-初步检测.xlsx')
    factor_list = factor_library['FACTOR_NUMBER'].tolist()
    factor_name_dict = {factor_library.loc[i, 'FACTOR_NUMBER']: factor_library.loc[i, 'FACTOR_NAME'] for i in range(factor_library.shape[0])}
    factor_second_class_dict = {factor_library.loc[i, 'FACTOR_NUMBER']: factor_library.loc[i, 'SECOND_CLASS'] for i in
                                range(factor_library.shape[0])}
    factor_first_class_dict = {factor_library.loc[i, 'FACTOR_NUMBER']: factor_library.loc[i, 'FIRST_CLASS'] for i in range(factor_library.shape[0])}
    return factor_list, factor_name_dict, factor_first_class_dict, factor_second_class_dict


def fill_factor_number_info(data):
    factor_list, factor_name_dict, factor_first_class_dict, factor_second_class_dict = get_factor_info()
    if factor_name_dict:
        data['factor_name'] = data['factor_number'].map(factor_name_dict)
    if factor_first_class_dict:
        data['first_class'] = data['factor_number'].map(factor_first_class_dict)
    if factor_second_class_dict:
        data['second_class'] = data['factor_number'].map(factor_second_class_dict)
    return data
