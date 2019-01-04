import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import pandas as pd
pd.set_option('max_columns', 20)
pd.set_option('display.width', 400)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
import numpy as np
import pickle
from Quantitative_FOF_Multi_Factors_Framework.DataBase import read_data_from_oracle_db
from Quantitative_FOF_Multi_Factors_Framework.Factor_Test.Stratification_Method import transform_dict_to_df, optimize_data_ram, pickle_dump_data
import itertools
from collections import Counter


def get_factor_info():
    factor_library = read_data_from_oracle_db('select * from lyzs_tinysoft.factor_library')
    # factor_library = pd.read_excel('/Users/yi.deng/Desktop/file/database/因子列表-初步检测.xlsx')
    factor_list = factor_library['FACTOR_NUMBER'].tolist()
    factor_name_dict = {factor_library.loc[i, 'FACTOR_NUMBER']: factor_library.loc[i, 'FACTOR_NAME'] for i in range(factor_library.shape[0])}
    factor_second_class_dict = {factor_library.loc[i, 'FACTOR_NUMBER']: factor_library.loc[i, 'SECOND_CLASS'] for i in
                                range(factor_library.shape[0])}
    factor_first_class_dict = {factor_library.loc[i, 'FACTOR_NUMBER']: factor_library.loc[i, 'FIRST_CLASS'] for i in range(factor_library.shape[0])}
    return factor_list, factor_name_dict, factor_first_class_dict, factor_second_class_dict


def fill_factor_number_info(data, factor_name_dict=None, factor_first_class_dict=None, factor_second_class_dict=None):

    if factor_name_dict:
        data['factor_name'] = data['factor_number'].map(factor_name_dict)
    if factor_first_class_dict:
        data['first_class'] = data['factor_number'].map(factor_first_class_dict)
    if factor_second_class_dict:
        data['sceond_class'] = data['factor_number'].map(factor_second_class_dict)
    return data


if __name__ == "__main__":

    # 测试
    output_file_url = 'D:/Quantitative_FOF_Framework_data/ResultDisplay_20181228'

    factor_list, factor_name_dict, factor_first_class_dict, factor_second_class_dict = get_factor_info()

    # factor_return_reg_columns = ['regression_model', 'rolling_window', 'factor_number', 'type_name', 'get_data_date',
    #                              'alpha_significance', 'alpha', 'alpha_ste', 'alpha_t_value', 'alpha_p_value',
    #                              'beta_significance', 'beta', 'beta_ste', 'beta_t_value', 'beta_p_value',
    #                              'adj_rsquared']
    #
    # factor_return_reg_result = read_data_from_oracle_db('select * from lyzs_tinysoft.factor_return_regression')[factor_return_reg_columns]
    # factor_return_reg_result = pickle.load(open('/Users/yi.deng/Desktop/file/database/factor_return_regression.dat', 'rb'))

    critirion = 0.1
    # signifcant_factor_return_reg = factor_return_reg_result[factor_return_reg_result['alpha_p_value'] <= critirion].copy()

    date_sql = '''
        select get_data_date
        from lyzs_tinysoft.get_data_date_library
        where get_data_freq = \'周度\'
        order by get_data_date
    '''
    all_get_data_date = read_data_from_oracle_db(date_sql)['GET_DATA_DATE']
    all_get_data_date.name = 'get_data_date'

    ranking_columns = ['regression_model', 'rolling_window', 'factor_number', 'type_name', 'get_data_date',
                       'alpha_significance', 'alpha', 'alpha_ste', 'alpha_t_value', 'alpha_p_value',
                       'beta_significance', 'beta', 'beta_ste', 'beta_t_value', 'beta_p_value',
                       'adj_rsquared']


    # # 每个日期下，每个模型、滚动窗口显著情况汇总显著个数
    #
    # # 1.1 时间序列
    #
    # primary_keys_name_list = ['regression_model', 'rolling_window', 'factor_number']
    # significant_pk_all_list = [signifcant_factor_return_reg[column_name].unique().tolist() for column_name in primary_keys_name_list]
    # primary_keys_list = list(itertools.product(*significant_pk_all_list))
    # week_data_rolling_pct_list = [4, 5, 6, 12, 36, 52]
    #
    # factor_return_reg_signifcant_ts_pct = {}
    #
    # for primary_key in primary_keys_list:
    #     temp_ts_pct_pivot_data = pd.DataFrame(index=all_get_data_date, columns=['第' + str(i + 1) + '档收益率' for i in range(10)])
    #     # 有的档位可能不存在显著的，所以导致该列pivot出来之后不存在
    #     temp_pivot_data = signifcant_factor_return_reg[signifcant_factor_return_reg[primary_keys_name_list].isin(primary_key).all(1)].pivot_table(
    #         index='get_data_date', columns='type_name', values='alpha')
    #     temp_ts_pct_pivot_data.loc[temp_pivot_data.index, temp_pivot_data.columns] = temp_pivot_data
    #
    #     temp_ts_pct_melt_data = temp_ts_pct_pivot_data.reset_index().rename(columns={'GET_DATA_DATE': 'get_data_date'}).melt(
    #         id_vars=['get_data_date'], var_name=['type_name'], value_name='alpha')
    #
    #     for rolling_num in week_data_rolling_pct_list:
    #         rolling_result = temp_ts_pct_melt_data.groupby(by=['type_name']).apply(
    #             lambda df: df['alpha'].rolling(rolling_num, min_periods=1).apply(lambda alpha: len(pd.Series(alpha).dropna()) / len(alpha))).reset_index()
    #         temp_ts_pct_melt_data.loc[rolling_result['level_1'], 'last_' + str(rolling_num) + '_week'] = rolling_result['alpha']
    #
    #     factor_return_reg_signifcant_ts_pct[primary_key] = temp_ts_pct_melt_data.copy()
    #     print('1.1 时间序列：' + '，'.join([str(pk) for pk in primary_key]))
    #
    # factor_return_reg_signifcant_ts_pct_df = transform_dict_to_df(factor_return_reg_signifcant_ts_pct, primary_keys_name_list)
    # factor_return_reg_signifcant_ts_pct_df = optimize_data_ram(factor_return_reg_signifcant_ts_pct_df)
    # pickle_dump_data(factor_return_reg_signifcant_ts_pct_df, output_file_url, data_name='factor_return_reg_signifcant_ts_pct_df')
    #
    # # 1.2.1 截面
    # factor_return_reg_signifcant = factor_return_reg_result.groupby(by=[ 'regression_model', 'rolling_window', 'factor_number', 'get_data_date']).apply(
    #     lambda df: sum(df['alpha_p_value'] <= critirion)).reset_index().rename(columns={0: '10档显著个数'})
    # factor_return_reg_significant_cs_pct = factor_return_reg_signifcant.groupby(by=['regression_model', 'rolling_window', 'get_data_date']).apply(
    #     lambda df: df.sort_values(by=['10档显著个数'], ascending=False)).reset_index(drop=True)
    # factor_return_reg_significant_cs_pct['10档显著占比'] = factor_return_reg_significant_cs_pct['10档显著个数'].div(10)
    # factor_return_reg_significant_cs_pct = \
    #     fill_factor_number_info(factor_return_reg_significant_cs_pct, factor_name_dict=None, factor_first_class_dict=factor_first_class_dict,
    #                             factor_second_class_dict=factor_second_class_dict)
    #
    # # 1.2.2 截面的时间序列表现
    #
    # for rolling_num in week_data_rolling_pct_list:
    #     temp_rolling = factor_return_reg_significant_cs_pct.groupby(by=['regression_model', 'rolling_window', 'factor_number']).apply(
    #         lambda cr_pct: cr_pct['10档显著占比'].rolling(rolling_num, min_periods=1).mean()).reset_index()
    #     factor_return_reg_significant_cs_pct.loc[temp_rolling['level_3'], 'last_' + str(rolling_num) + '_week'] = temp_rolling['10档显著占比'].values
    #
    # factor_return_reg_significant_cs_pct = optimize_data_ram(factor_return_reg_significant_cs_pct)
    # pickle_dump_data(factor_return_reg_significant_cs_pct, output_file_url, data_name='factor_return_reg_significant_cs_pct')
    #
    # # 2. 基于rank的因子显著性
    #
    #
    # # 2.1 基于alpha排序的
    #
    # ranking_sql = '''
    #     SELECT ''' + ','.join(ranking_columns) + ''',
    #         rank () OVER (PARTITION BY REGRESSION_MODEL, ROLLING_WINDOW, GET_DATA_DATE, FACTOR_NUMBER ORDER BY ALPHA DESC) rank
    #     FROM LYZS_TINYSOFT.FACTOR_RETURN_REGRESSION
    #     WHERE ALPHA_P_VALUE <= 0.1
    # '''
    #
    # ranking_alpha_data = read_data_from_oracle_db(ranking_sql)
    # ranking_alpha_data.columns = ranking_columns + ['rank']
    # ranking_alpha_data = ranking_alpha_data.sort_values(by=['regression_model', 'rolling_window', 'factor_number', 'type_name', 'get_data_date'])
    #
    # primary_keys_name_list = ['regression_model', 'rolling_window', 'factor_number']
    # significant_pk_all_list = [ranking_alpha_data[column_name].unique().tolist() for column_name in primary_keys_name_list]
    # primary_keys_list = list(itertools.product(*significant_pk_all_list))
    #
    # factor_number_sql = '''
    #     select factor_number
    #     from lyzs_tinysoft.factor_library
    #     order by factor_number
    # '''
    # factor_number_list = read_data_from_oracle_db(factor_number_sql)['FACTOR_NUMBER']
    # week_data_rolling_pct_list = [12, 24, 52]
    #
    # factor_return_alpha_ranking_consistency = {}
    #
    # for primary_key in primary_keys_list:
    #
    #     # primary_keys：('OLS', 32, 'factor1')
    #     temp_rank_pivot_data = pd.DataFrame(index=all_get_data_date, columns=['第' + str(i + 1) + '档收益率' for i in range(10)])
    #
    #     # 有的档位可能不存在显著的，所以导致该列pivot出来之后不存在
    #     temp_pivot_data = ranking_alpha_data[ranking_alpha_data[primary_keys_name_list].isin(primary_key).all(1)].pivot_table(
    #             index='get_data_date', columns='type_name', values='rank')
    #     temp_rank_pivot_data.loc[temp_pivot_data.index, temp_pivot_data.columns] = temp_pivot_data
    #     temp_rank_melt_data = temp_rank_pivot_data.reset_index().rename(columns={'GET_DATA_DATE': 'get_data_date'}).melt(
    #         id_vars=['get_data_date'], var_name=['type_name'], value_name='rank')
    #
    #     for rolling_num in week_data_rolling_pct_list:
    #         rolling_rank_number = temp_rank_melt_data.groupby(by=['type_name']).apply(
    #             lambda df:
    #             df['rank'].rolling(rolling_num, min_periods=1).apply(lambda rolling_data: Counter(pd.Series(rolling_data).dropna()).most_common(1)[0][0])
    #         ).reset_index()
    #         # Counter(rolling_data).most_common(2) -> [(1, 5), (2, 3)] 第一个tuple表示出现次数最多的数字1出现过的次数是5
    #         # Counter(rolling_data).most_common(1)[0][0] 即取出现次数第一多的数字是什么数字
    #         # Counter(rolling_data).most_common(1)[0][1] 即取出现次数第一多的数字出现过几次
    #         rolling_rank_number_pct = temp_rank_melt_data.groupby(by=['type_name']).apply(
    #             lambda df:
    #             df['rank'].rolling(rolling_num, min_periods=1).apply(
    #                 lambda rolling_data: Counter(pd.Series(rolling_data).dropna()).most_common(1)[0][1] / rolling_num)
    #         ).reset_index()
    #
    #         temp_rank_melt_data.loc[rolling_rank_number['level_1'], 'last_' + str(rolling_num) + '_week_most_freq_rank'] = rolling_rank_number['rank']
    #         temp_rank_melt_data.loc[rolling_rank_number_pct['level_1'], 'last_' + str(rolling_num) + '_week_most_freq_rank_pct'] = \
    #             rolling_rank_number_pct['rank']
    #
    #     # 计算一个全时段多最多rank，每个时点往前全时段
    #     temp_rank_melt_data_cumsum = temp_rank_melt_data.groupby(by=['type_name']).apply(
    #         lambda df: pd.get_dummies(df['rank']).cumsum().apply(
    #             lambda rank_cumsum: [np.argmax(rank_cumsum), rank_cumsum.max()]
    #             if rank_cumsum.max() != 0
    #             else [np.nan, np.nan], axis=1
    #         )
    #     ).reset_index()
    #
    #     # 若全都没有显著的则会出现NaN的情况
    #     temp_rank_melt_data_cumsum[0] = temp_rank_melt_data_cumsum[0].apply(lambda rank_list:[np.nan, np.nan] if rank_list != rank_list else rank_list)
    #
    #     # 先算好一个截止到当前的日期长度，作为后面的分母
    #     temp_rank_melt_data_cumsum[1] = 1
    #     temp_rank_melt_data_date_length = temp_rank_melt_data_cumsum.groupby(by=['type_name']).apply(lambda df: df.cumsum()[1]).reset_index()
    #
    #     temp_rank_melt_data.loc[temp_rank_melt_data_cumsum['level_1'], 'till_now_most_freq_rank'] = \
    #         temp_rank_melt_data_cumsum[0].apply(lambda l: l[0])
    #     # 注意index对的都是temp_rank_melt_data_cumsum的
    #     temp_rank_melt_data.loc[temp_rank_melt_data_cumsum['level_1'], 'till_now_most_freq_rank_pct'] = \
    #         temp_rank_melt_data_cumsum[0].apply(lambda l: l[1]).div(temp_rank_melt_data_date_length.loc[temp_rank_melt_data_cumsum['level_1']][1])
    #
    #     factor_return_alpha_ranking_consistency[primary_key] = temp_rank_melt_data.copy()
    #     print('2.1 基于alpha排序的：' + '，'.join([str(pk) for pk in primary_key]))
    #
    # factor_return_alpha_ranking_consistency_df = transform_dict_to_df(factor_return_alpha_ranking_consistency, primary_keys_name_list)
    # factor_return_alpha_ranking_consistency_df = optimize_data_ram(factor_return_alpha_ranking_consistency_df)
    # pickle_dump_data(factor_return_alpha_ranking_consistency_df, output_file_url, data_name='factor_return_alpha_ranking_consistency_df')

    # 2.2 基于r_squared排序的

    ranking_sql = '''
        SELECT ''' + ','.join(ranking_columns) + ''',
            rank () OVER (PARTITION BY REGRESSION_MODEL, ROLLING_WINDOW, GET_DATA_DATE, FACTOR_NUMBER ORDER BY ADJ_RSQUARED DESC) rank
        FROM LYZS_TINYSOFT.FACTOR_RETURN_REGRESSION
        WHERE ALPHA_P_VALUE <= 0.1
    '''

    ranking_rsquared_data = read_data_from_oracle_db(ranking_sql)
    ranking_rsquared_data.columns = ranking_columns + ['rank']
    ranking_rsquared_data = ranking_rsquared_data.sort_values(by=['regression_model', 'rolling_window', 'factor_number', 'type_name', 'get_data_date'])

    primary_keys_name_list = ['regression_model', 'rolling_window', 'factor_number']
    significant_pk_all_list = [ranking_rsquared_data[column_name].unique().tolist() for column_name in primary_keys_name_list]
    primary_keys_list = list(itertools.product(*significant_pk_all_list))

    factor_number_sql = '''
        select factor_number
        from lyzs_tinysoft.factor_library
        order by factor_number
    '''
    factor_number_list = read_data_from_oracle_db(factor_number_sql)['FACTOR_NUMBER']
    week_data_rolling_pct_list = [12, 24, 52]

    factor_return_rsquared_ranking_consistency = {}

    for primary_key in primary_keys_list:

        # primary_keys：('OLS', 32, 'factor1')
        temp_rank_pivot_data = pd.DataFrame(index=all_get_data_date, columns=['第' + str(i + 1) + '档收益率' for i in range(10)])
        # 有的档位可能不存在显著的，所以导致该列pivot出来之后不存在
        temp_pivot_data = ranking_rsquared_data[ranking_rsquared_data[primary_keys_name_list].isin(primary_key).all(1)].pivot_table(
                index='get_data_date', columns='type_name', values='rank')
        temp_rank_pivot_data.loc[temp_pivot_data.index, temp_pivot_data.columns] = temp_pivot_data
        temp_rank_melt_data = temp_rank_pivot_data.reset_index().rename(columns={'GET_DATA_DATE': 'get_data_date'}).melt(
            id_vars=['get_data_date'], var_name=['type_name'], value_name='rank')

        for rolling_num in week_data_rolling_pct_list:
            rolling_rank_number = temp_rank_melt_data.groupby(by=['type_name']).apply(
                lambda df:
                df['rank'].rolling(rolling_num, min_periods=1).apply(lambda rolling_data: Counter(pd.Series(rolling_data).dropna()).most_common(1)[0][0])
            ).reset_index()
            # Counter(rolling_data).most_common(2) -> [(1, 5), (2, 3)] 第一个tuple表示出现次数最多的数字1出现过的次数是5
            # Counter(rolling_data).most_common(1)[0][0] 即取出现次数第一多的数字是什么数字
            # Counter(rolling_data).most_common(1)[0][1] 即取出现次数第一多的数字出现过几次
            rolling_rank_number_pct = temp_rank_melt_data.groupby(by=['type_name']).apply(
                lambda df:
                df['rank'].rolling(rolling_num, min_periods=1).apply(
                    lambda rolling_data: Counter(pd.Series(rolling_data).dropna()).most_common(1)[0][1] / rolling_num)
            ).reset_index()

            temp_rank_melt_data.loc[rolling_rank_number['level_1'], 'last_' + str(rolling_num) + '_week_most_freq_rank'] = rolling_rank_number['rank']
            temp_rank_melt_data.loc[rolling_rank_number_pct['level_1'], 'last_' + str(rolling_num) + '_week_most_freq_rank_pct'] = \
                rolling_rank_number_pct['rank']

        # 计算一个全时段多最多rank，每个时点往前全时段

        temp_rank_melt_data_cumsum = temp_rank_melt_data.groupby(by=['type_name']).apply(
            lambda df: pd.get_dummies(df['rank']).cumsum().apply(
                lambda rank_cumsum: [np.argmax(rank_cumsum), rank_cumsum.max()] if rank_cumsum.max() != 0 else [np.nan, np.nan], axis=1
            )
        ).reset_index()

        # 若全都没有显著的则会出现NaN的情况
        temp_rank_melt_data_cumsum[0] = temp_rank_melt_data_cumsum[0].apply(
            lambda rank_list: [np.nan, np.nan] if rank_list != rank_list else rank_list)

        # 先算好一个截止到当前的日期长度，作为后面的分母
        temp_rank_melt_data_cumsum[1] = 1
        temp_rank_melt_data_date_length = temp_rank_melt_data_cumsum.groupby(by=['type_name']).apply(lambda df: df.cumsum()[1]).reset_index()

        temp_rank_melt_data.loc[temp_rank_melt_data_cumsum['level_1'], 'till_now_most_freq_rank'] = temp_rank_melt_data_cumsum[0].apply(lambda l: l[0])
        # 注意index对的都是temp_rank_melt_data_cumsum的
        temp_rank_melt_data.loc[temp_rank_melt_data_cumsum['level_1'], 'till_now_most_freq_rank_pct'] = \
            temp_rank_melt_data_cumsum[0].apply(lambda l: l[1]).div(temp_rank_melt_data_date_length.loc[temp_rank_melt_data_cumsum['level_1']][1])

        factor_return_rsquared_ranking_consistency[primary_key] = temp_rank_melt_data.copy()
        print('2.2 基于r_squared排序的：' + '，'.join([str(pk) for pk in primary_key]))

    factor_return_alpha_ranking_consistency_df = transform_dict_to_df(factor_return_rsquared_ranking_consistency, primary_keys_name_list)
    factor_return_alpha_ranking_consistency_df = optimize_data_ram(factor_return_alpha_ranking_consistency_df)
    pickle_dump_data(factor_return_alpha_ranking_consistency_df, output_file_url, data_name='factor_return_alpha_ranking_consistency_df')


