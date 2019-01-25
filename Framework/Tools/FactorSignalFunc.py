import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import math
from decimal import Decimal

# -------------------------细节函数------------------------------


def get_quantile_data_by_factor_quantile(single_date_data, factor_number, floor_quantile, cap_quantile):

    stock_floor_quantile = single_date_data[factor_number].quantile(floor_quantile)
    stock_cap_quantile = single_date_data[factor_number].quantile(cap_quantile)
    if floor_quantile == 0:  # 因为最小值不包括，所以要把最小值变小一点把它包进来
        stock_floor_quantile = stock_floor_quantile - 0.1

    quantile_data = single_date_data[(single_date_data[factor_number] > stock_floor_quantile) &
                                     (single_date_data[factor_number] <= stock_cap_quantile)].copy()
    return quantile_data


def get_factor_stratification_info(data, sample_scope=None, factor_number_list=None, stratification_num=10, quantile_dict=None):

    cap_quantile_list = [(quantile + 1) / stratification_num for quantile in range(0, stratification_num)]
    floor_quantile_list = [quantile / stratification_num for quantile in range(0, stratification_num)]

    factor_stratification_data = {}

    factor_data_columns = ['get_data_date', 'stockid']

    for sample_name in sample_scope:

        print('\t开始根据因子进行分档：' + sample_name)

        for factor_number in factor_number_list:

            for i in range(stratification_num):
                # 此段用来生成每个分位的组合。
                factor_stratification_data[(sample_name, factor_number, quantile_dict[i])] = \
                    data[(sample_name, factor_number)][factor_data_columns + [factor_number]].groupby(by=['数据提取日']).apply(
                        lambda df: get_quantile_data_by_factor_quantile(df, factor_number, floor_quantile_list[i], cap_quantile_list[i]))

                factor_stratification_data[(sample_name, factor_number, quantile_dict[i])].index = \
                    range(factor_stratification_data[(sample_name, factor_number, quantile_dict[i])].shape[0])

                min_count = factor_stratification_data[(sample_name, factor_number, quantile_dict[i])].groupby(
                    by=['数据提取日']).count()['stockid'].min()
                max_count = factor_stratification_data[(sample_name, factor_number, quantile_dict[i])].groupby(
                    by=['数据提取日']).count()['stockid'].max()

                print('\t完成根据因子进行分档：' + sample_name + '-' + factor_number + '-第' + quantile_dict[i] + '档' +
                      '(' + format(min_count, '.0f') + '/' + format(max_count, '.0f') + ')')

    return factor_stratification_data


def get_factor_stratification_portfolio_return(factor_stratification_data, stock_return_df, sample_scope=None, factor_number_list=None,
                                               startification_num=10, quantile_dict=None, yield_type_list=None, get_factor_data_date_list=None):

    factor_stratification_return = {}
    # 计算每个样本池
    for sample_name in sample_scope:

        print('\t开始计算因子分档后每档收益率:' + sample_name)

        for factor_number in factor_number_list:

            # 计算每档
            for i in range(startification_num):
                factor_stratification_return[(sample_name, factor_number, quantile_dict[i])] = \
                    pd.DataFrame(index=get_factor_data_date_list, columns=['因子均值'] + yield_type_list)

                factor_stratification_return[(sample_name, factor_number, quantile_dict[i])]['因子均值'] = \
                    factor_stratification_data[(sample_name, factor_number, quantile_dict[i])].groupby(by=['get_data_date']).apply(
                        lambda df: df[factor_number].mean()).astype('float32')

                for yield_type in yield_type_list:
                    tempo_yield_data = factor_stratification_data[(sample_name, factor_number, quantile_dict[i])].merge(
                        stock_return_df[['get_data_date', 'stockid'] + [yield_type]], on=['get_data_date', 'stockid'], how='left').copy()

                    factor_stratification_return[(sample_name, factor_number, quantile_dict[i])][yield_type] = \
                        tempo_yield_data.groupby(by=['get_data_date']).apply(lambda df: df[yield_type].mean()).astype('float32')

                    print('\t完成计算因子分档后每档收益率：' + sample_name + '-' + factor_number + '-第' + quantile_dict[i] + '档-' + yield_type)

    return factor_stratification_return


def set_ts_data_constant_test_result(time_series_data):
    result = pd.Series(index=['样本数', 't值', 'p值', '显著性', '最大滞后项'])
    adftest = adfuller(time_series_data.astype(float), regression="ct")
    result.loc['样本数'] = adftest[3]
    result.loc['t值'] = round(adftest[0], 2)
    result.loc['p值'] = Decimal(format(adftest[1], '.3f'))
    if adftest[0] <= adftest[4]['1%']:
        result.loc['显著性'] = '***'
    elif adftest[0] <= adftest[4]['5%']:
        result.loc['显著性'] = '**'
    elif adftest[0] <= adftest[4]['10%']:
        result.loc['显著性'] = '*'
    else:
        result.loc['显著性'] = '不显著'

    result.loc['最大滞后项'] = adftest[2]
    return result


def set_linear_regression_result(regression_result, result_content_list, method='OLS'):
    if (method == 'OLS') | (method == 'WLS'):
        result = pd.Series(index=result_content_list)
        result.loc['Adj. R-squared'] = round(regression_result.rsquared_adj, 4)
    elif method == 'RLM':
        result = pd.Series(index=result_content_list)

    result.loc[['Alpha', 'Beta']] = [round(value, 3) for value in list(regression_result.params)]
    result.loc[['Alpha t值', 'Beta t值']] = [round(value, 3) for value in list(regression_result.tvalues)]
    result.loc[['Alpha p值', 'Beta p值']] = [round(value, 4) for value in list(regression_result.pvalues)]
    result.loc[['Alpha标准误', 'Beta标准误']] = [round(value, 3) for value in list(regression_result.bse)]
    if regression_result.pvalues[0] <= 0.01:
        result.loc['Alpha显著性'] = '***'
    elif regression_result.pvalues[0] <= 0.05:
        result.loc['Alpha显著性'] = '**'
    elif regression_result.pvalues[0] <= 0.1:
        result.loc['Alpha显著性'] = '*'
    else:
        result.loc['Alpha显著性'] = ''

    if regression_result.pvalues[1] <= 0.01:
        result.loc['Beta显著性'] = '***'
    elif regression_result.pvalues[1] <= 0.05:
        result.loc['Beta显著性'] = '**'
    elif regression_result.pvalues[1] <= 0.1:
        result.loc['Beta显著性'] = '*'
    else:
        result.loc['Beta显著性'] = ''
    return result


def get_stratify_ts_data_regression_result(factor_stratification_return, index_return_df, sample_scope=None, factor_number_list=None,
                                           get_factor_data_date_list=None, regression_model_list=None,
                                           quantile_dict=None, rolling_window_list=None, stratification_num=10):

    # ts_constant_test_result_dict = {}
    factor_test_result = {}
    result_content_list = ['Alpha显著性', 'Alpha', 'Alpha t值', 'Alpha标准误', 'Alpha p值', 'Beta显著性', 'Beta', 'Beta t值', 'Beta标准误', 'Beta p值']
    result_value_content_list = ['Alpha', 'Alpha t值', 'Alpha标准误', 'Alpha p值', 'Beta', 'Beta t值', 'Beta标准误', 'Beta p值']

    for sample_name in sample_scope:

        print('\t开始单因子时间序列数据回归检验：' + sample_name)

        for factor_number in factor_number_list:

            factor_return_df = pd.DataFrame(index=get_factor_data_date_list, columns=['因子收益率', 'Alpha', 'Beta'])
            factor_return_df['Beta'] = index_return_df[sample_name + '收益率']
            factor_return_df['Alpha'] = 1

            # # 1. 时间序列平稳性检测
            # for i in range(stratification_num):
            #     factor_return_df['因子收益率'] = factor_stratification_return[(sample_name, factor_number, quantile_dict[i])]['持仓期收益率']
            #
            #     # (1) 每档都做回归，因为有的基准是从2007年才开始，因此要dropna
            #     ts_regression_df = factor_return_df.dropna()
            #     ts_regression_df.index = pd.Series(ts_regression_df.index).apply(lambda d: pd.to_datetime(d))
            #
            #     # (2) 对因子收益率和指数收益率序列进行平稳性检验
            #     ts_constant_test_result_dict[(sample_name, factor_number, quantile_dict[i])] = \
            #         pd.DataFrame(index=['因子收益率', '指数收益率'], columns=['样本数', 't值', 'p值', '显著性', '最大滞后项'])
            #     ts_constant_test_result_dict[(sample_name, factor_number, quantile_dict[i])].loc['因子收益率'] = \
            #         set_ts_data_constant_test_result(ts_regression_df['因子收益率'])
            #     ts_constant_test_result_dict[(sample_name, factor_number, quantile_dict[i])].loc['指数收益率'] = \
            #         set_ts_data_constant_test_result(ts_regression_df['Beta'])

            # 2. 滚动窗口回归
            for regression_model in regression_model_list:
                # (1) 选择不同的回归模型

                for rolling_window in rolling_window_list:
                    # (2) 选择不同的滚动窗口长度
                    rolling_window_end_date_list = get_factor_data_date_list[rolling_window - 1:]

                    for i in range(stratification_num):

                        # (3) 每档都分别进行滚动窗口回归
                        if (regression_model == 'WLS') | (regression_model == 'OLS'):
                            factor_test_result[(sample_name, factor_number, quantile_dict[i], regression_model, rolling_window)] = \
                                pd.DataFrame(index=rolling_window_end_date_list, columns=result_content_list + ['Adj. R-squared'])
                        elif regression_model == 'RLM':
                            factor_test_result[(sample_name, factor_number, quantile_dict[i], regression_model, rolling_window)] = \
                                pd.DataFrame(index=rolling_window_end_date_list, columns=result_content_list)

                        for date_i, date in enumerate(rolling_window_end_date_list, rolling_window):
                            # 选择每个滚动窗口的最后一个日期
                            regression_period = get_factor_data_date_list[date_i - rolling_window:date_i]
                            regression_data = pd.DataFrame(index=regression_period, columns=['因子收益率', 'Alpha', 'Beta'])
                            regression_data['因子收益率'] = \
                                factor_stratification_return[(sample_name, factor_number, quantile_dict[i])]['持仓期收益率'].loc[regression_period]
                            regression_data['Alpha'] = 1
                            regression_data['Beta'] = index_return_df[sample_name + '收益率'].loc[regression_period]

                            if regression_model == 'RLM':
                                regression_result = sm.RLM(regression_data.loc[regression_period, '因子收益率'].astype(float),
                                                           regression_data.loc[regression_period, ['Alpha', 'Beta']].astype(float)).fit()
                            elif regression_model == 'OLS':
                                regression_result = sm.OLS(regression_data.loc[regression_period, '因子收益率'].astype(float),
                                                           regression_data.loc[regression_period, ['Alpha', 'Beta']].astype(
                                                               float)).fit().get_robustcov_results()
                            elif regression_model == 'WLS':
                                weight_dict = {'cos': [1 / (math.cos(x) / sum([math.cos(x) for x in np.linspace(0, math.pi / 2, rolling_window)]))
                                                       for x in np.linspace(0, math.pi / 2, rolling_window)]}

                                regression_result = sm.WLS(regression_data.loc[regression_period, '因子收益率'].astype(float),
                                                           regression_data.loc[regression_period, ['Alpha', 'Beta']].astype(float),
                                                           weights=weight_dict['cos']).fit().get_robustcov_results()

                            factor_test_result[(sample_name, factor_number, quantile_dict[i], regression_model, rolling_window)].loc[date] = \
                                set_linear_regression_result(regression_result, result_content_list, method=regression_model)

                        # 把作为index的日期提出来成为一列，因为这一个dataframe只包括一个因子序号的一个档位回归结果，后期要整合所有档位到一起
                        factor_test_result[(sample_name, factor_number, quantile_dict[i], regression_model, rolling_window)] = \
                            factor_test_result[(sample_name, regression_model, rolling_window, factor_number, quantile_dict[i])].reset_index().rename(
                                columns={'index': '数据提取日'})
                        factor_test_result[(sample_name, factor_number, quantile_dict[i], regression_model, rolling_window)][
                            result_value_content_list] = \
                            factor_test_result[(sample_name, factor_number, quantile_dict[i], regression_model, rolling_window)][
                                result_value_content_list].apply(pd.to_numeric, downcast='float')
                        if (regression_model == 'OLS') | (regression_model == 'WLS'):
                            factor_test_result[(sample_name, factor_number, quantile_dict[i], regression_model, rolling_window)]['Adj. R-squared'] = \
                                factor_test_result[(sample_name, factor_number, quantile_dict[i], regression_model, rolling_window)][
                                    'Adj. R-squared'].apply(pd.to_numeric, downcast='float')

                        print('\t完成时间序列数据回归检验：' + sample_name + '-' + factor_number + '-第' + quantile_dict[i] + '档-回归模型'
                              + regression_model + '-滚动窗口期' + str(rolling_window))

    return factor_test_result




