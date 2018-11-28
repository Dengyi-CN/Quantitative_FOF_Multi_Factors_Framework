import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', 20)
pd.set_option('display.width', 320)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
import numpy as np
import math
import pickle
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from decimal import Decimal
import matplotlib.style as matstyle
matstyle.use('ggplot')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set_context(rc={'figure.figsize': (12, 7)})

# ----------------------------------------------------------函数（开始）-------------------------------------------------------------------------------


def drop_outlier_and_standardization(factor_data, factor_name, drop_outlier_method='winsorize', standardization=False,
                                     fillna_after_standardization=True):
    # (1) 先对被解释变量，即收益率端的异常值进行处理
    independent_var_outlier_index = factor_data[factor_data['持仓期停牌天数占比'] >= 0.1].index.tolist()
    factor_data.loc[independent_var_outlier_index, '持仓期停牌天数占比'] = np.nan

    if drop_outlier_method == 'median':
        Dm = factor_data[factor_name].median()
        Dm1 = abs(factor_data[factor_name] - Dm).median()
        cap_index = factor_data[factor_name] > (Dm + 5 * Dm1)
        floor_index = factor_data[factor_name] < (Dm - 5 * Dm1)
        factor_data.loc[cap_index, factor_name] = np.nan
        factor_data.loc[floor_index, factor_name] = np.nan
    elif drop_outlier_method == 'winsorize':
        outlier_index_list = \
            pd.Series(factor_data[factor_name].sort_values().iloc[0:factor_data[factor_name].shape[0] // 100].index).sort_values().tolist()
        outlier_index_list += \
            pd.Series(factor_data[factor_name].sort_values().iloc[-factor_data[factor_name].shape[0] // 100:].index).sort_values().tolist()
        factor_data.loc[outlier_index_list, factor_name] = np.nan

    if standardization:
        factor_data[factor_name] = \
            (factor_data[factor_name] - factor_data[factor_name].mean()) / factor_data[factor_name].std()
        if fillna_after_standardization :
            factor_data = factor_data.fillna(0)

    return factor_data.dropna()


def get_stock_by_factor_quantile(each_report_deadline_date_data, factor_name, floor_quantile, cap_quantile):

    stock_floor_quantile = each_report_deadline_date_data[factor_name].quantile(floor_quantile)
    stock_cap_quantile = each_report_deadline_date_data[factor_name].quantile(cap_quantile)
    if floor_quantile == 0:
        stock_floor_quantile = stock_floor_quantile - 0.1

    stock_data = each_report_deadline_date_data[(each_report_deadline_date_data[factor_name] > stock_floor_quantile) &
                                                (each_report_deadline_date_data[factor_name] <= stock_cap_quantile)]

    return stock_data


def get_portfolio_evaluation_indicator(portfolio_data, portfolio_name):
    list_1 = ['算数平均值', '几何平均值', '中位数', '标准差', '向上偏差', '向下偏差',  '收益为正的时期占比', '收益为负的时期占比']
    list_2 = ['年化收益率', '年化波动率', '夏普比率(2%)', '最大回撤比例', '最大回撤开始时间', '最大回撤结束时间']

    evaluation_indicator = pd.DataFrame(index=['持仓期收益率 - ' + s for s in list_1] + ['组合表现 - ' + s for s in list_2], columns=[portfolio_name])

    port_open_period_return_series = portfolio_data.groupby(by=['财务报告截止日']).mean()['持仓期收益率'].div(100).copy()
    evaluation_indicator.loc['持仓期收益率 - 算数平均值', portfolio_name] = format(port_open_period_return_series.mean(), '.2%')
    evaluation_indicator.loc['持仓期收益率 - 几何平均值', portfolio_name] = \
        format(np.power(port_open_period_return_series.add(1).cumprod().iloc[-1], 1 / port_open_period_return_series.shape[0]) - 1, '.2%')
    evaluation_indicator.loc['持仓期收益率 - 中位数', portfolio_name] = format(port_open_period_return_series.median(), '.2%')

    evaluation_indicator.loc['持仓期收益率 - 标准差', portfolio_name] = format(port_open_period_return_series.std(), '.2%')
    up_bias = port_open_period_return_series[port_open_period_return_series >= port_open_period_return_series.mean()].copy()
    down_bias = port_open_period_return_series[port_open_period_return_series < port_open_period_return_series.mean()].copy()
    evaluation_indicator.loc['持仓期收益率 - 向上偏差', portfolio_name] = \
        format(np.sqrt((up_bias - port_open_period_return_series.mean()).apply(lambda d: d * d).sum() / up_bias.shape[0]), '.2%')
    evaluation_indicator.loc['持仓期收益率 - 向下偏差', portfolio_name] = \
        format(np.sqrt((down_bias - port_open_period_return_series.mean()).apply(lambda d: d * d).sum() / down_bias.shape[0]), '.2%')

    positive = port_open_period_return_series[port_open_period_return_series >= 0]
    negative = port_open_period_return_series[port_open_period_return_series < 0]
    evaluation_indicator.loc['持仓期收益率 - 收益为正的时期占比', portfolio_name] = format(positive.shape[0] / port_open_period_return_series.shape[0], '.2%')
    evaluation_indicator.loc['持仓期收益率 - 收益为负的时期占比', portfolio_name] = format(negative.shape[0] / port_open_period_return_series.shape[0], '.2%')

    holding_period_portfolio_return_series = portfolio_data.groupby(by=['财务报告截止日']).mean()['持仓期收益率'].div(100).copy()
    holding_period_trade_days = portfolio_data.groupby(by=['财务报告截止日']).mean()['持仓天数'].sum()
    holding_period_portfolio_return = holding_period_portfolio_return_series.sum()
    evaluation_indicator.loc['组合表现 - 年化收益率', portfolio_name] = format(holding_period_portfolio_return / holding_period_trade_days * 250, '.2%')
    evaluation_indicator.loc['组合表现 - 年化波动率', portfolio_name] = \
        format(holding_period_portfolio_return_series.std() / np.sqrt(holding_period_trade_days) * np.sqrt(250), '.2%')
    sharp_ratio = (holding_period_portfolio_return / holding_period_trade_days * 250 - 0.02) / \
                  (holding_period_portfolio_return_series.std() / np.sqrt(holding_period_trade_days) * np.sqrt(250))
    evaluation_indicator.loc['组合表现 - 夏普比率(2%)', portfolio_name] = format(sharp_ratio, '.2')

    portfolio_net_value = holding_period_portfolio_return_series.add(1).cumprod()
    max_drawdown_end_date = np.argmax(np.maximum.accumulate(portfolio_net_value) - portfolio_net_value)
    max_drawdown_begin_date = np.argmax(portfolio_net_value[:max_drawdown_end_date])

    evaluation_indicator.loc['组合表现 - 最大回撤比例', portfolio_name] = \
        format((portfolio_net_value.loc[max_drawdown_end_date] - portfolio_net_value.loc[max_drawdown_begin_date]) /
               portfolio_net_value.loc[max_drawdown_begin_date], '.2%')
    evaluation_indicator.loc['组合表现 - 最大回撤开始时间', portfolio_name] = max_drawdown_begin_date
    evaluation_indicator.loc['组合表现 - 最大回撤结束时间', portfolio_name] = max_drawdown_end_date

    return evaluation_indicator


def constant_test(time_series_data):
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
        result.loc['Adj. R-squared'] = format(regression_result.rsquared_adj, '.2%')
    elif method == 'RLM':
        result = pd.Series(index=result_content_list)

    result.loc[['Alpha', 'Beta']] = [format(value, '.3f') for value in list(regression_result.params)]
    result.loc[['Alpha t值', 'Beta t值']] = [format(value, '.3f') for value in list(regression_result.tvalues)]
    result.loc[['Alpha p值', 'Beta p值']] = [format(value, '.3f') for value in list(regression_result.pvalues)]
    result.loc[['Alpha标准误', 'Beta标准误']] = [format(value, '.5f') for value in list(regression_result.bse)]
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


def finding_most_significant_factor(result_dict, rolling_window, factor_type_dict, factor_name_dict, quantile_dict, get_factor_data_date_list,
                                    factor_list, method='RLM'):

    significant_factor_df = pd.DataFrame(index=get_factor_data_date_list[rolling_window - 1:],
                                         columns=['因子序号', '因子小类', '因子名称', '档数', 'Alpha显著性', 'Alpha', 'Alpha p值'])

    for str_date in get_factor_data_date_list[rolling_window - 1:]:
        date = pd.to_datetime(str_date)

        significant_factor_df.loc[str_date, 'Alpha'] = 0
        significant_factor_df.loc[str_date, 'Alpha p值'] = 0.1
        for factor_name in factor_list:

            # 先找出每个因子Alpha为正的最显著的层数
            for i in range(10):
                tempo_data = result_dict[factor_name]['申万A股'][quantile_dict[i]][rolling_window].copy()
                if (float(tempo_data.loc[date, 'Alpha p值']) < significant_factor_df.loc[str_date, 'Alpha p值']) & \
                        (float(tempo_data.loc[date, 'Alpha']) > significant_factor_df.loc[str_date, 'Alpha']):
                    significant_factor_df.loc[str_date, '因子序号'] = factor_name
                    significant_factor_df.loc[str_date, '因子小类'] = factor_type_dict[factor_name]
                    significant_factor_df.loc[str_date, '因子名称'] = factor_name_dict[factor_name]
                    significant_factor_df.loc[str_date, '档数'] = '第' + quantile_dict[i] + '组'
                    significant_factor_df.loc[str_date, 'Alpha显著性'] = tempo_data.loc[date, 'Alpha显著性']
                    significant_factor_df.loc[str_date, 'Alpha'] = float(tempo_data.loc[date, 'Alpha'])
                    significant_factor_df.loc[str_date, 'Alpha p值'] = float(tempo_data.loc[date, 'Alpha p值'])
                    if (method == 'OLS') | (method == 'WLS'):
                        significant_factor_df.loc[str_date, 'Adj. R-squared'] = tempo_data.loc[date, 'Adj. R-squared']
        print('完成显著因子挑选：' + str_date)
    return significant_factor_df


def set_factor_info(regression_result, factor_num, factor_category, factor_type, factor_name, quantile_name):
    # significant_factor = regression_result[(regression_result['Alpha p值'].astype(float) < 0.1) &
    #                                        (regression_result['Alpha'].astype(float) > 0)].copy()
    significant_factor = regression_result.copy()
    significant_factor['因子序号'] = factor_num
    significant_factor['因子大类'] = factor_category
    significant_factor['因子小类'] = factor_type
    significant_factor['因子名称'] = factor_name
    significant_factor['档位'] = quantile_name

    significant_factor.index = range(significant_factor.shape[0])
    return significant_factor


def data_cleaning(data, sample_list=None, factor_list=None, kicked_sector_list=None, go_public_days=250, data_processing_base_columns=None):
    # 1. 数据清洗
    # 1.1 剔除某些行业
    raw_data_step1 = data[data['sectorname'].apply(lambda s: s not in kicked_sector_list)].copy()

    # 1.2 剔除天软的脏数据：报告期为1900-12-31、以及一些莫名的日期

    raw_data_step2 = raw_data_step1[raw_data_step1['财务数据最新报告期'] != '1900-12-31'].copy()
    raw_data_step2 = raw_data_step2[raw_data_step2['数据提取日'].apply(lambda date: date in get_factor_data_date_list)].copy()

    # 1.3 剔除ST、PT股票，以及当期停牌的股票

    raw_data_step3 = raw_data_step2[(raw_data_step2['是否st'] == 0) & (raw_data_step2['是否pt'] == 0) & (raw_data_step2['是否停牌'] == 0)].copy()

    # 1.4 剔除每个截面数据量少于95%的因子数据

    raw_data_step4 = raw_data_step3.copy()

    # 1.5 剔除次新股
    raw_data_step5 = raw_data_step4.groupby(by=['数据提取日']).apply(lambda df: df[df['上市天数'] > go_public_days]).copy()

    # 1.6 取得干净数据
    clean_data = raw_data_step5[data_processing_base_columns + [sample_name + '成分股' for sample_name in sample_list] + factor_list].copy()
    clean_data.index = range(clean_data.shape[0])

    return clean_data


def data_processing(clean_data, raw_data, sample_list=None, factor_list=None, factor_name_dict=None):

    factor_raw_data_describe = {}
    factor_clean_data_describe = {}
    outlier_stock_dict = {}
    outlier_data_dict = {}
    clean_data_after_outlier = {}
    clean_data_info_columns_list = clean_data.columns[:4].tolist()

    for sample_name in sample_list:

        print('开始数据处理：' + sample_name)

        tempo_clean_data = clean_data[clean_data[sample_name + '成分股'] == 1].copy()

        for factor_name in factor_list:

            print('----------------------------------')
            print('开始数据处理：' + sample_name + '-' + factor_name + '(' + factor_name_dict[factor_name] + ')')

            # (1) 对原始数据进行描述性统计
            factor_raw_data_describe[(sample_name, factor_name)] = \
                tempo_clean_data[['数据提取日'] + [factor_name]].groupby(by=['数据提取日']).apply(
                    lambda df: df[factor_name].describe([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]))

            factor_raw_data_describe[(sample_name, factor_name)]['skewness'] = \
                tempo_clean_data[['数据提取日'] + [factor_name]].groupby(by=['数据提取日']).apply(lambda df: df[factor_name].skew())

            factor_raw_data_describe[(sample_name, factor_name)]['kurtosis'] = \
                tempo_clean_data[['数据提取日'] + [factor_name]].groupby(by=['数据提取日']).apply(lambda df: df[factor_name].kurtosis())

            factor_raw_data_describe[(sample_name, factor_name)]['non-NAN-data_pct'] = \
                tempo_clean_data[['数据提取日'] + [factor_name]].groupby(by=['数据提取日']).apply(
                    lambda df: df[factor_name].dropna().shape[0] / df[factor_name].shape[0])

            print('数据处理前样本最大/最小数量：' +
                  format(factor_raw_data_describe[(sample_name, factor_name)]['count'].min(), '.0f') + '/' +
                  format(factor_raw_data_describe[(sample_name, factor_name)]['count'].max(), '.0f'))

            # (2) 对数据进行处理：去异常值、标准化及填0（可选）
            clean_data_after_outlier[(sample_name, factor_name)] = \
                tempo_clean_data[data_processing_base_columns + [factor_name]].groupby(by=['数据提取日']).apply(
                    lambda df: drop_outlier_and_standardization(df, factor_name, drop_outlier_method='winsorize', standardization=False))[
                    ['数据提取日', 'stockid'] + [factor_name]]

            clean_data_after_outlier[(sample_name, factor_name)].index = range(clean_data_after_outlier[(sample_name, factor_name)].shape[0])

            factor_clean_data_describe[(sample_name, factor_name)] = \
                clean_data_after_outlier[(sample_name, factor_name)][['数据提取日'] + [factor_name]].groupby(by=['数据提取日']).apply(
                    lambda df: df[factor_name].describe([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]))

            factor_clean_data_describe[(sample_name, factor_name)]['skewness'] = \
                clean_data_after_outlier[(sample_name, factor_name)][['数据提取日'] + [factor_name]].groupby(by=['数据提取日']).apply(
                    lambda df: df[factor_name].skew())
            factor_clean_data_describe[(sample_name, factor_name)]['kurtosis'] = \
                clean_data_after_outlier[(sample_name, factor_name)][['数据提取日'] + [factor_name]].groupby(by=['数据提取日']).apply(
                    lambda df: df[factor_name].kurtosis())

            print('数据处理后样本最大/最小数量：' +
                  format(factor_clean_data_describe[(sample_name, factor_name)]['count'].min(), '.0f') + '/' +
                  format(factor_clean_data_describe[(sample_name, factor_name)]['count'].max(), '.0f'))

            # (3) 保存异常值股票池及其数据

            # print('保存异常值数据。')
            # outlier_stock_dict[factor_name] = clean_data[clean_data_info_columns_list + [factor_name]].groupby(by=['数据提取日']).apply(
            #     lambda df: get_outlier_stock_list(df, factor_name, method='winsorize'))
            #
            # outlier_data_dict[factor_name] = pd.DataFrame()
            # for date in outlier_stock_dict[factor_name].index:
            #     outlier_tempo_date_data = clean_data[clean_data['数据提取日'] == date][clean_data_info_columns_list + [factor_name]].copy()
            #     outlier_tempo_date_data['异常值类型'] = np.nan
            #     outlier_type_and_index = outlier_tempo_date_data['stockid'].apply(
            #         lambda stock: '收益端异常值' if stock in outlier_stock_dict[factor_name].loc[date][0]
            #         else('因子端异常值' if stock in outlier_stock_dict[factor_name].loc[date][1] else np.nan)).dropna()
            #     outlier_tempo_date_data.loc[outlier_type_and_index.index, '异常值类型'] = outlier_type_and_index
            #     output_columns = ['异常值类型'] + clean_data_info_columns_list + [factor_name]
            #     outlier_data_dict[factor_name] = pd.concat([outlier_data_dict[factor_name],
            #                                                 outlier_tempo_date_data.loc[outlier_type_and_index.index][output_columns]])

            print('完成数据处理：' + factor_name + '(' + factor_name_dict[factor_name] + ')')

    return clean_data_after_outlier, factor_raw_data_describe, factor_clean_data_describe


def get_factor_stratification_data(data, sample_list=None, factor_list=None, stratification_num=10, quantile_dict=None):
    cap_quantile_list = [(quantile + 1) / stratification_num for quantile in range(0, stratification_num)]
    floor_quantile_list = [quantile / stratification_num for quantile in range(0, stratification_num)]

    factor_stratification_data = {}

    factor_data_columns = ['数据提取日', 'stockid']

    for sample_name in sample_list:

        print('开始因子构建：' + sample_name)
        print('----------------------------------')

        for factor_name in factor_list:

            for i in range(stratification_num):
                # 此段用来生成每个分位的组合。
                factor_stratification_data[(sample_name, factor_name, quantile_dict[i])] = \
                    data[(sample_name, factor_name)][factor_data_columns + [factor_name]].groupby(by=['数据提取日']).apply(
                        lambda df: get_stock_by_factor_quantile(df, factor_name, floor_quantile_list[i], cap_quantile_list[i]))

                factor_stratification_data[(sample_name, factor_name, quantile_dict[i])].index = \
                    range(factor_stratification_data[(sample_name, factor_name, quantile_dict[i])].shape[0])

                min_count = factor_stratification_data[(sample_name, factor_name, quantile_dict[i])].groupby(by=['数据提取日']).count()['stockid'].min()
                max_count = factor_stratification_data[(sample_name, factor_name, quantile_dict[i])].groupby(by=['数据提取日']).count()['stockid'].max()
                print('完成因子构建：' + sample_name + '-' + factor_name + '(' + factor_name_dict[factor_name] + ')'
                      + '-第' + quantile_dict[i] + '组' + '(' + format(min_count, '.0f') + '/' + format(max_count, '.0f') + ')')
            print('----------------------------------')

        print('-----------------------------------------------------------------------')

    return factor_stratification_data


def get_factor_stratification_return(factor_stratification_data, stock_return_df, sample_list=None, factor_list=None, startification_num=10,
                                     quantile_dict=None, yield_type_list=None):
    factor_stratification_return = {}
    # 计算每个样本池
    for sample_name in sample_list:

        print('开始计算分层收益率:' + sample_name)
        print('-----------------------------------------------------------------------')

        for factor_name in factor_list:

            # 计算每层
            for i in range(startification_num):
                factor_stratification_return[(sample_name, factor_name, quantile_dict[i])] = \
                    pd.DataFrame(index=get_factor_data_date_list, columns=['因子均值'] + yield_type_list)

                factor_stratification_return[(sample_name, factor_name, quantile_dict[i])]['因子均值'] = \
                    factor_stratification_data[(sample_name, factor_name, quantile_dict[i])].groupby(by=['数据提取日']).apply(
                        lambda df: df[factor_name].mean())

                for yield_type in yield_type_list:
                    tempo_yield_data = factor_stratification_data[(sample_name, factor_name, quantile_dict[i])].merge(
                        stock_return_df[['数据提取日', 'stockid'] + [yield_type]], on=['数据提取日', 'stockid'], how='left').copy()

                    factor_stratification_return[(sample_name, factor_name, quantile_dict[i])][yield_type] = \
                        tempo_yield_data.groupby(by=['数据提取日']).apply(lambda df: df[yield_type].mean())

                    print('完成分组收益率计算：' + sample_name + '-' +
                          factor_name + '(' + factor_name_dict[factor_name] + ')' + '-第' + quantile_dict[i] + '组-' + yield_type)

            print('----------------------------------')

    return factor_stratification_return


def get_factor_test_result(factor_stratification_return, index_return_df, sample_list=None, factor_list=None,
                           get_factor_data_date_list=None, regression_model_list=None,
                           quantile_dict=None, rolling_window_list=None, stratification_num=10):

    ts_constant_test_result_dict = {}
    factor_test_result = {}
    result_content_list = ['Alpha显著性', 'Alpha', 'Alpha t值', 'Alpha标准误', 'Alpha p值', 'Beta显著性', 'Beta', 'Beta t值', 'Beta标准误', 'Beta p值']

    for sample_name in sample_list:

        print('开始单因子回归检测：' + sample_name)
        print('-----------------------------------------------------------------------')

        for factor_name in factor_list:

            factor_return_df = pd.DataFrame(index=get_factor_data_date_list, columns=['因子收益率', 'Alpha', 'Beta'])
            factor_return_df['Beta'] = index_return_df[sample_name + '收益率']
            factor_return_df['Alpha'] = 1

            # 1. 时间序列平稳性检测
            for i in range(stratification_num):
                factor_return_df['因子收益率'] = factor_stratification_return[(sample_name, factor_name, quantile_dict[i])]['持仓期收益率']

                # (1) 每档都做回归，因为有的基准是从2007年才开始，因此要dropna
                ts_regression_df = factor_return_df.dropna()
                ts_regression_df.index = pd.Series(ts_regression_df.index).apply(lambda d: pd.to_datetime(d))

                # (2) 对因子收益率和指数收益率序列进行平稳性检验
                ts_constant_test_result_dict[(sample_name, factor_name, quantile_dict[i])] = \
                    pd.DataFrame(index=['因子收益率', '指数收益率'], columns=['样本数', 't值', 'p值', '显著性', '最大滞后项'])
                ts_constant_test_result_dict[(sample_name, factor_name, quantile_dict[i])].loc['因子收益率'] = constant_test(ts_regression_df['因子收益率'])
                ts_constant_test_result_dict[(sample_name, factor_name, quantile_dict[i])].loc['指数收益率'] = constant_test(ts_regression_df['Beta'])

            # 2. 滚动窗口回归
            for regression_model in regression_model_list:
                # (1) 选择不同的回归模型

                for rolling_window in rolling_window_list:
                    # (2) 选择不同的滚动窗口长度
                    rolling_window_end_date_list = get_factor_data_date_list[rolling_window - 1:]

                    for i in range(stratification_num):

                        # (3) 每档都分别进行滚动窗口回归
                        if (regression_model == 'WLS') | (regression_model == 'OLS'):
                            factor_test_result[(sample_name, regression_model, rolling_window, factor_name, quantile_dict[i])] = \
                                pd.DataFrame(index=rolling_window_end_date_list, columns=result_content_list + ['Adj. R-squared'])
                        elif regression_model == 'RLM':
                            factor_test_result[(sample_name, regression_model, rolling_window, factor_name, quantile_dict[i])] = \
                                pd.DataFrame(index=rolling_window_end_date_list, columns=result_content_list)

                        for date_i, date in enumerate(rolling_window_end_date_list, rolling_window):
                            # 选择每个滚动窗口的最后一个日期
                            regression_period = get_factor_data_date_list[date_i - rolling_window:date_i]
                            regression_data = pd.DataFrame(index=regression_period, columns=['因子收益率', 'Alpha', 'Beta'])
                            regression_data['因子收益率'] = \
                                factor_stratification_return[(sample_name, factor_name, quantile_dict[i])]['持仓期收益率'].loc[regression_period]
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

                            factor_test_result[(sample_name, regression_model, rolling_window, factor_name, quantile_dict[i])].loc[date] = \
                                set_linear_regression_result(regression_result, result_content_list, method=regression_model)

                        # 把作为index的日期提出来成为一列，因为这一个dataframe只包括一个因子序号的一个档位回归结果，后期要整合所有档位到一起
                        factor_test_result[(sample_name, regression_model, rolling_window, factor_name, quantile_dict[i])] = \
                            factor_test_result[(sample_name, regression_model, rolling_window, factor_name, quantile_dict[i])].reset_index().rename(
                                columns={'index': '数据提取日'})
                        print('完成单因子回归检验：' + sample_name + '-' + factor_name + '(' + factor_name_dict[factor_name] + ')' +
                              '-第' + quantile_dict[i] + '组-窗口期' + str(rolling_window))

            print('----------------------------------')

    print('完成全部单因子回归检验。')

    return factor_test_result, ts_constant_test_result_dict


def get_factor_stratification_hp_return(factor_stratification_return, market_mean_return, sample_list=None, factor_list=None,
                                        stratification_num=10, quantile_dict=None, factor_name_dict=None):

    factor_stratification_hp_return = {}

    for sample_name in sample_list:

        for factor_name in factor_list:

            factor_stratification_hp_return[(sample_name, factor_name)] = \
                pd.DataFrame(index=get_factor_data_date_list, columns=list(quantile_dict.values()) + ['相邻档超额收益差平方和', '相邻档超额收益差平方均值'])
            for i in range(stratification_num):
                factor_stratification_hp_return[(sample_name, factor_name)][quantile_dict[i]] = \
                    factor_stratification_return[(sample_name, factor_name, quantile_dict[i])]['持仓期收益率']

            factor_stratification_hp_return[(sample_name, factor_name)]['相邻档超额收益差平方和'] = \
                factor_stratification_hp_return[(sample_name, factor_name)][list(quantile_dict.values())].sub(
                    market_mean_return['持仓期收益率'], axis=0).rolling(2, axis=1).apply(lambda s: s[1] - s[0]).applymap(lambda v: v ** 2).sum(axis=1)
            factor_stratification_hp_return[(sample_name, factor_name)]['相邻档超额收益差平方均值'] = \
                factor_stratification_hp_return[(sample_name, factor_name)][list(quantile_dict.values())].sub(
                    market_mean_return['持仓期收益率'], axis=0).rolling(2, axis=1).apply(lambda s: s[1] - s[0]).applymap(lambda v: v ** 2).mean(axis=1)

            factor_stratification_hp_return[(sample_name, factor_name)] = \
                factor_stratification_hp_return[(sample_name, factor_name)].reset_index().rename(columns={'index': '数据提取日'})
            print('保存各因子序号分档收益率，计算各档收益发散性指标：' + sample_name + '-' + factor_name + '(' + factor_name_dict[factor_name] + ')')

        print('----------------------------------')

    return factor_stratification_hp_return


def transform_dict_to_dataframe(dict_data, keys_column_name_list):
    """
    用于将处理过程中保存的dict类型转为dataframe，因为在处理过程中用dict更加方便明了，但是在最终结果展示环节可能还需要dataframe的形式导出成excel
    :param dict_data:
    :param keys_column_name_list: 该参数是dict_data中要转为一列column的key的所属类别，例如dict_data的keys是('申万A股', 'WLS')，
    那么该参数就是['股票池', '回归模型']
    :return: 将dict_data中的keys全部转换为了columns的dataframe
    """

    dataframe_data = pd.DataFrame()
    for key_list, df_data in dict_data.items():
        for key_i, key_column_name in enumerate(keys_column_name_list):
            df_data[key_column_name] = key_list[key_i]
        dataframe_data = pd.concat([dataframe_data, df_data])
    dataframe_data.index = range(dataframe_data.shape[0])

    return dataframe_data


def get_purified_factor(prior_purified_data, purified_class=None, lower_class=None, sample_name=None, regression_model=None, rolling_window=None,
                        factor_stratification_return=None, index_return_df=None, get_factor_data_date_list=None):
    """
    通用的因子提纯函数，因子序列提纯为因子小类，因子小类提纯为因子大类都能使用，暂时只用了取解释度最高的因子，因此只能用WLS或者OLS的方法
    :param prior_purified_data:
    :param purified_class:
    :param lower_class:
    :param sample_name:
    :param regression_model:
    :param rolling_window:
    :param factor_stratification_return:
    :param index_return_df:
    :param get_factor_data_date_list:
    :return:
    """

    # 变量声明
    within_purified_class_factor_corr = {}  # 提纯后，该类别内所有纯净小类因子的相关系数

    # 变量定义
    purified_class_name_list = prior_purified_data[purified_class].unique().tolist()  # 得到所有类别名称
    factor_info_list = ['因子大类', '因子小类', '因子序号', '档位']
    required_factor_info_list = factor_info_list[factor_info_list.index(lower_class):]
    complement_factor_info_list = factor_info_list[:factor_info_list.index(lower_class)]
    factor_test_info_list = ['Alpha显著性', 'Alpha', 'Alpha t值', 'Alpha标准误', 'Alpha p值', 'Beta显著性', 'Beta', 'Beta t值', 'Beta标准误',
                             'Beta p值', 'Adj. R-squared']
    all_factor_info_list = ['数据提取日'] + factor_info_list + factor_test_info_list

    all_purified_data = pd.DataFrame(columns=all_factor_info_list)
    MES_purified_data = pd.DataFrame(columns=all_factor_info_list)

    if regression_model == 'WLS':
        weight_dict = {'cos': [1 / (math.cos(x) / sum([math.cos(x) for x in np.linspace(0, math.pi / 2, rolling_window)]))
                               for x in np.linspace(0, math.pi / 2, rolling_window)]}

    for class_name in purified_class_name_list:

        tempo_all_purified_data = pd.DataFrame(columns=all_factor_info_list)
        tempo_MES_purified_data = pd.DataFrame(columns=all_factor_info_list)

        tempo_data = prior_purified_data[prior_purified_data[purified_class] == class_name].copy()

        # 注意：此处直接忽略了没有显著因子的日期，只对存在显著因子的日期进行提纯
        tempo_regression_end_date_list = tempo_data['数据提取日'].unique().tolist()

        for regression_end_date in tempo_regression_end_date_list:

            # (1) 根据回归最后的时间点找到该回归显著的因子，如果只有1个那么就不用提纯了
            tempo_date_data = tempo_data[tempo_data['数据提取日'] == regression_end_date].copy()
            tempo_date_data.index = range(tempo_date_data.shape[0])

            lower_class_name_list = tempo_date_data[lower_class].tolist()  # 由于最底层只取最显著的那一档，所以每个因子序号/因子名称唯一
            if len(lower_class_name_list) == 1:
                continue

            # (2) 进行Fama-MacBeth两次回归进行提纯

            # (2.1) 根据回归最后的时间点找到滚动窗口时间段
            regression_end_date_index = get_factor_data_date_list.index(regression_end_date)  # index是31，表示第32个周
            # list后不包，所以要+1
            regression_period = get_factor_data_date_list[regression_end_date_index - rolling_window + 1:regression_end_date_index + 1]

            # (2.2) 构建第一次回归所需数据
            first_regression_data = pd.DataFrame(index=regression_period, columns=lower_class_name_list)
            for i in tempo_date_data.index:
                factor_num = tempo_date_data.loc[i, '因子序号']
                factor_stratification_num = tempo_date_data.loc[i, '档位']
                first_regression_data[tempo_date_data.loc[i, lower_class]] = \
                    factor_stratification_return[(sample_name, factor_num, factor_stratification_num)]['持仓期收益率'].loc[regression_period]

            # (2.3) 第一阶段回归：回归取残差项
            first_regression_residual = pd.DataFrame(index=regression_period, columns=lower_class_name_list)  # 残差数据
            purified_factor_data = pd.DataFrame(index=regression_period, columns=lower_class_name_list)  # 因子减去残差得到提纯数据
            # 保存一阶段回归详细结果，以便后续查看提纯情况
            first_regression_result = pd.DataFrame(index=lower_class_name_list, columns=['Adj. R-squared', 'F值', 'F值显著性'])

            for purified_factor in lower_class_name_list:
                # 得到因子回归提纯后的残差
                other_factors_list = [factor for factor in lower_class_name_list if factor not in [purified_factor]]
                if regression_model == 'WLS':
                    first_regression_model = sm.WLS(first_regression_data[purified_factor].astype(float),
                                                    first_regression_data[other_factors_list].astype(float),
                                                    weights=weight_dict['cos']).fit().get_robustcov_results()

                first_regression_residual[purified_factor] = first_regression_model.resid
                purified_factor_data[purified_factor] = \
                    first_regression_data[purified_factor].astype(float) - first_regression_residual[purified_factor]

                first_regression_result.loc[purified_factor, 'Adj. R-squared'] = format(first_regression_model.rsquared_adj,
                                                                                        '.2%')
                first_regression_result.loc[purified_factor, 'F值'] = format(first_regression_model.fvalue[0][0], '.2f')
                first_regression_result.loc[purified_factor, 'F值显著性'] = format(first_regression_model.f_pvalue, '.3f')

            # (2.4) 得到原收益率减去残差值的相关系数矩阵
            within_purified_class_factor_corr[(class_name, regression_end_date)] = purified_factor_data.corr()

            # (2.5) 第二阶段回归：使用残差值再进行一次单因子回归检验，使用原收益率减去残差值再进行一次单因子回归检验
            second_regression_data = pd.DataFrame(index=regression_period, columns=['Alpha', 'Beta'])
            second_regression_data['Alpha'] = 1
            second_regression_data['Beta'] = index_return_df.loc[regression_period, sample_name + '收益率']

            second_regression_result = pd.DataFrame(index=lower_class_name_list, columns=factor_test_info_list)

            for purified_factor in lower_class_name_list:
                if regression_model == 'WLS':
                    second_regression_model = sm.WLS(purified_factor_data[purified_factor],
                                                     second_regression_data[['Alpha', 'Beta']].astype(float),
                                                     weights=weight_dict['cos']).fit().get_robustcov_results()

                second_regression_result.loc[purified_factor] = \
                    set_linear_regression_result(second_regression_model, factor_test_info_list, method='WLS')

            # (2.6) 得到因子类型内提纯后的结果

            # 挑选提纯后仍然显著的因子
            significant_condition = (second_regression_result['Alpha p值'].astype(float) < 0.1) & \
                                    (second_regression_result['Alpha'].astype(float) > 0)

            significant_result = second_regression_result[significant_condition].sort_values(
                by=['Adj. R-squared'], ascending=False).reset_index().rename(columns={'index': lower_class})

            significant_result = tempo_date_data[required_factor_info_list].merge(
                significant_result, on=[lower_class], how='right').sort_values(by=['Adj. R-squared'], ascending=False)
            for info_name in complement_factor_info_list:
                if info_name == '因子大类':
                    significant_result[info_name] = significant_result['因子序号'].map(factor_category_dict)
                elif info_name == '因子小类':
                    significant_result[info_name] = significant_result['因子序号'].map(factor_type_dict)
            significant_result['数据提取日'] = regression_end_date

            tempo_all_purified_data = pd.concat([tempo_all_purified_data, significant_result[all_factor_info_list]])

            # 挑选解释度最高的那个
            most_explaination_result = significant_result.iloc[:1, :].copy()
            tempo_MES_purified_data = pd.concat([tempo_MES_purified_data, most_explaination_result[all_factor_info_list]])

        all_purified_data = pd.concat([all_purified_data, tempo_all_purified_data])
        MES_purified_data = pd.concat([MES_purified_data, tempo_MES_purified_data])

        print('完成因子提纯：' + purified_class + '(' + class_name + ')')

    return all_purified_data, MES_purified_data, within_purified_class_factor_corr


def get_MES_factor_stratification_number_in_factor_number(all_factor_stratification_number_regression_test_result,
                                                          sample_list, regression_model_list, rolling_window_list,
                                                          factor_category_dict, factor_type_dict):
    MES_factor_stratification_number = {}
    factor_info_list = ['因子大类', '因子小类', '因子序号', '档位']
    factor_test_info_list = ['Alpha显著性', 'Alpha', 'Alpha t值', 'Alpha标准误', 'Alpha p值', 'Beta显著性', 'Beta', 'Beta t值', 'Beta标准误', 'Beta p值',
                             'Adj. R-squared']
    all_info_list = ['数据提取日'] + factor_info_list + factor_test_info_list

    for sample_name in sample_list:

        for regression_model in regression_model_list:

            for rolling_window in rolling_window_list:

                tempo_factor_stratification_number = all_factor_stratification_number_regression_test_result[
                    (all_factor_stratification_number_regression_test_result['样本范围'] == sample_name) &
                    (all_factor_stratification_number_regression_test_result['回归模型'] == regression_model) &
                    (all_factor_stratification_number_regression_test_result['滚动窗口'] == rolling_window)].copy()

                tempo_significant_factor_stratifaction_number = tempo_factor_stratification_number[
                    (tempo_factor_stratification_number['Alpha'].astype(float) > 0) &
                    (tempo_factor_stratification_number['Alpha p值'].astype(float) <= 0.1)].copy()

                # (1) 每期每个因子序号下解释度最高的那个Alpha显著为正的档位
                MES_factor_stratification_number[(sample_name, regression_model, rolling_window)] = \
                    tempo_significant_factor_stratifaction_number.groupby(by=['数据提取日', '因子序号']).apply(
                        lambda df: df[['档位'] + factor_test_info_list].sort_values(by='Adj. R-squared', ascending=False).iloc[0, :]).reset_index()

                MES_factor_stratification_number[(sample_name, regression_model, rolling_window)]['因子大类'] = \
                    MES_factor_stratification_number[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_category_dict)
                MES_factor_stratification_number[(sample_name, regression_model, rolling_window)]['因子小类'] = \
                    MES_factor_stratification_number[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_type_dict)

                # (2) 调换columns顺序方便后续查看
                MES_factor_stratification_number[(sample_name, regression_model, rolling_window)] = \
                    MES_factor_stratification_number[(sample_name, regression_model, rolling_window)][all_info_list]

    return MES_factor_stratification_number


def purify_factor_number_in_factor_type(MES_factor_stratification_number, factor_stratification_return, index_return_df,
                                        sample_list, regression_model_list, rolling_window_list,
                                        get_factor_data_date_list, factor_category_dict, factor_type_dict):
    """ 暂时用解释度最高的档位作为该因子序号的代表，在每个因子小类中进行提纯
    :param MES_factor_stratification_number:在某个因子序号中解释度最高的档位
    :param factor_stratification_return:
    :param index_return_df:
    :param sample_list:
    :param regression_model_list:
    :param rolling_window_list:
    :param get_factor_data_date_list:
    :param factor_category_dict:
    :param factor_type_dict:
    :return:参数1是提纯后仍显著的所有因子序号，参数2是提纯后仍显著的Adj. R-squared最高的那个因子序号
    """

    factor_info_list = ['数据提取日','因子大类', '因子小类', '因子序号', '档位']
    factor_test_info_list = ['Alpha显著性', 'Alpha', 'Alpha t值', 'Alpha标准误', 'Alpha p值', 'Beta显著性', 'Beta', 'Beta t值', 'Beta标准误', 'Beta p值',
                             'Adj. R-squared']
    all_info_list = factor_info_list + factor_test_info_list

    all_factor_number_after_purified = {}
    MES_factor_number_after_purified = {}  # Most Explaination Significant

    for sample_name in sample_list:

        for regression_model in regression_model_list:

            for rolling_window in rolling_window_list:

                # (2) 在同一因子小类中，进行因子序号提纯
                all_factor_number_after_purified[(sample_name, regression_model, rolling_window)], \
                MES_factor_number_after_purified[(sample_name, regression_model, rolling_window)], _  = \
                    get_purified_factor(MES_factor_stratification_number[(sample_name, regression_model, rolling_window)],
                                        purified_class='因子小类', lower_class='因子序号', sample_name=sample_name, regression_model=regression_model,
                                        rolling_window=rolling_window, factor_stratification_return=factor_stratification_return,
                                        index_return_df=index_return_df, get_factor_data_date_list=get_factor_data_date_list)

                all_factor_number_after_purified[(sample_name, regression_model, rolling_window)]['因子大类'] = \
                    all_factor_number_after_purified[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_category_dict)
                all_factor_number_after_purified[(sample_name, regression_model, rolling_window)]['因子小类'] = \
                    all_factor_number_after_purified[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_type_dict)
                MES_factor_number_after_purified[(sample_name, regression_model, rolling_window)]['因子大类'] = \
                    MES_factor_number_after_purified[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_category_dict)
                MES_factor_number_after_purified[(sample_name, regression_model, rolling_window)]['因子小类'] = \
                    MES_factor_number_after_purified[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_type_dict)

                all_factor_number_after_purified[(sample_name, regression_model, rolling_window)] = \
                    all_factor_number_after_purified[(sample_name, regression_model, rolling_window)][all_info_list]
                MES_factor_number_after_purified[(sample_name, regression_model, rolling_window)] = \
                    MES_factor_number_after_purified[(sample_name, regression_model, rolling_window)][all_info_list]


                print('完成因子小类内提纯：' + '，'.join([sample_name, regression_model, str(rolling_window)]))
            print('-----------------------------------')

    return all_factor_number_after_purified, MES_factor_number_after_purified


def purify_factor_type_in_factor_category(MES_purified_factor_number, factor_stratification_return, index_return_df,
                                          sample_list, regression_model_list, rolling_window_list,
                                          get_factor_data_date_list, factor_category_dict):

    """ 暂时用解释度最高的因子序号作为该因子小类的代表，在每个因子大类中进行提纯
    :param MES_purified_factor_number:因子小类提纯后仍显著的Adj. R-squared最高的那个因子序号
    :param factor_stratification_return:
    :param index_return_df:
    :param sample_list:
    :param regression_model_list:
    :param rolling_window_list:
    :param get_factor_data_date_list:
    :param factor_category_dict:
    :return:
    """

    factor_info_list = ['数据提取日', '因子大类', '因子小类', '因子序号', '档位']
    factor_test_info_list = ['Alpha显著性', 'Alpha', 'Alpha t值', 'Alpha标准误', 'Alpha p值', 'Beta显著性', 'Beta', 'Beta t值', 'Beta标准误', 'Beta p值',
                             'Adj. R-squared']
    all_info_list = factor_info_list + factor_test_info_list

    all_factor_type_after_purified = {}
    MES_factor_after_purified = {}  # Most Explaination Significant

    for sample_name in sample_list:

        for regression_model in regression_model_list:

            for rolling_window in rolling_window_list:

                # (1) 在同一因子大类中，进行因子小类提纯
                all_factor_type_after_purified[(sample_name, regression_model, rolling_window)], \
                MES_factor_after_purified[(sample_name, regression_model, rolling_window)], _  = \
                    get_purified_factor(MES_purified_factor_number[(sample_name, regression_model, rolling_window)],
                                        purified_class='因子大类', lower_class='因子小类', sample_name=sample_name, regression_model=regression_model,
                                        rolling_window=rolling_window, factor_stratification_return=factor_stratification_return,
                                        index_return_df=index_return_df, get_factor_data_date_list=get_factor_data_date_list)

                # (2) 补充factor信息
                all_factor_type_after_purified[(sample_name, regression_model, rolling_window)]['因子大类'] = \
                    all_factor_type_after_purified[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_category_dict)
                # all_factor_type_after_purified[(sample_name, regression_model, rolling_window)]['因子小类'] = \
                #     all_factor_type_after_purified[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_type_dict)
                MES_factor_after_purified[(sample_name, regression_model, rolling_window)]['因子大类'] = \
                    MES_factor_after_purified[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_category_dict)
                # MES_factor_after_purified[(sample_name, regression_model, rolling_window)]['因子小类'] = \
                #     MES_factor_after_purified[(sample_name, regression_model, rolling_window)]['因子序号'].map(factor_type_dict)

                # (3) 调换columns顺序，方便查看
                all_factor_type_after_purified[(sample_name, regression_model, rolling_window)] = \
                    all_factor_type_after_purified[(sample_name, regression_model, rolling_window)][all_info_list]
                MES_factor_after_purified[(sample_name, regression_model, rolling_window)] = \
                    MES_factor_after_purified[(sample_name, regression_model, rolling_window)][all_info_list]

                print('完成因子大类内提纯：' + '，'.join([sample_name, regression_model, str(rolling_window)]))
            print('-----------------------------------')

    return all_factor_type_after_purified, MES_factor_after_purified


# ----------------------------------------------------------函数（结束）-------------------------------------------------------------------------------

# ----------------------------------------------------------基础数据准备（开始）------------------------------------------------------------------------

raw_data = pickle.load(open('/Users/yi.deng/凌云至善/投研/FOF研究/分组体系/核心驱动因子/weekly_core_factor_raw_data.dat', 'rb'))

output_url = '/Users/yi.deng/凌云至善/投研/FOF研究/分组体系/核心驱动因子'
factor_library = pd.read_excel('/Users/yi.deng/凌云至善/投研/FOF研究/分组体系/核心驱动因子/因子列表_核心驱动因子.xlsx')
get_factor_data_date_list = [date.strftime('%Y-%m-%d') for date in
                             pd.read_excel('/Users/yi.deng/凌云至善/投研/FOF研究/分组体系/核心驱动因子/日期序列-周度.xlsx')['endt'].tolist()]
rolling_window_list = [32, 52, 156, 260]
sample_list = ['申万A股']
# sample_list = ['申万A股', '沪深300', '中证500', '中证800']
stratification_number = 10
quantile_dict = {**{0: 'low'}, **{i: str(i + 1) for i in range(1, stratification_number - 1)}, **{stratification_number - 1: 'high'}}

factor_list = factor_library['factor'].tolist()
factor_name_dict = {factor_library.loc[i, 'factor']: factor_library.loc[i, 'name'] for i in range(factor_library.shape[0])}
factor_type_dict = {factor_library.loc[i, 'factor']: factor_library.loc[i, 'type'] for i in range(factor_library.shape[0])}
factor_category_dict = {factor_library.loc[i, 'factor']: factor_library.loc[i, 'category'] for i in range(factor_library.shape[0])}

base_info_columns_list = raw_data.columns[:23].tolist()
base_info_data = raw_data[base_info_columns_list].copy()
index_return_list = ['申万行业收益率', '沪深300收益率', '中证500收益率', '中证800收益率', '上证综指收益率', '申万A股收益率']
yield_type_list = ['持仓期收益率'] + [index_name[:-3] + '相对' + index_name[-3:] for index_name in index_return_list[1:]]
base_info_columns = base_info_columns_list + [index_name[:-3] + '相对' + index_name[-3:] for index_name in index_return_list[1:]]
# 避免每个变量都保存一遍基础数据，释放内存
data_processing_base_columns = ['数据提取日', 'stockid', '持仓期停牌天数占比']

# 计算指数收益率(因为不想另外再单独取指数的收益率，所以在天软中取基础数据的时候同时取了)
index_return_df = pd.DataFrame(index=get_factor_data_date_list, columns=index_return_list[1:])
for date in get_factor_data_date_list:
    index_return_df.loc[date] = raw_data[raw_data['数据提取日'] == date][index_return_list[1:]].iloc[0, :]

# 计算相对收益率(后续在天软中实现，同时保留指数的收益率和相对收益率)
for index_name in index_return_list[1:]:
    raw_data[index_name[:-3] + '相对' + index_name[-3:]] = raw_data['持仓期收益率'] - raw_data[index_name]

stock_return_df = raw_data[['数据提取日', 'stockid'] + yield_type_list].copy()

# 计算市场平均收益率、中位数收益率
market_mean_return = raw_data.groupby(by=['数据提取日']).apply(lambda df: df[yield_type_list].mean())
market_median_return = raw_data.groupby(by=['数据提取日']).apply(lambda df: df[yield_type_list].median())

# 货币基金收益率
monetary_fund_return = pd.read_excel('/Users/yi.deng/凌云至善/投研/FOF研究/分组体系/核心驱动因子/货币基金收益.xlsx', index_col=0)

# ----------------------------------------------------------基础数据准备（结束）------------------------------------------------------------------------

# ----------------------------------------------------------数据处理及分布描述（开始）-------------------------------------------------------------------

print('-----------------------------------------------------------------------')
print('开始数据清洗')
print('原始数据样本各期数量最大/最小值：' +
      format(raw_data.groupby(by=['数据提取日']).count()['stockid'].min(), '.0f') + '/' +
      format(raw_data.groupby(by=['数据提取日']).count()['stockid'].max(), '.0f'))

clean_data = data_cleaning(raw_data, sample_list=['申万A股'], factor_list=factor_list,
                           kicked_sector_list=['申万金融服务', '申万非银金融', '申万综合', '申万银行'],
                           go_public_days=250, data_processing_base_columns=['数据提取日', 'stockid', '持仓期停牌天数占比'])

print('数据清洗后样本各期数量最大/最小值：' +
      format(clean_data.groupby(by=['数据提取日']).count()['stockid'].min(), '.0f') + '/' +
      format(clean_data.groupby(by=['数据提取日']).count()['stockid'].max(), '.0f'))

print('完成数据清洗')

# 2. 数据分布及异常值处理

print('-----------------------------------------------------------------------')
print('开始数据处理')

# 2.1 数据分布和异常值数据

clean_data_after_outlier, factor_raw_data_describe, factor_clean_data_describe = \
    data_processing(clean_data, raw_data, sample_list=['申万A股'], factor_list=factor_list, factor_name_dict=factor_name_dict)


# ----------------------------------------------------------数据处理及分布描述（结束）-------------------------------------------------------------------

# ----------------------------------------------------------分组构建（开始）----------------------------------------------------------------------------

print('-----------------------------------------------------------------------')
print('开始根据因子生成组合')

factor_stratification_data = get_factor_stratification_data(clean_data_after_outlier, sample_list=['申万A股'], factor_list=factor_list,
                                                            stratification_num=stratification_number, quantile_dict=quantile_dict)

# ****************************************************************************************************************************************************

print('-----------------------------------------------------------------------')
print('开始计算分层收益率')

factor_stratification_return = get_factor_stratification_return(factor_stratification_data, stock_return_df, sample_list=['申万A股'],
                                                                factor_list=factor_list, startification_num=stratification_number,
                                                                quantile_dict=quantile_dict, yield_type_list=['持仓期收益率'])

# ----------------------------------------------------------分组构建（结束）----------------------------------------------------------------------------

# ----------------------------------------------------------时间序列回归计算alpha、beta（开始）----------------------------------------------------------

print('-----------------------------------------------------------------------')
print('开始进行单因子检测')

factor_test_result, _ = get_factor_test_result(factor_stratification_return, index_return_df, sample_list=['申万A股'], factor_list=factor_list,
                                               get_factor_data_date_list=get_factor_data_date_list,
                                               regression_model_list=['WLS'], quantile_dict=quantile_dict, rolling_window_list=[32],
                                               stratification_num=stratification_number)


# ----------------------------------------------------------时间序列回归计算alpha、beta（结束）----------------------------------------------------------

# ----------------------------------------------------------显著因子挑选及所需存储数据（开始）------------------------------------------------------------

# 1. 小类因子收益率

print('-----------------------------------------------------------------------')
print('保存因子序号各档收益率数据')

factor_stratification_hp_return = get_factor_stratification_hp_return(factor_stratification_return, market_mean_return, sample_list=['申万A股'],
                                                                      factor_list=factor_list, stratification_num=stratification_number,
                                                                      quantile_dict=quantile_dict, factor_name_dict=factor_name_dict)

# 2. 将回归结果保存在一个完成的dataframe中，以便后续保存

print('-----------------------------------------------------------------------')
print('保存因子序号各档回归检测结果')

factor_test_result_df = transform_dict_to_dataframe(factor_test_result, ['样本范围', '回归模型', '滚动窗口', '因子序号', '档位'])

print('完成：保存因子序号各档回归检测结果')

# 3. 在显著的(Alpha为正且p值小于1)各档位中，暂时筛选出解释度最高的那个档位作为该因子序号的代表
print('-----------------------------------------------------------------------')
print('筛选解释度最高的有效因子序号档位')

MES_factor_stratification_number = get_MES_factor_stratification_number_in_factor_number(
    factor_test_result_df, ['申万A股'], ['WLS'], [32], factor_category_dict, factor_type_dict)

print('完成：筛选解释度最高的有效因子序号档位')

# 4. 对因子小类进行提纯
print('-----------------------------------------------------------------------')
print('开始因子小类提纯')

all_factor_number_after_purified, MES_factor_number_after_purified = \
    purify_factor_number_in_factor_type(MES_factor_stratification_number, factor_stratification_return, index_return_df, ['申万A股'], ['WLS'], [32],
                                        get_factor_data_date_list, factor_category_dict, factor_type_dict)
print('完成：因子小类提纯')

# 5. 对因子大类进行提纯

print('-----------------------------------------------------------------------')
print('开始因子大类提纯')

all_factor_type_after_purified, MSE_factor_type_after_purified = \
    purify_factor_type_in_factor_category(MES_factor_number_after_purified, factor_stratification_return, index_return_df, ['申万A股'], ['WLS'], [32],
                                          get_factor_data_date_list, factor_category_dict)

print('完成：因子大类提纯')

# ----------------------------------------------------------显著因子挑选及所需存储数据（结束）------------------------------------------------------------

# ----------------------------------------------------------大类因子收益率（开始）-----------------------------------------------------------------------

