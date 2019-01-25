import re
import pandas as pd
pd.set_option('max_columns', 20)
pd.set_option('display.width', 320)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
import numpy as np
from Quantitative_FOF_Multi_Factors_Framework.DataBase import read_data_from_oracle_db
from Quantitative_FOF_Multi_Factors_Framework.Framework.Tools.FactorInfoFunc import fill_factor_number_info
from Quantitative_FOF_Multi_Factors_Framework.Framework.Tools.FactorSignalFunc import standardization
from functools import reduce
import pickle
import statsmodels.api as sm
import scipy.stats as stats


class BaseDataType(object):

    def __init__(self):
        pass

    # columns是数据属性（如sz000001、sz000002或factor1、factor2），index是时间序列
    pivot_data_structure = pd.DataFrame()
    # columns是数据标识（如股票代码、日期、因子值…），index是序号（无特殊形式）
    melt_data_structure = pd.DataFrame()


class BaseData(object):
    """
    用于获取基础数据
    """
    def __init__(self):
        self.stock_return = pd.DataFrame()
        self.time_series_regression_result = pd.DataFrame()
        self.stock_status = pd.DataFrame()
        self.industry_classification = pd.DataFrame()
        self.floating_mv = pd.DataFrame()
        self.factor_raw_data = pd.DataFrame()

    def get_data_from_local_database(self, data_sql=None, data_name=None):
        data = read_data_from_oracle_db(sql=data_sql)
        data.columns = [column.lower() for column in data.columns.tolist()]

        if data_name == 'get_data_date_list':
            return data['get_data_date'].tolist()
        elif data_name == 'stock_return':
            return data.pivot_table(values=data_name, index='get_data_date', columns='stock_id')
        else:
            return data

    def get_stock_return_by_freq(self, calender_freq='周度'):
        """
        用于把日度数据结合成某种频率的收益率
        :param stock_return:
        :param calender_freq:
        :return:
        """
        stock_return_sql = 'select get_data_date, stock_id, stock_return from lyzs_tinysoft.stock_return'
        stock_return = self.get_data_from_local_database(data_sql=stock_return_sql, data_name='stock_return')

        if calender_freq == '周度':
            stock_return = stock_return.div(100)

            trading_day_status = self.get_data_from_local_database(data_sql='select * from lyzs_tinysoft.trading_day_status order by trading_day')
            trading_day_status = trading_day_status.rename(columns={'trading_day': 'get_data_date'})

            stock_list = stock_return.columns.tolist()
            combined_stock_return = trading_day_status[['get_data_date', 'week_number']].merge(stock_return.reset_index(), on=['get_data_date'])
            combined_stock_return = combined_stock_return.groupby(by=['week_number']).apply(
                lambda df: df[stock_list].add(1).cumprod().iloc[-1, :].sub(1)).reset_index()
            combined_stock_return['get_data_date'] = trading_day_status[trading_day_status['week_end_date'] == 1]['get_data_date'].values

            result_data = combined_stock_return[['get_data_date'] + stock_list].set_index('get_data_date')

            return result_data

    def get_base_data(self, data_name=None, data_freq='周度', data_sql=None):

        if data_name == 'stock_return':
            if self.stock_return.empty:
                self.stock_return = self.get_stock_return_by_freq(calender_freq=data_freq)
        elif data_name == 'industry_classification':
            if self.stock_status.empty:
                self.stock_status = self.get_data_from_local_database(
                    data_sql='select * from lyzs_tinysoft.stock_status'
                )
            self.industry_classification = self.stock_status.pivot(values='sw_sector_id', index='get_data_date', columns='stock_id')
        elif data_name == 'floating_mv':
            if self.stock_status.empty:
                self.stock_status = self.get_data_from_local_database(
                    data_sql='select get_data_date, stock_id, floating_mv from lyzs_tinysoft.stock_status')
            self.floating_mv = self.stock_status.pivot_table(values='floating_mv', index='get_data_date', columns='stock_id')

        elif data_name == 'time_series_regression_result':
            if self.time_series_regression_result.empty:
                self.time_series_regression_result = self.get_data_from_local_database(
                    data_sql='select * from lyzs_tinysoft.factor_return_regression'
                )

        elif data_name == 'factor_raw_data':
            if self.factor_raw_data.empty:
                if not data_sql:
                    data_sql = 'select * from lyzs_tinysoft.factor_raw_data'
                self.factor_raw_data = self.get_data_from_local_database(data_sql=data_sql)


class Stock(object):
    def __init__(self):
        self.stock_code = None


class Factor(Stock):
    # 因子就像是坐标系，是为了找到恒星的地图

    def __init__(self):
        super(Stock).__init__()
        self.__factor_number = None  # 类内private属性，类外最好不要访问，若要取值，请用get_factor_number方法
        self.factor_raw_value = None
        self.date = None

    def set_factor_number(self, factor_number):

        if not isinstance(factor_number, str):
            raise TypeError('\tfactor_number必须是字符串类型')
        if len(re.split(r'(factor)(\d+)', factor_number)) not in [2, 3, 4]:  # 将字符串分成s1,factor,XXX,s4四个部分，如果没有匹配到factorXXX，则只有1个部分
            raise ValueError('\tfactor_number必须是factorXXX的格式，XXX为序号，如123')
        # sql = '''
        #     select factor_number
        #     from lyzs_tinysoft.factor_library
        #     where factor_number = '%s'
        # '''
        # if read_data_from_oracle_db(sql % factor_number).empty:
        #     raise ValueError('\t' + factor_number + '不在数据库\"lyzs_tinysoft\"的数据表\"factor_library\"中，请先添加相应的基础数据')

        self.__factor_number = factor_number
        # print('创建了Factor的factor_number')

    def get_factor_number(self):
        return self.__factor_number

    def set_factor_raw_value(self, factor_raw_value):
        self.factor_raw_value = factor_raw_value

    def get_factor_raw_value(self, begin_and_end_date=('', '')):

        if not self.factor_raw_value:
            if begin_and_end_date[0] != '':  # 有指定日期范围的情况下
                sql = '''
                    select stock_id, get_data_date, factor_raw_value
                    from lyzs_tinysoft.factor_raw_data
                    where factor_number = '%s'
                    and get_data_date between '%s' and '%s'
                    order by get_data_date, stock_id
                '''
                self.factor_raw_value = read_data_from_oracle_db(sql % (self.__factor_number, begin_and_end_date[0], begin_and_end_date[1]))
            else:
                sql = '''
                    select stock_id, get_data_date, factor_raw_value
                    from lyzs_tinysoft.factor_raw_data
                    where factor_number = '%s'
                    order by get_data_date, stock_id
                '''
                self.factor_raw_value = read_data_from_oracle_db(sql % self.__factor_number)

        return self.factor_raw_value


class FactorSignal(object):
    # 用于因子信号生成

    def __init__(self, data_freq='周度'):
        self.data_freq = data_freq

    def stratify_ts_data_regression(self):
        # 用于分档形成时间序列数据进行回归

        # 1. 分档收益率时间序列数据、回归要用的时间序列数据

        # 1.1 分档收益率时间序列数据
        # 每个时点：所选股票池内，根据因子从高到低排序，将股票分成若干个组合，每个组合的平均收益即为一条单位数据

        pass


class TimeSeriesDataSignal(FactorSignal):

    def __init__(self):
        super(TimeSeriesDataSignal, self).__init__()


class CrossSectionDataSignal(FactorSignal):

    def __init__(self, base_data=BaseData()):
        super(CrossSectionDataSignal, self).__init__()
        self.stock_return = base_data.stock_return
        self.industry_classification = base_data.industry_classification
        self.floating_mv = base_data.floating_mv

    def regression(self, factor_data=None, used_industry_name='sw_sector_id', factor_number=None):
        """
        函数用于每个截面数据进行回归
        :param factor_data:
        :return:
        """
        # 将self数据先弄出来，防止被修改，且方便调试
        data_freq = self.data_freq

        # 先获取必要数据
        stock_return = self.stock_return

        get_data_date_list = list(set(stock_return.index).intersection(set(factor_data.index)))
        get_data_date_list.sort()

        industry_classification = self.industry_classification
        floating_mv = self.floating_mv
        # factor_return只用于存储因子收益相关的结果
        factor_return = pd.DataFrame(index=get_data_date_list, columns=['factor_return', 't_value', 'p_value', 'adj_rsquared'])

        # 下面的变量用于存储所有的回归结果
        all_industry_list = industry_classification[used_industry_name].dropna().unique().tolist()
        all_regression_result = {}
        all_regression_coef = pd.DataFrame(index=get_data_date_list, columns=['factor', 'market_factor'] + all_industry_list)
        all_regression_t_value = pd.DataFrame(index=get_data_date_list, columns=['factor', 'market_factor'] + all_industry_list)
        all_regression_p_value = pd.DataFrame(index=get_data_date_list, columns=['factor', 'market_factor'] + all_industry_list)
        all_regression_adj_rsquared = pd.Series(index=get_data_date_list)

        # 每个截面分别进行回归
        for j, begin_date in enumerate(get_data_date_list[:-1], 1):

            # 得到本次回归所需要的期初、期末日期
            end_date = get_data_date_list[j]
            # 得到本次截面下的所有股票、行业
            temp_stock_list = factor_data.loc[begin_date].dropna().index.tolist()
            temp_industry_data = industry_classification[industry_classification['get_data_date'] == begin_date]
            industry_dummy_factor = pd.get_dummies(temp_industry_data.set_index('stock_id')[used_industry_name])
            industry_list = industry_dummy_factor.columns.tolist()

            regression_raw_data = pd.DataFrame(
                index=temp_stock_list,
                columns=['stock_return'] + ['factor'] + ['market_factor'] + industry_list
            )

            regression_raw_data['stock_return'] = stock_return.loc[end_date, temp_stock_list].copy()
            regression_raw_data['factor'] = factor_data.loc[begin_date, temp_stock_list].copy()
            regression_raw_data['market_factor'] = 1
            regression_raw_data[industry_list] = industry_dummy_factor

            # 丢掉nan数据
            regression_data = regression_raw_data.dropna(how='any')

            # 得到wls所需要的市值权重
            wls_weight = 1 / floating_mv.loc[begin_date, regression_data.index]

            # 因子数据标准化
            regression_data['factor'] = standardization(regression_data, 'factor', False)['factor'].copy()

            # 做单因子回归
            single_factor_regression_results = sm.WLS(regression_data['stock_return'],
                                                      regression_data[['factor', 'market_factor'] + industry_list].astype(float),
                                                      weights=wls_weight).fit()

            factor_return.loc[end_date, 'factor_return'] = single_factor_regression_results.params['factor']
            factor_return.loc[end_date, 't_value'] = single_factor_regression_results.tvalues['factor']
            factor_return.loc[end_date, 'p_value'] = single_factor_regression_results.pvalues['factor']
            factor_return.loc[end_date, 'adj_rsquared'] = single_factor_regression_results.rsquared_adj

            all_regression_coef.loc[end_date] = single_factor_regression_results.params
            all_regression_t_value.loc[end_date] = single_factor_regression_results.tvalues
            all_regression_p_value.loc[end_date] = single_factor_regression_results.pvalues
            all_regression_adj_rsquared.loc[end_date] = single_factor_regression_results.rsquared_adj

            print(factor_number + ' -> CrossSectionDataSignal -> regression ->', end_date)

        all_regression_result['coef'] = all_regression_coef
        all_regression_result['t_value'] = all_regression_t_value
        all_regression_result['p_value'] = all_regression_p_value
        all_regression_result['adj_rsquared'] = all_regression_adj_rsquared

        return factor_return, all_regression_result

    def correlation(self, factor_data=None, return_lag_number=1):
        """
        用于每个截面数据进行排序相关
        :param factor_data:
        :return:
        """

        # 将self数据先弄出来，防止被修改，且方便调试
        data_freq = self.data_freq
        # 先获取必要数据
        if self.stock_return.empty:
            self.get_base_data('stock_return')
        stock_return = self.stock_return

        get_data_date_list = list(set(stock_return.index).intersection(set(factor_data.index)))
        get_data_date_list.sort()

        if self.industry_classification.empty:
            self.get_base_data('industry_classification')
        industry_classification = self.industry_classification

        if self.floating_mv.empty:
            self.get_base_data('floating_mv')
        floating_mv = self.floating_mv

        factor_ic = pd.DataFrame(index=get_data_date_list, columns=['spearman_correlation', 'p_value'])

        for j, begin_date in enumerate(get_data_date_list[:-return_lag_number], 1):

            # 得到本次回归所需要的期初、期末日期
            end_date = get_data_date_list[j]
            return_end_date = get_data_date_list[j + return_lag_number - 1]


            # 得到本次截面下的所有股票
            stock_list = factor_data.loc[begin_date].dropna().index.tolist()

            correlation_data = pd.DataFrame(
                index=stock_list,
                columns=['stock_return'] + ['factor']
            )

            # 收益率为持仓期收益率
            correlation_data['stock_return'] = \
                stock_return.loc[end_date:return_end_date, stock_list].add(1).cumprod().loc[return_end_date].subtract(1).copy()
            correlation_data['factor'] = factor_data.loc[begin_date, stock_list].copy()

            # 丢掉nan数据
            correlation_data = correlation_data.dropna(how='any')

            # 做单因子回归
            factor_ic.loc[return_end_date, 'spearman_correlation'] = stats.spearmanr(correlation_data['stock_return'], correlation_data['factor'])[0]
            factor_ic.loc[return_end_date, 'p_value'] = stats.spearmanr(correlation_data['stock_return'], correlation_data['factor'])[1]

            print('CrossSectionDataSignal -> correlation (t+' + str(return_lag_number) + ') ->', return_end_date)

        return factor_ic

    def ic_time_attenuate(self, factor_data=None, max_lag_number=6):

        factor_ic = pd.DataFrame(columns=['ic_' + str(i) for i in range(1, max_lag_number + 1)])
        factor_ic_p_value = pd.DataFrame(columns=['ic_' + str(i) for i in range(1, max_lag_number + 1)])

        for i in range(1, max_lag_number + 1):
            ic_result = self.correlation(factor_data=factor_data, return_lag_number=i)
            factor_ic['ic_' + str(i)] = ic_result['spearman_correlation']
            factor_ic_p_value['ic_' + str(i)] = ic_result['p_value']

        return factor_ic, factor_ic_p_value


class SignalTesting(object):

    def __init__(self):
        pass


class TimeSeriesDataSignalTesting(SignalTesting, BaseData):
    # 用于时间序列数据回归产生的信号检验数据生成，暂时不负责生成检验结果

    def __init__(self, regression_result=pd.DataFrame()):
        super(SignalTesting).__init__()
        self.regression_result = regression_result

    def get_alpha_abs_info(self):

        if self.regression_result.empty:
            self.get_base_data(data_name='time_series_regression_result')
        regression_result = self.regression_result

        # 所有档位alpha绝对值加总
        alpha_abs_sum = regression_result.groupby(by=['regression_model', 'rolling_window', 'factor_number', 'get_data_date']).apply(
            lambda df: ((df['alpha_p_value'] <= 0.1) * df['alpha'].abs()).sum()).reset_index().rename(columns={0: 'sAlpha_absSum'})
        alpha_abs_mean = regression_result.groupby(by=['regression_model', 'rolling_window', 'factor_number', 'get_data_date']).apply(
            lambda df: ((df['alpha_p_value'] <= 0.1) * df['alpha'].abs()).mean()).reset_index().rename(columns={0: 'sAlpha_absMean'})
        alpha_abs_median = regression_result.groupby(by=['regression_model', 'rolling_window', 'factor_number', 'get_data_date']).apply(
            lambda df: ((df['alpha_p_value'] <= 0.1) * df['alpha'].abs()).median()).reset_index().rename(columns={0: 'sAlpha_absMedian'})

        merge_dfs = [alpha_abs_sum, alpha_abs_mean, alpha_abs_median]
        alpha_abs_info = reduce(
            lambda left_df, right_df:
            pd.merge(left_df, right_df, on=['regression_model', 'rolling_window', 'factor_number', 'get_data_date'], how='left'),
            merge_dfs)
        alpha_abs_info = fill_factor_number_info(alpha_abs_info)

        return alpha_abs_info


class CrossSectionDataSignalTesting(SignalTesting):

    def __init__(self):
        super(SignalTesting).__init__()


class FactorReturn(Factor):

    def __init__(self):
        super(Factor).__init__()

    # def set_factor_number(self, factor_number):
    # 如果不用super函数，那么相当于对父类Factor中的set_factor_number进行重构，并且覆盖父类方法；如果用的super，相对于继承了父类中的该方法
    #     super(FactorReturn, self).set_factor_number(factor_number)
    #     print('创建了FactorReturn的factor_number')

    def get_factor_return(self, begin_and_end_date=('', ''), type_name='分档'):

        if type_name == '分档':

            if begin_and_end_date[0] != '':  # 有指定日期范围的情况下
                sql = '''
                    select get_data_date, type_name
                    from lyzs_tinysoft.factor_return
                    where factor_number = '%s'
                    and get_data_date between '%s' and '%s'
                    order by get_data_date, type_name
                '''
                self.factor_raw_data = read_data_from_oracle_db(sql % (self.__factor_number, begin_and_end_date[0], begin_and_end_date[1]))
            else:
                sql = '''
                    select stock_id, get_data_date, factor_raw_value
                    from lyzs_tinysoft.factor_return
                    where factor_number = '%s'
                    order by get_data_date, stock_id
                '''
                self.factor_raw_data = read_data_from_oracle_db(sql % self.__factor_number)


class FactorExposure(Factor):

    def __init__(self):
        super(Factor).__init__()


if __name__ == '__main__':

    base_data = BaseData()

    # 截面信号:

    base_data.get_base_data(data_name='stock_return')
    base_data.get_base_data(data_name='industry_classification')
    base_data.get_base_data(data_name='floating_mv')
    base_data.get_base_data(data_name='factor_raw_data')  # 默认取所有的因子数据

    cs_signal = CrossSectionDataSignal(base_data=base_data)

    factor_number_list = base_data.factor_raw_data['factor_number'].unique().tolist()
    factor_return_dict = {}
    all_regression_result_dict = {}

    for factor_number in factor_number_list:

        temp_data = base_data.factor_raw_data[base_data.factor_raw_data['factor_number'] == factor_number].copy()
        temp_pivot_data = temp_data.pivot_table(values='raw_value', index='get_data_date', columns='stock_id')
        factor_return_dict[factor_number], all_regression_result_dict[factor_number] = \
            cs_signal.regression(factor_data=temp_pivot_data, factor_number=factor_number)
        # cs_factor_ic, cs_factor_ic_p_value = cs_signal.ic_time_attenuate(factor_data=factor_raw_data, max_lag_number=6)

    # 时间序列信号检验
    base_data.get_base_data(data_name='time_series_regression_result')
    base_data.time_series_regression_result = fill_factor_number_info(base_data.time_series_regression_result)

    ts_signal_test = TimeSeriesDataSignalTesting(regression_result=base_data.time_series_regression_result)

    # 现在有一堆信息（信号+噪音），我一只在想怎么样做一个大筛子去有效的筛掉更多的噪音。但是也许这样做不对呢？
    # 所以现在尝试：就直接构建一个策略，看看就拿情绪因子来做策略会怎样

    test_data = base_data.time_series_regression_result[(base_data.time_series_regression_result['first_class'] == '情绪') &
                                                        (base_data.time_series_regression_result['regression_model'] == 'OLS') &
                                                        (base_data.time_series_regression_result['rolling_window'] == 260)].copy()
    test_signal = TimeSeriesDataSignalTesting(regression_result=test_data)
    alpha_info = test_signal.get_alpha_abs_info()

