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
import re
import sys
import math
import cvxpy


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
    def __init__(self, begin_date, data_freq):
        self.begin_date = begin_date
        self.data_freq = data_freq
        # self.stock_return = pd.DataFrame()
        self.stock_active_return = pd.DataFrame()
        self.time_series_regression_result = pd.DataFrame()
        self.stock_status = pd.DataFrame()
        self.industry_classification = pd.DataFrame()
        self.floating_mv = pd.DataFrame()
        self.factor_raw_data = pd.DataFrame()
        self.date_list = None
        self.benchmark_weight = pd.DataFrame()
        self.factor_stratificated_return = pd.DataFrame()
        self.factor_return_regression = pd.DataFrame()

    def optimize_data_ram(self, data):
        """
        主要是将int、float、object等类型转变为小一点的类型
        :param data:
        :return:
        """
        action_str = '优化变量存储结构'
        print('\n' + '{0:*>{width}}'.format(action_str, width=40) + '{0:*>{width}}'.format('', width=40 - len(action_str)) + '\n')

        if isinstance(data, pd.DataFrame):

            print('\t传入数据：' + ','.join([str(type) + '(' + str(num) + '列)' for type, num in data.get_dtype_counts().to_dict().items()]))
            before_memory = data.memory_usage(deep=True).sum() / 1024 ** 2
            print('\t传入数据大小：' + "{:03.2f}MB".format(before_memory))
            if before_memory <= 100:
                print('\t数据大小小于100M，暂时不进行优化')
                return data
            print('\t正在优化数据结构及存储空间……')

            if not data.select_dtypes(include=['int']).empty:
                data[data.select_dtypes(include=['int']).columns] = \
                    data.select_dtypes(include=['int']).apply(pd.to_numeric, downcast='integer')  # 最小为int8

            if not data.select_dtypes(include=['float']).empty:
                data[data.select_dtypes(include=['float']).columns] = \
                    data.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')  # 最小为float32

            if not data.select_dtypes(include=['object']).empty:
                for col in data.select_dtypes(include=['object']).columns:
                    num_unique_values = len(data.select_dtypes(include=['object'])[col].unique())
                    num_total_values = len(data.select_dtypes(include=['object'])[col])
                    if num_unique_values / num_total_values < 0.5:  # 因为是用字典存，所以重复率较高的数据才适合
                        data.loc[:, col] = data.select_dtypes(include=['object'])[col].astype('category')  # 将object转为catagory
            print('\t优化后数据：' + ','.join([str(type) + '(' + str(num) + '列)' for type, num in data.get_dtype_counts().to_dict().items()]))
            print('\t优化后数据大小：' + "{:03.2f}MB".format(data.memory_usage(deep=True).sum() / 1024 ** 2))
            change_pct = before_memory / (data.memory_usage(deep=True).sum() / 1024 ** 2) - 1
            print('\t数据存储优化幅度：' + format(change_pct, '.2%'))

        elif isinstance(data, dict):
            type_count = {}
            for key, df in data.items():
                for type, count in df.get_dtype_counts().to_dict().items():
                    if type in type_count.keys():
                        type_count[type] += count
                    else:
                        type_count[type] = count
            print('\t传入数据：' + ', '.join([str(type) + '(' + str(count) + ')' for type, count in type_count.items()]))
            before_memory = sys.getsizeof(data) / 1024 ** 2
            if before_memory <= 100:
                print('\t数据大小小于100M，暂时不进行优化')
                return data
            print('\t传入数据大小：' + "{:03.2f}MB".format(before_memory))

            for key, df in data.items():

                if not df.select_dtypes(include=['int']).empty:
                    df[df.select_dtypes(include=['int']).columns] = \
                        df.select_dtypes(include=['int']).apply(pd.to_numeric, downcast='integer')  # 最小为int8

                if not df.select_dtypes(include=['float']).empty:
                    df[df.select_dtypes(include=['float']).columns] = \
                        df.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')  # 最小为float32

                if not df.select_dtypes(include=['object']).empty:
                    for col in df.select_dtypes(include=['object']).columns:
                        num_unique_values = len(df.select_dtypes(include=['object'])[col].unique())
                        num_total_values = len(df.select_dtypes(include=['object'])[col])
                        if num_unique_values / num_total_values < 0.5:  # 因为是用字典存，所以重复率较高的数据才适合
                            df.loc[:, col] = df.select_dtypes(include=['object'])[col].astype('category')  # 将object转为catagory
                data[key] = df

            type_count = {}
            for key, df in data.items():
                for type, count in df.get_dtype_counts().to_dict().items():
                    if type in type_count.keys():
                        type_count[type] += count
                    else:
                        type_count[type] = count
            print('\t优化后数据：' + ', '.join([str(type) + '(' + str(count) + ')' for type, count in type_count.items()]))
            after_memory = sys.getsizeof(data) / 1024 ** 2
            print('\t优化后数据大小：' + "{:03.2f}MB".format(after_memory))
            change_pct = before_memory / after_memory - 1
            print('\t数据存储优化幅度：' + format(change_pct, '.2%'))

        print('\n' + '{0:*>{width}}'.format(action_str, width=40) + '{0:*>{width}}'.format('', width=40 - len(action_str)) + '\n')
        return data

    def get_data_from_local_database(self, data_sql=None, data_name=None, optimize_data=False):
        data = read_data_from_oracle_db(sql=data_sql)
        data.columns = [column.lower() for column in data.columns.tolist()]
        if optimize_data:
            data = self.optimize_data_ram(data)

        if data_name == 'get_data_date_list':
            return data['get_data_date'].tolist()
        elif (data_name == 'stock_return') | (data_name == 'stock_active_return'):
            return data.pivot_table(values=data_name, index='get_data_date', columns='stock_id')
        else:
            return data

    def get_stock_return_by_freq(self, active_benchmark=None):
        """
        用于把日度数据结合成某种频率的收益率
        :param stock_return:
        :param calender_freq:
        :return:
        """
        if active_benchmark:
            stock_return_sql = '''
            select a.get_data_date, a.stock_id, a.stock_return, b.DAILY_RETURN, a.STOCK_RETURN - b.DAILY_RETURN as stock_active_return
            from lyzs_tinysoft.stock_return a
            LEFT JOIN LYZS_TINYSOFT.INDEX_RETURN b
            on a.GET_DATA_DATE = b.GET_DATA_DATE
            where b.STOCK_NAME = \'''' + active_benchmark + '''\' and a.GET_DATA_DATE >= \'''' + self.begin_date + '''\'
            ORDER BY a.GET_DATA_DATE, a.STOCK_ID
            '''
            stock_return = self.get_data_from_local_database(data_sql=stock_return_sql, data_name='stock_active_return').astype(np.float16)
        else:
            stock_return_sql = '''select get_data_date, stock_id, stock_return from lyzs_tinysoft.stock_return 
            where get_data_date >= \'''' + self.begin_date + '''\'
            '''
            stock_return = self.get_data_from_local_database(data_sql=stock_return_sql, data_name='stock_return').astype(np.float16)

        if self.data_freq == '周度':
            stock_return = stock_return.div(100)

            trading_day_status = self.get_data_from_local_database(
                data_sql='''select * from lyzs_tinysoft.trading_day_status 
                where trading_day >= \'''' + self.begin_date + '''\' order by trading_day
                ''')
            trading_day_status = trading_day_status.rename(columns={'trading_day': 'get_data_date'})

            stock_list = stock_return.columns.tolist()
            combined_stock_return = trading_day_status[['get_data_date', 'week_number']].merge(stock_return.reset_index(), on=['get_data_date'])
            combined_stock_return = combined_stock_return.groupby(by=['week_number']).apply(
                lambda df: df[stock_list].add(1).cumprod().iloc[-1, :].sub(1)).reset_index()
            combined_stock_return['get_data_date'] = trading_day_status[trading_day_status['week_end_date'] == 1]['get_data_date'].values

            result_data = combined_stock_return[['get_data_date'] + stock_list].set_index('get_data_date')

            return result_data

    def get_base_data(self, data_name=None, data_sql=None, active_benchmark=None):

        # if data_name == 'stock_return':
        #     if self.stock_return.empty:
        #         self.stock_return = self.get_stock_return_by_freq()
        if data_name == 'stock_active_return':
            if self.stock_active_return.empty:
                self.stock_active_return = self.get_stock_return_by_freq(active_benchmark)
        elif data_name == 'stock_status':
            if self.stock_status.empty:
                self.stock_status = self.get_data_from_local_database(
                    data_sql='select * from lyzs_tinysoft.stock_status where get_data_date >= \'' + self.begin_date + '\''
                )
        elif data_name == 'industry_classification':
            if self.stock_status.empty:
                self.stock_status = self.get_data_from_local_database(
                    data_sql='select * from lyzs_tinysoft.stock_status where get_data_date >= \'' + self.begin_date + '\''
                )
            # self.industry_classification = self.stock_status.pivot(values='sw_sector_id', index='get_data_date', columns='stock_id')
            self.industry_classification = self.stock_status[['get_data_date', 'stock_id', 'sw_sector_id']]
        elif data_name == 'floating_mv':
            if self.stock_status.empty:
                self.stock_status = self.get_data_from_local_database(
                    data_sql='''
                    select get_data_date, stock_id, floating_mv from lyzs_tinysoft.stock_status 
                    where get_data_date >= \'''' + self.begin_date + '''\'
                    ''')
            self.floating_mv = self.stock_status.pivot_table(values='floating_mv', index='get_data_date', columns='stock_id')

        elif data_name == 'time_series_regression_result':
            if self.time_series_regression_result.empty:
                self.time_series_regression_result = self.get_data_from_local_database(
                    data_sql='select * from lyzs_tinysoft.factor_return_regression where get_data_date >= \'' + self.begin_date + '\'',
                    optimize_data=True
                )

        elif data_name == 'factor_raw_data':
            if self.factor_raw_data.empty:
                if not data_sql:
                    data_sql = 'select * from lyzs_tinysoft.factor_raw_data where get_data_date >= \'' + self.begin_date + '\''
                self.factor_raw_data = self.get_data_from_local_database(data_sql=data_sql, optimize_data=True)

        elif data_name == 'date_list':
            if not self.date_list:
                self.date_list = self.get_data_from_local_database(
                    '''
                    select get_data_date from lyzs_tinysoft.get_data_date_library 
                    where get_data_freq = \'''' + self.data_freq + '''\' and get_data_date >= \'''' + self.begin_date + '''\'
                    '''
                )['get_data_date'].tolist()
        elif data_name == 'benchmark_weight':
            if self.benchmark_weight.empty:
                self.benchmark_weight = self.get_data_from_local_database(
                    data_sql='''
                    select get_data_date, stock_id, weight from lyzs_tinysoft.index_weight
                    where index_name = \'''' + active_benchmark + '''\' and get_data_date >= \'''' + self.begin_date + '''\'
                    '''
                )

        elif data_name == 'all_stocks_list':
            if self.stock_status.empty:
                self.stock_status = self.get_data_from_local_database(
                    data_sql='''
                    select get_data_date, stock_id, floating_mv from lyzs_tinysoft.stock_status 
                    where get_data_date >= \'''' + self.begin_date + '''\'
                    ''')
            self.all_stocks_list = self.stock_status['stock_id'].unique().tolist()

        elif data_name == 'factor_stratificated_return':
            if self.factor_stratificated_return.empty:
                self.factor_stratificated_return = self.get_data_from_local_database(
                    data_sql='''
                    select get_data_date, factor_number, type_name, return from lyzs_tinysoft.factor_stratificated_return
                    '''
                )
        elif data_name == 'factor_return_regression':
            if self.factor_return_regression.empty:
                self.factor_return_regression = self.get_data_from_local_database(
                    data_sql='''
                    select * from lyzs_tinysoft.factor_return_regression
                    '''
                )


class Model(object):

    def __init__(self):
        pass

    def cross_section_data_regression(self, left_var_data=None, factor_melt_data=None, industry_classification_melt_data=None,
                                      floating_mv_pivot_data=None, used_industry_name='sw_sector_id'):

        # left_var_data是等式左边的变量，为pivot类型
        # factor_melt_data为melt类型

        # 先获取必要数据
        stock_active_return = left_var_data
        get_data_date_list = list(set(stock_active_return.index).intersection(set(factor_melt_data['get_data_date'])))
        get_data_date_list.sort()

        factor_list = factor_melt_data['factor_number'].unique().tolist()
        all_stock_list = factor_melt_data['stock_id'].unique().tolist()
        # 下面的变量用于存储所有的回归结果
        all_industry_list = industry_classification_melt_data[used_industry_name].dropna().unique().tolist()
        right_var_list = factor_list + all_industry_list

        # 用于存储因子收益相关的结果
        regression_result = {i: pd.DataFrame(index=get_data_date_list, columns=right_var_list) for i in ['coef', 'coef_t_value', 'coef_p_value']}
        regression_result['r-suqared'] = pd.DataFrame(index=get_data_date_list, columns=['r-squared', 'Adj-rsquared'])
        regression_residual = pd.DataFrame(index=get_data_date_list, columns=all_stock_list)

        factor_exposure = {}

        # 每个截面分别进行回归
        for j, begin_date in enumerate(get_data_date_list[:-1], 1):

            begin_date_factor_data = factor_melt_data[factor_melt_data['get_data_date'] == begin_date].pivot_table(
                values='raw_value', index='stock_id', columns='factor_number')
            begin_date_industry_data = industry_classification_melt_data[industry_classification_melt_data['get_data_date'] == begin_date]
            # 得到本次回归所需要的期初、期末日期
            end_date = get_data_date_list[j]

            # 得到本次截面下的所有股票、行业
            begin_date_stock_list = begin_date_factor_data.index.tolist()
            begin_date_industry_list = begin_date_industry_data[used_industry_name].dropna().unique().tolist()
            begin_date_right_var_list = factor_list + begin_date_industry_list

            # 初始化回归数据
            regression_raw_data = pd.DataFrame(index=begin_date_stock_list, columns=begin_date_right_var_list)

            regression_raw_data['stock_active_return'] = stock_active_return.loc[end_date, begin_date_stock_list]
            regression_raw_data[factor_list] = begin_date_factor_data.loc[begin_date_stock_list, factor_list]
            # regression_raw_data['market_factor'] = 1
            regression_raw_data[begin_date_industry_list] = pd.get_dummies(begin_date_industry_data.set_index('stock_id')[used_industry_name]).loc[
                begin_date_stock_list, begin_date_industry_list]

            # 丢掉nan数据
            regression_data = regression_raw_data.dropna(how='any')

            # 得到wls所需要的市值权重
            wls_weight = 1 / floating_mv_pivot_data.loc[begin_date, regression_data.index]

            # 因子数据标准化
            regression_data[factor_list] = regression_raw_data[factor_list].apply(
                lambda f: standardization(regression_raw_data, f.name, False)[f.name], axis=0)
            factor_exposure[begin_date] = regression_raw_data[begin_date_right_var_list]

            regression_results = sm.WLS(regression_data['stock_active_return'],
                                        regression_data[begin_date_right_var_list].astype(float),
                                        weights=wls_weight).fit()

            regression_result['coef'].loc[end_date] = regression_results.params
            regression_result['coef_t_value'].loc[end_date] = regression_results.tvalues
            regression_result['coef_p_value'].loc[end_date] = regression_results.pvalues
            regression_result['r-suqared'].loc[end_date] = [regression_results.rsquared, regression_results.rsquared_adj]

            regression_residual.loc[begin_date] = regression_results.resid

            print('Factors Model('+ str(len(factor_list)) + ' factors) -> linear regression ->', end_date)

        return regression_result, factor_exposure, regression_residual

    def time_series_data_regression(self):
        pass


class Factor(object):
    # 因子就像是坐标系，是为了找到恒星的地图

    def __init__(self):
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


class CrossSectionDataSignal(FactorSignal):

    def __init__(self, base_data):
        super(CrossSectionDataSignal, self).__init__()
        self.base_data = base_data
        self.stock_active_return = base_data.stock_active_return
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
        stock_active_return = self.stock_active_return

        get_data_date_list = list(set(stock_active_return.index).intersection(set(factor_data.index)))
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
                columns=['stock_active_return'] + ['factor'] + ['market_factor'] + industry_list
            )

            regression_raw_data['stock_active_return'] = stock_active_return.loc[end_date, temp_stock_list].copy()
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
            single_factor_regression_results = sm.WLS(regression_data['stock_active_return'],
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
        if self.stock_active_return.empty:
            self.base_data.get_base_data('stock_active_return')
        stock_active_return = self.stock_active_return

        get_data_date_list = list(set(stock_active_return.index).intersection(set(factor_data.index)))
        get_data_date_list.sort()

        if self.industry_classification.empty:
            self.base_data.get_base_data('industry_classification')
        industry_classification = self.industry_classification

        if self.floating_mv.empty:
            self.base_data.get_base_data('floating_mv')
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
                columns=['stock_active_return'] + ['factor']
            )

            # 收益率为持仓期收益率
            correlation_data['stock_active_return'] = \
                stock_active_return.loc[end_date:return_end_date, stock_list].add(1).cumprod().loc[return_end_date].subtract(1).copy()
            correlation_data['factor'] = factor_data.loc[begin_date, stock_list].copy()

            # 丢掉nan数据
            correlation_data = correlation_data.dropna(how='any')

            # 做单因子回归
            factor_ic.loc[return_end_date, 'spearman_correlation'] = stats.spearmanr(correlation_data['stock_active_return'], correlation_data['factor'])[0]
            factor_ic.loc[return_end_date, 'p_value'] = stats.spearmanr(correlation_data['stock_active_return'], correlation_data['factor'])[1]

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


class BackTest(Model):

    def __init__(self, base_data):
        super(BackTest, self).__init__()
        self.date_list = base_data.date_list
        self.stock_status = base_data.stock_status
        self.all_stocks_list = base_data.all_stocks_list

    def forecast_factor_return(self, factor_return_data):
        current_data_length = factor_return_data.shape[0]
        cos_value = [math.cos(x) for x in np.linspace(-math.pi / 2, 0, current_data_length)]
        weight_list = [x / sum(cos_value) for x in cos_value]
        forecasted_factor_return = factor_return_data.mul(weight_list, axis=0).cumsum().iloc[-1, :].dropna()
        return forecasted_factor_return

    def forecast_factor_residual_risk(self):
        pass

    def get_prior_return_and_risk(self, post_factor_return=None, factor_exposure_data=None, post_stock_residual_return=None,
                                  date=None, all_factors=None, feasible_stocks_list=None):

        # 3.2 t+1期因子预测收益率、股票残差收益率
        forecasted_factor_return = self.forecast_factor_return(post_factor_return.loc[:date, all_factors].dropna(how='all'))
        forecasted_stock_beta_return = factor_exposure_data.mul(forecasted_factor_return).sum(axis=1)
        forecasted_stock_residual_return = self.forecast_factor_return(post_stock_residual_return.loc[:date, feasible_stocks_list])
        forecasted_stock_total_return = forecasted_stock_beta_return + forecasted_stock_residual_return

        # 3.3 t+1期因子预测共同风险矩阵
        factors_return_cov = np.asmatrix(post_factor_return.loc[:date, all_factors].dropna(how='all').astype(float).cov())
        factor_exposure_matrix = np.asmatrix(factor_exposure_data)
        forecasted_common_risk = factor_exposure_matrix.dot(factors_return_cov).dot(factor_exposure_matrix.T)

        # 3.4 t+1期股票残差收益率 -> t+1期异质风险矩阵
        stock_residual_return_matrix = post_stock_residual_return.loc[:date, feasible_stocks_list].copy()
        stock_residual_return_matrix.loc['预测', feasible_stocks_list] = forecasted_stock_residual_return
        forecasted_idiosyncratic_risk = np.diag(stock_residual_return_matrix.astype(float).var())

        # 3.5 加总风险矩阵，注意是var不是std
        forecasted_risk = forecasted_common_risk + forecasted_idiosyncratic_risk
        return forecasted_stock_total_return, forecasted_risk

    def get_historical_post_return_and_risk(self, date_list=None, factor_exposure_data=None, factor_return_data=None,
                                            stock_residual_return_data=None):

        post_return = {}
        post_risk = {}

        for j, date in enumerate(date_list[:-1]):
            post_date = date_list[j + 1]

            date_factors = factor_exposure_data[date].columns.tolist()
            date_stocks = factor_exposure_data[date].index.tolist()

            # return
            post_factor_return = factor_return_data.loc[post_date]
            stock_beta_return = factor_exposure_data[date].mul(post_factor_return.loc[date_factors], axis=1).sum(axis=1)
            stock_residual_return = stock_residual_return_data.loc[post_date, date_stocks]
            stock_post_return = stock_beta_return + stock_residual_return

            # risk
            post_factor_return_cov = factor_return_data.loc[:post_date, date_factors].astype(float).cov()
            post_factor_return_cov_matrix = np.asmatrix(post_factor_return_cov)
            factor_exposure_matrix = np.asmatrix(factor_exposure_data[date])
            common_risk_matrix = factor_exposure_matrix.dot(post_factor_return_cov_matrix).dot(factor_exposure_matrix.T)
            idiosyncratic_risk = stock_residual_return_data.loc[:post_date, date_stocks].astype(float).var()
            idiosyncratic_risk_matrix = np.asmatrix(np.diag(idiosyncratic_risk))
            risk_matrix = common_risk_matrix + idiosyncratic_risk_matrix

            post_return[date] = stock_post_return
            post_risk[date] = risk_matrix
            print('BackTest -> get post return and risk -> ' + date)

        return post_return, post_risk

    def optimization(self, forecasted_stock_return=None, forecasted_risk=None, date_factor_exposure_data=None, benchmark_weight_data=None,
                     stock_industry_classification=None, date=None, industries=None, stocks_list=None, current_position=None):

        # stocks_list是本次优化可以交易的股票池，current_position是本次可以交易的且有持仓的信息
        stocks_count = len(stocks_list)
        x = cvxpy.Variable(shape=(stocks_count, 1))
        if current_position is not None:
            x_0 = current_position.values
            x_s = cvxpy.Variable(shape=(stocks_count, 1))
        else:
            x_0 = np.zeros(stocks_count)
            x_s = np.zeros(stocks_count)
            x_s.shape = (stocks_count, 1)

        x_0.shape = (stocks_count, 1)
        x_b = cvxpy.Variable(shape=(stocks_count, 1))

        miu = 0.5
        q = forecasted_stock_return.values.copy()
        q.shape = (stocks_count, 1)
        portfolio_return = q.T * x
        risk_punishment = cvxpy.quad_form(x, miu * forecasted_risk)

        buy_cost_series = np.array([0.0003] * stocks_count)
        buy_cost_series.shape = (stocks_count, 1)
        buy_cost = buy_cost_series.T * x_b
        sale_cost_series = np.array([0.0013] * stocks_count)
        sale_cost_series.shape = (stocks_count, 1)
        if current_position is not None:
            sale_cost = sale_cost_series.T * x_s
        else:
            sale_cost = sale_cost_series.T.dot(x_s)

        # 等式约束
        eq = []

        # a. 行业中性

        A = np.asmatrix(date_factor_exposure_data[industries].T.astype(float).values)
        date_benchmark_stock_weight = benchmark_weight_data[benchmark_weight_data['get_data_date'] == date].dropna()
        date_benchmark_stock_industry = stock_industry_classification[stock_industry_classification['get_data_date'] == date]
        industry_weights = date_benchmark_stock_weight.merge(date_benchmark_stock_industry, on=['stock_id'])[['sw_sector_id', 'weight']]
        industry_weights_sum = industry_weights.groupby(by=['sw_sector_id']).sum().loc[industries]

        b = industry_weights_sum.values
        # st_1_right_matrix.shape = (len(industries), 1)

        eq += [A * x == b]

        # b. 仓位约束

        A1 = np.ones(stocks_count)
        A1.shape = (stocks_count, 1)
        b1 = np.array([1])
        b1.shape = (1, 1)
        eq += [A1.T * x == b1]

        # c. 多空交易成本约束

        eq += [x - x_0 == x_b - x_s]

        # 不等式约束
        ineq = []

        # a. 个股权重上下限
        G = np.diag(np.ones(stocks_count)).astype(float)

        # 注意，很可能由于上下限约束太强，导致无解
        cap_bound = np.array([0.01] * stocks_count)
        cap_bound.shape = (stocks_count, 1)
        floor_bound = np.array([0] * stocks_count)
        floor_bound.shape = (stocks_count, 1)
        ineq += [G * x >= floor_bound, G * x <= cap_bound]

        # b. 多空交易仓位
        if current_position is not None:
            ineq += [x_b >= 0, x_s >= 0]

        optimums_results = cvxpy.Problem(cvxpy.Maximize(portfolio_return - risk_punishment - buy_cost - sale_cost), eq + ineq)
        optimums_results.solve(solver=cvxpy.SCS, verbose=True)
        if current_position is not None:
            cost = pd.DataFrame(buy_cost.value).iloc[0, 0] + pd.DataFrame(sale_cost.value).iloc[0, 0]
        else:
            cost = pd.DataFrame(buy_cost.value).iloc[0, 0]

        return optimums_results, x, cost

    def back_test(self, factors=None, factor_exposure_data=None, factor_return_data=None, stock_residual_return=None, benchmark_weight_data=None,
                  stock_industry_classification=None):

        back_test_date_list = pd.Series(back_test.date_list)[pd.Series(back_test.date_list) >= '2010-01-01'].tolist()
        back_test_portfolio_info = pd.DataFrame(index=back_test_date_list,
                                                columns=['solver_status', 'return', 'risk', 'total_cost'])
        stocks_list_info = pd.DataFrame(index=back_test_date_list, columns=['feasible'])
        back_test_stocks_weight = pd.DataFrame(index=back_test_date_list, columns=back_test.all_stocks_list)
        holding_position_info = {}
        trading_detail = pd.DataFrame()

        for j, date in enumerate(back_test_date_list):

            # 每个date进行组合优化得到最优权重

            # 1.  准备基础数据
            all_factors = factor_exposure_data[date].columns.tolist()
            stocks = factor_exposure_data[date].index.tolist()
            industries = list(set(all_factors).difference(set(factors)))

            # 2.  确定当前时点能够交易对股票池
            # 2.1 没st、没pt、没停牌、上市时间超过1年的股票
            stocks_list_1 = back_test.stock_status[(back_test.stock_status['get_data_date'] == date) &
                                                   (back_test.stock_status['is_st'] == 0) &
                                                   (back_test.stock_status['is_pt'] == 0) &
                                                   (back_test.stock_status['is_suspended'] == 0) &
                                                   (back_test.stock_status['days_from_public_date'] >= 365)]['stock_id'].tolist()

            # 2.2 拥有足够异质风险数据的股票
            # 由于因子收益率是每个截面回归得到的，因子残差收益率也是截面上的，所以会造成这个截面上的股票有一部分残差收益率的时间序列数据确实严重
            residual_enough_stocks = stock_residual_return.loc[:date, stocks].apply(
                lambda s: s.dropna().shape[0] >= 52, axis=0).replace(False, np.nan).dropna().index.tolist()

            feasible_stocks_list = list(set(stocks_list_1).intersection(set(residual_enough_stocks)))
            feasible_stocks_list.sort()
            stocks_list_info.loc[date, 'feasible'] = feasible_stocks_list

            # 2.3 确定当前有持仓且能参与优化的股票池
            if j == 0:
                current_position = None
            else:
                current_position = holding_position_info[back_test_date_list[j - 1]].loc[feasible_stocks_list].fillna(0).copy()

            # 3.  date日期的每个股票的因子暴露;每个因子的预测收益率;预测的共同风险矩阵;预测的异质风险矩阵
            # 3.1 t期因子暴露
            date_factor_exposure_data = factor_exposure_data[date].loc[feasible_stocks_list, all_factors]

            prior_return, prior_risk = back_test.get_prior_return_and_risk(post_factor_return=factor_return_data,
                                                                      factor_exposure_data=date_factor_exposure_data,
                                                                      post_stock_residual_return=stock_residual_return,
                                                                      date=date, all_factors=all_factors,
                                                                      feasible_stocks_list=feasible_stocks_list)

            # ***************************************************组合优化得到最优权重***************************************************
            optimums_results, x, trading_cost = back_test.optimization(
                forecasted_stock_return=prior_return, forecasted_risk=prior_risk,
                date_factor_exposure_data=date_factor_exposure_data, benchmark_weight_data=benchmark_weight_data,
                stock_industry_classification=stock_industry_classification, date=date, industries=industries,
                stocks_list=feasible_stocks_list, current_position=current_position)

            # ***************************************************组合优化得到最优权重***************************************************
            # 记录组合优化结果
            back_test_portfolio_info.loc[date, 'solver_status'] = optimums_results.status
            back_test_portfolio_info.loc[date, 'return'] = pd.DataFrame(prior_return.values.dot(x.value)).iloc[0, 0]
            back_test_portfolio_info.loc[date, 'risk'] = math.sqrt(x.value.T.dot(prior_risk).dot(x.value))
            back_test_stocks_weight.loc[date, feasible_stocks_list] = pd.DataFrame(x.value).iloc[:, 0].apply(
                lambda w: 0 if np.abs(w) < 0.00001 else w).values
            holding_position_info[date] = back_test_stocks_weight.loc[date].replace(0, np.nan).dropna()

            # 记录交易细节
            if j == 0:
                current_portfolio_weight = pd.Series(0, index=feasible_stocks_list)
                rebalance_stocks_list = feasible_stocks_list.copy()
            else:
                current_portfolio_weight = back_test_stocks_weight.loc[
                    back_test_date_list[j - 1], stocks_list_info.loc[back_test_date_list[j - 1], 'feasible']]
                rebalance_stocks_list = list(set(stocks_list_info.loc[back_test_date_list[j - 1], 'feasible']).intersection(
                    set(feasible_stocks_list)))

            target_portfolio_weight = back_test_stocks_weight.loc[date, feasible_stocks_list]
            rebalance_stocks_list.sort()

            weight_change_info = target_portfolio_weight.loc[rebalance_stocks_list] - current_portfolio_weight.loc[rebalance_stocks_list]

            buy_stocks_weight = weight_change_info[weight_change_info > 0].apply(lambda w: np.nan if np.abs(w) < 0.00001 else w).dropna()
            buy_info = pd.DataFrame(index=buy_stocks_weight.index, columns=['former_weight', 'new_weight'])
            buy_info['former_weight'] = current_portfolio_weight.loc[buy_stocks_weight.index]
            buy_info['new_weight'] = target_portfolio_weight.loc[buy_stocks_weight.index]
            buy_info['increment'] = buy_info['new_weight'] - buy_info['former_weight']
            buy_info['commissions'] = buy_info['increment'] * 0.0003
            buy_info['position'] = 'buy'
            buy_info['date'] = date
            buy_info = buy_info.reset_index().rename(columns={'index': 'stock_id'})

            sell_stocks_weight = weight_change_info[weight_change_info < 0].apply(lambda w: np.nan if np.abs(w) < 0.00001 else w).dropna()
            sell_info = pd.DataFrame(index=sell_stocks_weight.index, columns=['former_weight', 'new_weight'])
            sell_info['former_weight'] = current_portfolio_weight.loc[sell_stocks_weight.index]
            sell_info['new_weight'] = target_portfolio_weight.loc[sell_stocks_weight.index]
            sell_info['increment'] = sell_info['new_weight'] - sell_info['former_weight']
            sell_info['commissions'] = sell_info['increment'].abs() * 0.0013
            sell_info['position'] = 'sell'
            sell_info['date'] = date
            sell_info = sell_info.reset_index().rename(columns={'index': 'stock_id'})

            temp_trading_detail = pd.concat([buy_info, sell_info]).reset_index(drop=True)[
                ['date', 'stock_id', 'former_weight', 'new_weight', 'increment', 'commissions']]
            back_test_portfolio_info.loc[date, 'total_cost'] = temp_trading_detail['commissions'].sum()
            trading_detail = pd.concat([trading_detail, temp_trading_detail]).reset_index(drop=True)

            print('******************************************************************')
            print('*\t\tBack Testing -> ' + date)
            print('******************************************************************')

        return back_test_portfolio_info, stocks_list_info, back_test_stocks_weight, holding_position_info, trading_detail


if __name__ == '__main__':

    base_data = BaseData(begin_date='2013-01-01', data_freq='周度')

    # 基础数据:

    base_data.get_base_data(data_name='stock_active_return', active_benchmark='中证800')
    base_data.get_base_data(data_name='stock_status')
    base_data.get_base_data(data_name='industry_classification')
    base_data.get_base_data(data_name='floating_mv')
    base_data.get_base_data(data_name='factor_raw_data')  # 默认取所有的因子数据
    base_data.get_base_data(data_name='benchmark_weight', active_benchmark='中证800')
    base_data.get_base_data(data_name='date_list')
    base_data.get_base_data(data_name='all_stocks_list')

    cs_signal = CrossSectionDataSignal(base_data=base_data)

    factor_number_list = base_data.factor_raw_data['factor_number'].unique().tolist()
    factor_return_dict = {}
    all_regression_result_dict = {}

    for factor_number in factor_number_list:

        temp_data = base_data.factor_raw_data[base_data.factor_raw_data['factor_number'] == factor_number]
        temp_pivot_data = temp_data.pivot_table(values='raw_value', index='get_data_date', columns='stock_id')
        factor_return_dict[factor_number], all_regression_result_dict[factor_number] = \
            cs_signal.regression(factor_data=temp_pivot_data, factor_number=factor_number)
        cs_factor_ic, cs_factor_ic_p_value = cs_signal.ic_time_attenuate(factor_data=temp_pivot_data, max_lag_number=6)

    # 回测
    # bt_factor_list = ['factor3', 'factor6', 'factor9', 'factor12', 'factor15', 'factor18', 'factor21', 'factor24', 'factor27']
    # bt_factor_raw_data = base_data.factor_raw_data[base_data.factor_raw_data['factor_number'].apply(lambda s: s in bt_factor_list)]
    #
    # model = Model()
    # regression_result, factor_exposure, stock_residual = \
    #     model.cross_section_data_regression(left_var_data=base_data.stock_active_return, factor_melt_data=bt_factor_raw_data,
    #                                         industry_classification_melt_data=base_data.industry_classification,
    #                                         floating_mv_pivot_data=base_data.floating_mv)
    #
    # back_test = BackTest(base_data)
    # result = back_test.back_test()
    # 
    # # 尝试构建纯因子策略
    #
    # # 1. 要有个流程把显著的单因子选出来
    #
    # # 1.1 所有单因子检验显著性
    #
    # single_factor_list = list(all_regression_result_dict.keys())
    #
    #
    # all_regression_result_dict['factor1']['p_value']
    #
    #
    # all_regression_result_dict['factor1']['adj_rsquared']
    #
    # # 2. 把单因子进行结合：线性、非线性
    #
    #
    #
    # # 3. 因子收益预测：得到因子预期收益率，得到股票的异质收益率预测，然后得到每个股票的预期收益率
    #
    # # 4. 风险矩阵预测：根据因子预期收益率得到协方差矩阵，根据股票的异质收益率得到异质风险矩阵
    #
    # # 5. 组合优化：得到每期的最优组合
    #
    # # 6. 最优组合的历史净值分析：即策略分析


