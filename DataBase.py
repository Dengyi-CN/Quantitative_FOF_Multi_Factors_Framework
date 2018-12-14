import cx_Oracle
import pandas as pd
import numpy as np
import sys
import math
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK'
os.environ['NLS_CHARACTERSET'] = 'ZHS16GBK'
os.environ['NLS_NCHAR_CHARACTERSET'] = 'AL16UTF16'
import pickle
# ----------------------------------------------------------函数（开始）-------------------------------------------------------------------------------
low_level_divided_str = '{0: >{width}}'.format('', width=4) + '{0:~>{width}}'.format('', width=92) + '{0: >{width}}'.format('', width=4)


def print_seq_line(action_str):
    def decorate(func):
        def wrapper(*args, **kwargs):
            print('\n' + '{0:*>{width}}'.format(action_str, width=50) + '{0:*>{width}}'.format('', width=50 - len(action_str)) + '\n')
            result = func(*args, **kwargs)
            print('\n' + '{0:*>{width}}'.format(action_str, width=50) + '{0:*>{width}}'.format('', width=50 - len(action_str)) + '\n')
            return result
        return wrapper
    return decorate


def logging():
    def decorate(func):
        def wrapper(*args, **kwargs):
            print('\tlogging')
            return func(*args, **kwargs)
        return wrapper
    return decorate


def connect_oracle_db(account='lyzs_tinysoft', passport='lyzs@2018'):
    print('\t正在连接数据库…')
    # dsn = cx_Oracle.makedsn('10.1.1.10', '1521', 'ly_orcl')
    connection = cx_Oracle.connect(account, passport, '10.1.1.10:1521/orclly')
    print('\t数据库连接成功')
    print(low_level_divided_str)
    return connection


@print_seq_line('写入数据')
def insert_data_to_oracle_db(data=None, table_name=None, account='lyzs_tinysoft', passport='lyzs@2018'):
    try:
        connection = connect_oracle_db(account, passport)
    except BaseException:
        raise BaseException('数据库连接出错。')

    print('\t开始写入数据表' + table_name + '…')
    cursor = connection.cursor()

    # 生成对应类型的格式化输出
    format_string = ','.join(data.iloc[0, :].apply(lambda s: '\'%s\'' if isinstance(s, str) else(
        '%d' if np.issubdtype(s, np.integer) else '%f')).tolist())
    insert_sql = 'insert into ' + table_name + ' (' + ','.join(data.columns.tolist()) + ') values (' + format_string + ')'

    for i in range(data.shape[0]):
        cursor.execute(insert_sql % tuple(data.iloc[i, :][data.columns].apply(lambda s: s.replace('\'', '\'\'') if isinstance(s, str) else s).values))
        # if i % (data.shape[0] // 50) == 0:
        percent = '{:.2%}'.format(i / data.shape[0])
        sys.stdout.write('\r')
        sys.stdout.write("\t数据写入完成进度：[%-50s] %s" % ('#' * int(math.floor(i * 50 / data.shape[0])), percent))
        sys.stdout.flush()

    connection.commit()
    # 关闭游标
    cursor.close()
    connection.close()
    print('\n\t数据提交成功，已关闭数据库连接')
    print(low_level_divided_str)


@print_seq_line('读取数据')
@logging()
def read_data_from_oracle_db(sql=None, account='lyzs_tinysoft', passport='lyzs@2018'):

    try:
        connection = connect_oracle_db(account, passport)
    except BaseException:
        raise BaseException('数据库连接出错。')

    print('\t开始读取数据…')
    print('\t传入的sql语句为:\n')
    for line in sql.splitlines():
        print('\t\t> ' + line)
    print()
    cursor = connection.cursor()
    cursor.execute(sql)
    exec_result = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    print('\t读取数据成功，共' + 'x'.join([str(i) for i in exec_result.shape]) + '条')
    return exec_result


# ----------------------------------------------------------函数（结束）-------------------------------------------------------------------------------
params_data_url = eval(input('请输入需要写入数据库的数据存放文件夹地址：'))
account = eval(input('请输入连接数据库的账号：'))
passport = eval(input('请输入连接数据库的密码：'))

raw_data = pickle.load(open(params_data_url + '/raw_data.dat', 'rb'))
factor_library = pd.read_excel(params_data_url + '/因子列表-初步检测.xlsx')
factor_list = factor_library['factor_number'].tolist()

# # 1. stock_info_data
# stock_info_data = raw_data[['数据提取日', '财务数据最新报告期', 'stockid', 'stockname', 'sectorid', 'sectorname', '上市天数', '沪深300成分股',
#           '中证500成分股', '中证800成分股', '申万A股成分股', '是否st', '是否pt', '是否停牌']].copy()
# stock_info_data.columns = pd.read_excel(params_data_url + '/量化FOF研究-数据库表设计.xlsx', sheet_name='Stock_Info_Data')['字段英文名'].tolist()
#
# insert_data_to_oracle_db(data=stock_info_data, table_name='lyzs_tinysoft.stock_info_data', account=account, passport=passport)

# 2. factor_raw_data
factor_raw_data = raw_data[['数据提取日', 'stockid'] + factor_list].rename(
    columns={'数据提取日': 'get_data_date', 'stockid': 'stock_id'}).melt(
    id_vars=['get_data_date', 'stock_id'], var_name=['factor_number'], value_name='factor_raw_value')

insert_data_to_oracle_db(data=factor_raw_data, table_name='lyzs_tinysoft.factor_raw_data', account=account, passport=passport)

# 3. return_data
return_dict = {'持仓天数': 'holding_period_days', '持仓期停牌天数占比': 'hp_suspension_days_pct', '持仓期收益率': 'holding_period_return',
               '申万行业收益率': 'sw_1st_sector_hpr', '沪深300收益率': 'hs300_hpr', '中证500收益率': 'zz500_hpr', '中证800收益率': 'zz800_hpr',
               '上证综指收益率': 'szzz_hpr', '申万A股收益率': 'swag_hpr'}
return_data = raw_data[['数据提取日', 'stockid'] + list(return_dict.keys())].rename(columns=return_dict)
insert_data_to_oracle_db(data=return_data, table_name='lyzs_tinysoft.return_data', account=account, passport=passport)

# 4. factor_stratificated_return

factor_stratificated_return = pickle.load(open(params_data_url + '/factor_stratificated_return.dat', 'rb'))
quantile_name_dict = {'low': '第1档收益率', **{str(i): '第' + str(i) + '档收益率' for i in range(2,10)}, 'high': '第10档收益率',
                      '数据提取日': 'get_data_date'}
factor_stratificated_return = factor_stratificated_return.rename(columns=quantile_name_dict).melt(
    id_vars=['factor_number', 'get_data_date', 'sample_scope'], var_name=['type_name'], value_name='value')
insert_data_to_oracle_db(data=factor_stratificated_return, table_name='lyzs_tinysoft.factor_return', account=account, passport=passport)

# 5. factor_return_regression

factor_return_regression = pickle.load(open(params_data_url + '/factor_return_regression.dat', 'rb'))
insert_data_to_oracle_db(data=factor_return_regression, table_name='lyzs_tinysoft.factor_return_regression', account=account, passport=passport)
