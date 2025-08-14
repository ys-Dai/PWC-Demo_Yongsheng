import pandas as pd
import numpy as np
from datetime import datetime
import copy  
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import bisect








########################
def get_OCR(df_sample):
    df_sample_time = df_sample.copy()

    # 转换为datetime，注意单位指定为毫秒（ms）
    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')


    ### 2. Aggregate Sell-Side Liquidity by (Time, SaleOrderID)
    sell_side_df = df_sample_time.groupby(['datetime', 'SaleOrderID']).agg({
        'SaleOrderVolume': 'max'  # Max available sell-side volume
    }).reset_index()

    ### 3. Aggregate Buy-Side Liquidity by (Time, BuyOrderID)
    buy_side_df = df_sample_time.groupby(['datetime', 'BuyOrderID']).agg({
        'BuyOrderVolume': 'max'  # Max available buy-side volume
    }).reset_index()

    ### 4. Aggregate by time
    final_agg_df = df_sample_time.groupby('datetime').agg({'Price': 'max'}).reset_index()

    sell_v_maxsum = sell_side_df.groupby('datetime').agg({'SaleOrderVolume':'sum'}).reset_index()
    buy_v_maxsum = buy_side_df.groupby('datetime').agg({'BuyOrderVolume':'sum'}).reset_index()

    ### 6. Compute Market Microstructure Metrics
    final_agg_df['OCR_B'] = buy_v_maxsum['BuyOrderVolume'].diff().abs() / (buy_v_maxsum['BuyOrderVolume'] + sell_v_maxsum['SaleOrderVolume'])
    # final_agg_df['OCR'] = buy_v_maxsum['BuyOrderVolume'].diff().abs() / (buy_v_maxsum['BuyOrderVolume'])
    # final_agg_df['OCR'] = (buy_v_maxsum['BuyOrderVolume'])


    final_agg_df['OCR_S'] = sell_v_maxsum['SaleOrderVolume'].diff().abs() / (sell_v_maxsum['SaleOrderVolume'] + buy_v_maxsum['BuyOrderVolume'])
    # final_agg_df['Trade_Anomaly_Score'] = sell_v_maxsum['SaleOrderVolume'].diff().abs() / (sell_v_maxsum['SaleOrderVolume'])
    # final_agg_df['Trade_Anomaly_Score'] = (sell_v_maxsum['SaleOrderVolume'])

    # return sell_v_maxsum, buy_v_maxsum
    return final_agg_df


########################
def chunks_time_Trans(df_buy_rolling):
    # compute any kind of aggregation
    buy_volume = df_buy_rolling['Price'].max()  ## 返回聚类操作后，一维的series(对应'price'列) ## 在每组内，统计'price'数值列的最大值
    buy_volume.dropna(inplace=True)  ## 从 buy_volume 中删除所有含有NA值的行。  na的来源可能因为，有些time_freq分组中一行数据都没有(一个B的记录都没)
    #the index contains time info
    return buy_volume.index 

def chunks_time_Trans_price(df_buy_rolling):
    # compute any kind of aggregation
    buy_volume = df_buy_rolling['Price'].max()  ## 返回聚类操作后，一维的series(对应'price'列) ## 在每组内，统计'price'数值列的最大值
    buy_volume.dropna(inplace=True)  ## 从 buy_volume 中删除所有含有NA值的行。  na的来源可能因为，有些time_freq分组中一行数据都没有
    #the index contains time info
    return buy_volume.index, buy_volume.values

def chunks_time_Trans_volume(df_buy_rolling):
    # compute any kind of aggregation
    buy_volume = df_buy_rolling['Volume'].max()  ## 返回聚类操作后，一维的series(对应'price'列) ## 在每组内，统计'price'数值列的最大值
    buy_volume.dropna(inplace=True)  ## 从 buy_volume 中删除所有含有NA值的行。  na的来源可能因为，有些time_freq分组中一行数据都没有
    #the index contains time info
    return buy_volume.index, buy_volume.values


#######################  获取原始 price
def get_original_price(time_freq, df_sample):
    df_sample_time = df_sample.copy()


    # 转换为datetime，注意单位指定为毫秒（ms）
    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')

    ## 将 time 列设置为新的索引（set_index('time')）
    df_sample_time = df_sample_time.reset_index().set_index('datetime')  

    # # 只选取B的交易
    # df_sample_time = df_sample_time[df_sample_time['Type'] == 'B']

    df_buy_grouped = df_sample_time.groupby(pd.Grouper(freq=time_freq)) 

    date, price = chunks_time_Trans_price(df_buy_grouped)  ##### 返回每个freq分段/窗口内（论文中的chunk），‘成交价格(Deal\'s price)’最大的行所对应的datetime

    results_df = pd.DataFrame(
        {'date': date,
         'rush_order':price
         })
    
    return results_df

#######################   获取原始 volume 
def get_original_volume(time_freq, df_sample):
    df_sample_time = df_sample.copy()


    # 转换为datetime，注意单位指定为毫秒（ms）
    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')

    ## 将 time 列设置为新的索引（set_index('time')）
    df_sample_time = df_sample_time.reset_index().set_index('datetime')  

    # # 只选取B的交易
    # df_sample_time = df_sample_time[df_sample_time['Type'] == 'B']

    df_buy_grouped = df_sample_time.groupby(pd.Grouper(freq=time_freq)) 

    date, volume = chunks_time_Trans_volume(df_buy_grouped)  ##### 返回每个freq分段/窗口内（论文中的chunk），‘成交价格(Deal\'s price)’最大的行所对应的datetime

    results_df = pd.DataFrame(
        {'date': date,
         'rush_order':volume
         })
    
    return results_df


########################  对 每时刻的OCR 按照time_freq 进行聚合 
def agg_OCR(time_freq, df_sample, df_OCR, BorS = 'B'):
    ##### aggregate time  
    df_sample_time = df_sample.copy()
    # 转换为datetime，注意单位指定为毫秒（ms）
    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')

    ## 将 time 列设置为新的索引（set_index('time')。这样才可以使用后边的pd.Grouper(freq=time_freq)) 
    df_sample_time = df_sample_time.reset_index().set_index('datetime')  

    df_buy_grouped = df_sample_time.groupby(pd.Grouper(freq=time_freq)) 

    date = chunks_time_Trans(df_buy_grouped)  ##### 返回每个freq分段/窗口内（论文中的chunk），‘成交价格(Deal\'s price)’最大的行所对应的datetime

    print('length of date', len(date))

    results_df_sample = pd.DataFrame(
        {'date': date
        })

    ##### aggregate OCR
    df_OCR_time = df_OCR.copy()

    df_OCR_time = df_OCR_time.reset_index().set_index('datetime')  

    date_OCR = df_OCR_time.groupby(pd.Grouper(freq=time_freq))
    date_OCR = date_OCR['Price'].max().index  # 获取每个时间段内的最大价格对应的时间

    print('length of date_OCR', len(date_OCR))

    if BorS == 'B':
        OCR_sum = df_OCR_time.groupby(pd.Grouper(freq=time_freq))['OCR_B'].sum()
    else:
        OCR_sum = df_OCR_time.groupby(pd.Grouper(freq=time_freq))['OCR_S'].sum()

    print('length of OCR_sum', len(OCR_sum))

    results_df_OCR = pd.DataFrame(
        {'date': date_OCR,
         'OCR_sum': OCR_sum.values
        })
    
    results_df = pd.merge(results_df_sample, results_df_OCR, on='date', how='left')
    
    return results_df, results_df_OCR, results_df_sample

#############################
def normalize(data):
    # 计算最大最小值
    data_min = np.min(data)
    data_max = np.max(data)
    
    # 归一化计算
    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data    


#############################
def rush_order_Trans(df_buy, time_freq):
    df_buy = df_buy.groupby(df_buy.index).count() ## 每组内，按列统计每组的成员数(行数)。
    ## 每列的统计结果是一样的。即，所返回的df_buy，是一个列数不变的df表格。且每一行中各列的值相同
    ## groupby方法将DataFrame中所有行按照一列或多列来划分，分为多个组，列值(这里是index的值)相同的在同一组，列值不同的在不同组。
    df_buy[df_buy == 1] = 0 ## 判断：同一组/一个time(上边的index)中只有一行数据
    df_buy[df_buy > 1] = 1   ##### 判断：同一组/一个time(上边的index)中有多行数据 ———— “一个” rush order!
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['Volume'].sum() ## 这里不一定选['Volume']，随便选一列都可以。因为df_buy一行中所有列都相同。上边将rush order转换为1(df_buy所有列都为1)，这里相当于计算一个time_freq窗口中rush order的数量
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['Volume'].count()
    ## buy_count中的0 是 buy_volume中的0 的子集，前者比后者更过分，表示这个freq区间中1个交易都没有，后者还“可能”表示这个freq区间只有1个交易或没构成任何一次连续交易、一次rush order。所以后边需要将buy_count=0的地方去掉
    
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)  ## 从 buy_volume 中删除那些在相应time_freq时间段内没有买单的记录/这个时段内一行数据都没（即 buy_count 为0的情况）
    buy_volume.dropna(inplace=True)  ## 从 buy_volume 中删除所有含有NA值的行
    
    return buy_volume


## 获取rush order      ##################  考虑 S

def build_features_Trans_S(time_freq, df_sample):
    df_sample_time = df_sample.copy()


    # 转换为datetime，注意单位指定为毫秒（ms）
    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')

    ## 将 time 列设置为新的索引（set_index('time')）
    df_sample_time = df_sample_time.reset_index().set_index('datetime')  

    # 只选取B的交易
    df_sample_time = df_sample_time[(df_sample_time['Type'] == 'S') ]  ## 考虑 S

    # df_sample_time = df_sample_time[(df_sample_time['Type'] == 'B') | 
    #                                 (df_sample_time['Type'] == 'S')]   ## 同时考虑B & S

    df_buy_grouped = df_sample_time.groupby(pd.Grouper(freq=time_freq)) 

    date = chunks_time_Trans(df_buy_grouped)  ##### 返回每个freq分段/窗口内（论文中的chunk），‘成交价格(Deal\'s price)’最大的行所对应的datetime

    results_df = pd.DataFrame(
        {'date': date,
         'rush_order_S':rush_order_Trans(df_sample_time, time_freq).values,
         'rush_order':rush_order_Trans(df_sample_time, time_freq).values
         })
    
    return results_df


## 获取rush order      ##################  考虑 B

def build_features_Trans_B(time_freq, df_sample):
    df_sample_time = df_sample.copy()


    # 转换为datetime，注意单位指定为毫秒（ms）
    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')

    ## 将 time 列设置为新的索引（set_index('time')）
    df_sample_time = df_sample_time.reset_index().set_index('datetime')  

    # 只选取B的交易
    df_sample_time = df_sample_time[(df_sample_time['Type'] == 'B') ]  ## 考虑 B

    # df_sample_time = df_sample_time[(df_sample_time['Type'] == 'B') | 
    #                                 (df_sample_time['Type'] == 'S')]   ## 同时考虑B & S

    df_buy_grouped = df_sample_time.groupby(pd.Grouper(freq=time_freq)) 

    date = chunks_time_Trans(df_buy_grouped)  ##### 返回每个freq分段/窗口内（论文中的chunk），‘成交价格(Deal\'s price)’最大的行所对应的datetime

    results_df = pd.DataFrame(
        {'date': date,
         'rush_order_B':rush_order_Trans(df_sample_time, time_freq).values,
         'rush_order':rush_order_Trans(df_sample_time, time_freq).values
         })
    
    return results_df


## 合并B&S

def merge_B_S(df_rushorder_B, df_rushorder_S):
    # 我们已经有了两个数据框
    # df_rushorder_45_B 和 df_rushorder_45_S

    # 步骤1：去除最后一列'rush_order'
    df_B = df_rushorder_B.drop(columns=['rush_order'])
    df_S = df_rushorder_S.drop(columns=['rush_order'])

    # 步骤2：基于df_B的date合并df_S
    # 将date设置为索引，以便于合并
    df_B_indexed = df_B.set_index('date')
    df_S_indexed = df_S.set_index('date')

    # 使用left join方式合并，这样会保留df_B中的所有日期
    merged_df = df_B_indexed.join(df_S_indexed, how='left')

    # 将缺失值(NaN)填充为0
    merged_df['rush_order_S'] = merged_df['rush_order_S'].fillna(0)

    # 重置索引，将date变回普通列
    merged_df = merged_df.reset_index()

    # 步骤3：创建新列，计算rush_order_B和rush_order_S的和
    merged_df['rush_order'] = merged_df['rush_order_B'] + merged_df['rush_order_S']

    # 查看结果
    # print(merged_df.head())

    return merged_df


## 合并S&B

def merge_S_B(df_rushorder_B, df_rushorder_S):
    # 我们已经有了两个数据框
    # df_rushorder_45_B 和 df_rushorder_45_S

    # 步骤1：去除最后一列'rush_order'
    df_B = df_rushorder_B.drop(columns=['rush_order'])
    df_S = df_rushorder_S.drop(columns=['rush_order'])

    # 步骤2：基于df_B的date合并df_S
    # 将date设置为索引，以便于合并
    df_B_indexed = df_B.set_index('date')
    df_S_indexed = df_S.set_index('date')

    # 使用left join方式合并，这样会保留df_S中的所有日期
    merged_df = df_S_indexed.join(df_B_indexed, how='left')

    # 将缺失值(NaN)填充为0
    merged_df['rush_order_B'] = merged_df['rush_order_B'].fillna(0)

    # 重置索引，将date变回普通列
    merged_df = merged_df.reset_index()

    # 步骤3：创建新列，计算rush_order_B和rush_order_S的和
    merged_df['rush_order'] = merged_df['rush_order_B'] + merged_df['rush_order_S']

    # 查看结果
    # print(merged_df.head())

    return merged_df


#####

def slice_df_bytime(results_df_original, start_time, end_time):
    # 设置开始和结束时间
    start = pd.Timestamp('1900-01-01 ' + start_time)
    end = pd.Timestamp('1900-01-01 ' + end_time)

    # 截取在范围内的表格
    results_df = results_df_original[(results_df_original['date'] >= start) & (results_df_original['date'] <= end)]

    # 重置index
    results_df = results_df.reset_index(drop=True)

    return results_df

#####

def merge_rush_roc_S(df_rush_ori, df_roc_ori, max_roc=1.2, min_roc=0.8, start_time=None , end_time=None):
    df_rush = df_rush_ori.copy()
    df_roc = df_roc_ori.copy()

    if start_time is not None:
        df_rush = slice_df_bytime(df_rush, start_time, end_time)
        df_roc = slice_df_bytime(df_roc, start_time, end_time)


    # 设置max_roc和min_roc作为归一化的边界值
    max_value = df_roc['OCR_sum'].max()
    min_value = df_roc['OCR_sum'].min()
    
    # 步骤1：对df_roc的OCR_sum列进行归一化处理
    df_roc['roc_sum_nor'] = (df_roc['OCR_sum'] - min_value) / (max_value - min_value) * (max_roc - min_roc) + min_roc
    
    # 步骤2：基于df_rush的date合并df_roc
    # 将date设置为索引，以便于合并
    df_rush_indexed = df_rush.set_index('date')
    df_roc_indexed = df_roc.set_index('date')
    
    # 使用left join方式合并，这样会保留df_roc中的所有日期
    merged_df = df_rush_indexed.join(df_roc_indexed, how='left')
    
    # 将缺失值(NaN)填充为整数1
    merged_df['roc_sum_nor'] = merged_df['roc_sum_nor'].fillna(1)
    
    # 重置索引，将date变回普通列
    merged_df = merged_df.reset_index()
    
    # 步骤3：计算rush_order_S和roc_sum_nor的乘积，并保存在新的列中
    merged_df['rush_order_S_new'] = merged_df['rush_order_S'] * merged_df['roc_sum_nor']
    
    # # 更改原始rush_order_S这一列的数据的列名为rush_order_S_original
    # merged_df.rename(columns={'rush_order_S': 'rush_order_S_new', 'rush_order_S_original': 'rush_order_S'}, inplace=True)
    
    # 重命名新列rush_order_S_new为rush_order_S，并将原始rush_order_S重命名为rush_order_S_original
    # merged_df.rename(columns={'rush_order_S_new': 'rush_order_S', 'rush_order_S': 'rush_order_S_original'}, inplace=True)
    
    new_S_df = pd.DataFrame({'date': merged_df['date'], 
                             'rush_order_S': merged_df['rush_order_S_new'], 
                             'rush_order': merged_df['rush_order_S_new']})
                        

    return new_S_df, merged_df

#####

def merge_rush_roc_B(df_rush_ori, df_roc_ori, max_roc=1.2, min_roc=0.8, start_time=None , end_time=None):
    df_rush = df_rush_ori.copy()
    df_roc = df_roc_ori.copy()

    if start_time is not None:
        df_rush = slice_df_bytime(df_rush, start_time, end_time)
        df_roc = slice_df_bytime(df_roc, start_time, end_time)

    # 设置max_roc和min_roc作为归一化的边界值
    max_value = df_roc['OCR_sum'].max()
    min_value = df_roc['OCR_sum'].min()
    
    # 步骤1：对df_roc的OCR_sum列进行归一化处理
    df_roc['roc_sum_nor'] = (df_roc['OCR_sum'] - min_value) / (max_value - min_value) * (max_roc - min_roc) + min_roc
    
    # 步骤2：基于df_rush的date合并df_roc
    # 将date设置为索引，以便于合并
    df_rush_indexed = df_rush.set_index('date')
    df_roc_indexed = df_roc.set_index('date')
    
    # 使用left join方式合并，这样会保留df_roc中的所有日期
    merged_df = df_rush_indexed.join(df_roc_indexed, how='left')
    
    # 将缺失值(NaN)填充为整数1
    merged_df['roc_sum_nor'] = merged_df['roc_sum_nor'].fillna(1)
    
    # 重置索引，将date变回普通列
    merged_df = merged_df.reset_index()
    
    # 步骤3：计算rush_order_S和roc_sum_nor的乘积，并保存在新的列中
    merged_df['rush_order_B_new'] = merged_df['rush_order_B'] * merged_df['roc_sum_nor']
    
    # # 更改原始rush_order_S这一列的数据的列名为rush_order_S_original
    # merged_df.rename(columns={'rush_order_S': 'rush_order_S_new', 'rush_order_S_original': 'rush_order_S'}, inplace=True)
    
    # 重命名新列rush_order_S_new为rush_order_S，并将原始rush_order_S重命名为rush_order_S_original
    # merged_df.rename(columns={'rush_order_B_new': 'rush_order_B', 'rush_order_B': 'rush_order_B_original'}, inplace=True)
    
    new_B_df = pd.DataFrame({'date': merged_df['date'], 
                             'rush_order_B': merged_df['rush_order_B_new'], 
                             'rush_order': merged_df['rush_order_B_new']})


    return new_B_df, merged_df









##### show original price and features -- 01-06
def v_rushTrans_sub_shadow_outputtest_plotly(results_df_original, shadow=None, shadow_color='orange', shadow_alpha=0.3, 
                                            start_time=None, end_time=None, save_fig=False, save_path=None, 
                                            fig_name=None, y_max=None, norm_y=False, width=1000, height=500, 
                                            line_color='#696969', line_name='Price', y_column='rush_order'):
    """
    使用Plotly绘制交互式图表的版本
    
    参数:
    - width: 图表宽度 (默认1000)
    - height: 图表高度 (默认500)
    - 其他参数与原函数相同
    """
    
    # 设置开始和结束时间
    start = pd.Timestamp('1900-01-01 ' + start_time)
    end = pd.Timestamp('1900-01-01 ' + end_time)

    # 截取在范围内的表格
    results_df = results_df_original[(results_df_original['date'] >= start) & (results_df_original['date'] <= end)]

    # 重置index
    results_df = results_df.reset_index(drop=True)

    if y_column == 'rush_order':
        # 生成y数据
        if norm_y:
            y2 = normalize(results_df['rush_order'])
        else:
            y2 = results_df['rush_order']

        y = results_df['rush_order']
    else:
        if norm_y:
            y2 = normalize(results_df[y_column])
        else:
            y2 = results_df[y_column]

        y = results_df[y_column]
        

    # 处理时间标签
    time_labels = results_df['date']
    new_time = time_labels.dt.time  # 只保留时间
    new_time = new_time.astype(str).tolist()  # 转换为字符串列表

    print('len(y): ', len(y))

    # 创建Plotly图表
    fig = go.Figure()

    # 添加主要的折线图
    fig.add_trace(go.Scatter(
        x=new_time,
        y=y2,
        mode='lines',  # 'lines+markers'
        name=line_name,
        line=dict(width=3, color=line_color),
        # marker=dict(size=4),
        hovertemplate='<b>Time</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>',
        
    ))

    # 处理阴影区域和异常标注
    vector = copy.deepcopy(y2)
    ano_sample = []

    if shadow is not None:
        for i, shadow_range in enumerate(shadow):
            start_shadow = shadow_range[0]
            end_shadow = shadow_range[1]

            # 找到对应的索引
            shadow_start = bisect.bisect_right(new_time, start_shadow) - 1
            shadow_end = bisect.bisect_left(new_time, end_shadow)
            
            # 确保索引在有效范围内
            shadow_start = max(0, shadow_start)
            shadow_end = min(len(new_time) - 1, shadow_end)

            # 添加垂直阴影区域
            fig.add_vrect(
                x0=new_time[shadow_start],
                x1=new_time[shadow_end],
                fillcolor=shadow_color,
                opacity=shadow_alpha,
                layer="below",
                line_width=0,
                # annotation_text=f"Manipulation Interval {i+1}",
                # annotation_position="top left",
                # annotation=dict(
                #     text=f"Manipulation Interval {i+1}",
                #     font=dict(size=14, color='red'),  # 设置字体大小和颜色
                #     # x=new_time[shadow_start],
                #     # y=max(vector),
                #     # xref="x",
                #     # yref="y",
                #     showarrow=False,
                #     xanchor="left",
                #     yanchor="top"
                # ),
                name = f'Manipulation Interval'
            )

            # 添加一个不可见的trace用于图例显示（只在第一个阴影区域时添加）, 显示阴影图例
            if i == 0:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color=shadow_color, opacity=shadow_alpha, size=15, symbol='square'),
                    name='Manipulation Interval',
                    showlegend=True
                ))

            ano = np.array([shadow_start, shadow_end])
            ano_sample.append(ano)

            # 添加标注点
            shadow_points_x = [new_time[shadow_start], new_time[shadow_end]]
            shadow_points_y = [vector[shadow_start], vector[shadow_end]]
            
            fig.add_trace(go.Scatter(
                x=shadow_points_x,
                y=shadow_points_y,
                mode='markers+text',
                marker=dict(color='red', size=8, symbol='diamond'),
                text=[f'({vector[shadow_start]:.2f}, {new_time[shadow_start]})', 
                      f'({vector[shadow_end]:.2f}, {new_time[shadow_end]})'],
                textposition='top center',
                textfont=dict(color='red', size=11),
                name=f'Manipulation Boundary',
                # hovertemplate='<b>异常点</b><br><b>时间</b>: %{x}<br><b>值</b>: %{y:.2f}<extra></extra>'
            ))

    # 设置图表布局
    fig.update_layout(

        plot_bgcolor='white',      # 绘图区域背景色
        # paper_bgcolor='white',     # 整个图片背景色

        title=dict(
            text=fig_name if fig_name else 'Rush Order 时间序列图',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Time',
            tickangle=45,
            # 设置显示的刻度（每20个点显示一个）
            tickmode='array',
            tickvals=new_time[::20],
            ticktext=new_time[::20],
            # 新增网格配置
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            griddash='dot',
            # 新增边界框配置
            showline=True,
            linewidth=1.2,
            linecolor='black',
            mirror=True,  # mirror=True 确保在图表的对面也显示边界线，形成完整的边界框
            autorange=True,  # 新增：X轴自动缩放
        ),
        yaxis=dict(
            # title='Rush Order 数量',
            range=[min(y2)-0.1, y_max if y_max is not None and not norm_y else max(y2)+0.1],
            tickfont=dict(size=14),  # 新增字体大小控制
            # 新增网格配置
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            griddash='dot',
            # 新增边界框配置
            showline=True,
            linewidth=1.2,
            linecolor='black',
            mirror=True,  # mirror=True 确保在图表的对面也显示边界线，形成完整的边界框
            autorange=True,  # 新增：Y轴自动缩放
        ),
        width=width,
        height=height,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12.)  # 新增字体大小控制
        ),
        margin=dict(l=50, r=50, t=80, b=100)
    )

    # 添加工具栏配置
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
        # 'autosizable': True,  # 新增：启用自动调整大小
        # 'responsive': True,   # 新增：响应式布局
        'toImageButtonOptions': {
            'format': 'png',
            'filename': fig_name if fig_name else 'rush_order_chart',
            'height': height,
            'width': width,
            'scale': 1
        }
    }

    # 保存图表
    if save_fig and save_path and fig_name:
        fig.write_html(f"{save_path}{fig_name}.html")
        # fig.write_image(f"{save_path}{fig_name}.png", width=width, height=height)

    # 显示图表
    # fig.show(config=config)

    return y, ano_sample


##### show detect results -- 07 08
from sklearn.preprocessing import MinMaxScaler
def visualize_anomaly_scores_in_original_series_plotly(results_df_original, result, shadow=None, start_time=None, end_time=None, 
                                                      save_fig=False, save_path=None, fig_name=None, show_text=False,
                                                      width=1000, height=500, shadow_color='orange', shadow_alpha=0.3,
                                                      line_color='#696969', line_name='Price'):
    """
    在原始完整序列中可视化异常分数 - Plotly版本
    
    参数:
    results_df_original: 原始数据DataFrame
    result: 异常分数，长度为160的一维数组，值在0-1之间
    shadow: 原始异常区间，格式为[('开始时间','结束时间'),...]
    start_time: 子序列开始时间
    end_time: 子序列结束时间
    save_fig: 是否保存图像
    save_path: 图像保存路径
    fig_name: 图像名称
    show_text: 是否显示说明文本
    width: 图表宽度 (默认1000)
    height: 图表高度 (默认500)
    shadow_color: 阴影颜色 (默认'orange')
    shadow_alpha: 阴影透明度 (默认0.3)
    """
    # 设置子序列的开始和结束时间
    sub_start = pd.Timestamp('1900-01-01 ' + start_time)
    sub_end = pd.Timestamp('1900-01-01 ' + end_time)

    # 截取子序列范围内的数据，用于确定索引映射
    sub_results_df = results_df_original[(results_df_original['date'] >= sub_start) & 
                                        (results_df_original['date'] <= sub_end)]
    sub_results_df = sub_results_df.reset_index()
    
    # 确保子序列长度足够
    if len(sub_results_df) < 160:
        print(f"警告: 子序列长度 {len(sub_results_df)} 小于期望的160")
        return
    
    # 获取原始序列中对应子序列前160个点的索引
    original_indices = sub_results_df.iloc[:160]['index'].tolist()
    
    # 处理时间标签
    time_labels = results_df_original['date']
    new_time = time_labels.dt.time  # 只保留时间部分
    new_time = new_time.astype(str).tolist()  # 转换为字符串列表
    
    # 生成原始数据
    y = results_df_original['rush_order'].values
    
    # 归一化原始数据
    scaler = MinMaxScaler()
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 创建异常分数数组，映射到原始序列
    anomaly_scores = np.zeros(len(results_df_original))
    for i in range(len(result)):
        if i < len(original_indices):  # 确保索引在范围内
            anomaly_scores[original_indices[i]] = result[i]
    
    # 创建Plotly图表
    fig = go.Figure()

    # 绘制归一化数据
    fig.add_trace(go.Scatter(
        x=list(range(len(y_normalized))),
        y=y_normalized,
        mode='lines',
        name=line_name,
        line=dict(width=3, color=line_color),
        hovertemplate='<b>Time</b>: %{x}<br><b>Price-Norm</b>: %{y:.2f}<extra></extra>'
    ))

    # 绘制异常分数
    fig.add_trace(go.Scatter(
        x=list(range(len(anomaly_scores))),
        y=anomaly_scores,
        mode='lines',
        name='Detection Anomaly Score',
        line=dict(width=2, color='red'),
        hovertemplate='<b>Time</b>: %{x}<br><b>Anomaly Score</b>: %{y:.2f}<extra></extra>'
    ))
    
    # 设置x轴标签
    total_points = len(results_df_original)
    xticks_count = min(10, total_points)  # 最多显示10个刻度
    xticks_pos = np.linspace(0, total_points-1, xticks_count, dtype=int)
    xticks_labels = [new_time[i] for i in xticks_pos]
    
    # 标记原始标注的异常区间
    if shadow is not None:
        for i in range(len(shadow)):
            start_shadow = shadow[i][0]
            end_shadow = shadow[i][1]
            
            # 找到刚好小于等于等于start的行的索引
            shadow_start = bisect.bisect_right(new_time, start_shadow)
            shadow_start_idx = shadow_start - 1

            # 找到刚好大于等于end2的行的索引
            shadow_end_idx = bisect.bisect_left(new_time, end_shadow)

            # 添加垂直阴影区域
            fig.add_vrect(
                x0=shadow_start_idx,
                x1=shadow_end_idx,
                fillcolor=shadow_color,
                opacity=shadow_alpha,
                layer="below",
                line_width=0
            )

            # 添加图例显示（只在第一个阴影区域时添加）
            if i == 0:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color=shadow_color, opacity=shadow_alpha, size=15, symbol='square'),
                    name='Ground Truth',
                    showlegend=True
                ))

    # 设置图表布局
    fig.update_layout(
        plot_bgcolor='white',
        title=dict(
            text=fig_name if fig_name else 'Detection Output',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Time',
            tickmode='array',
            tickvals=xticks_pos,
            ticktext=xticks_labels,
            tickangle=45,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            griddash='dot',
            showline=True,
            linewidth=1.2,
            linecolor='black',
            mirror=True,
            autorange=True
        ),
        yaxis=dict(
            # title='Price / Anomaly Score',
            range=[-0.05, 1.05],
            tickfont=dict(size=14),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            griddash='dot',
            showline=True,
            linewidth=1.2,
            linecolor='black',
            mirror=True,
            autorange=True
        ),
        width=width,
        height=height,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12.9)
        ),
        margin=dict(l=50, r=50, t=80, b=100)
    )

    # 配置工具栏
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': fig_name + '_original_scores' if fig_name else 'anomaly_detection',
            'height': height,
            'width': width,
            'scale': 1
        }
    }

    # 保存和显示
    if save_fig and save_path and fig_name:
        fig.write_html(f"{save_path}{fig_name}_original_scores.html")

    # fig.show(config=config)

##### show detect results -- 09
def visualize_anomalies_in_original_series_fewershallow_plotly(results_df_original, result, shadow=None, start_time=None, end_time=None, 
                                                              save_fig=False, save_path=None, fig_name=None, y_max=None,
                                                              width=1000, height=500, shadow_color='orange', shadow_alpha=0.3,
                                                              detection_color='red', detection_alpha=0.4,
                                                              line_color='#696969', line_name='Price'):
    """
    在原始完整序列中可视化异常检测结果 - Plotly版本
    
    参数:
    results_df_original: 原始数据DataFrame
    result: 异常检测结果，长度为160的一维数组，1表示异常点
    shadow: 原始异常区间，格式为[('开始时间','结束时间'),...]
    start_time: 子序列开始时间
    end_time: 子序列结束时间
    save_fig: 是否保存图像
    save_path: 图像保存路径
    fig_name: 图像名称
    y_max: y轴最大值
    width: 图表宽度 (默认1000)
    height: 图表高度 (默认500)
    shadow_color: Ground Truth阴影颜色 (默认'orange')
    shadow_alpha: Ground Truth阴影透明度 (默认0.3)
    detection_color: 检测结果阴影颜色 (默认'red')
    detection_alpha: 检测结果阴影透明度 (默认0.4)
    """
    # 设置子序列的开始和结束时间
    sub_start = pd.Timestamp('1900-01-01 ' + start_time)
    sub_end = pd.Timestamp('1900-01-01 ' + end_time)

    # 截取子序列范围内的数据，用于确定索引映射
    sub_results_df = results_df_original[(results_df_original['date'] >= sub_start) & 
                                        (results_df_original['date'] <= sub_end)]
    sub_results_df = sub_results_df.reset_index()
    
    # 确保子序列长度足够
    if len(sub_results_df) < 160:
        print(f"警告: 子序列长度 {len(sub_results_df)} 小于期望的160")
        return
    
    # 获取原始序列中对应子序列前160个点的索引
    original_indices = sub_results_df.iloc[:160]['index'].tolist()

    # 处理时间标签
    time_labels = results_df_original['date']
    new_time = time_labels.dt.time  # 只保留时间部分
    new_time = new_time.astype(str).tolist()  # 转换为字符串列表

    # 绘制原始数据
    y = results_df_original['rush_order']
    
    # 创建Plotly图表
    fig = go.Figure()

    # 添加主要的价格线
    fig.add_trace(go.Scatter(
        x=list(range(len(y))),
        y=y,
        mode='lines',
        name=line_name,
        line=dict(width=3, color=line_color),
        hovertemplate='<b>Time</b>: %{x}<br><b>Price</b>: %{y:.2f}<extra></extra>'
    ))
    
    # 标记机器学习模型检测到的异常区间（映射到原始序列）
    anomaly_regions = []
    start_idx = None
    
    # 创建一个长度与原始序列相同的数组，标记异常点
    anomaly_markers = np.zeros(len(results_df_original))
    
    for i in range(len(result)):
        if i < len(original_indices):  # 确保索引在范围内
            if result[i] == 1:
                anomaly_markers[original_indices[i]] = 1
    
    # 从标记数组中提取连续的异常区间
    for i in range(len(anomaly_markers)):
        if anomaly_markers[i] == 1 and (i == 0 or anomaly_markers[i-1] == 0):
            start_idx = i
        elif anomaly_markers[i] == 0 and i > 0 and anomaly_markers[i-1] == 1:
            anomaly_regions.append((start_idx, i - 1))
            start_idx = None
    
    # 处理最后一个异常区间
    if start_idx is not None and anomaly_markers[-1] == 1:
        anomaly_regions.append((start_idx, len(anomaly_markers) - 1))
    
    # 计算检测结果阴影的y轴范围
    y_range = max(y) - min(y)
    y_middle = (max(y) + min(y)) / 2
    ml_bottom = y_middle - y_range * 0.1
    ml_top = y_middle + y_range * 0.1

    # 在图上标记机器学习检测到的异常区域（红色阴影）
    num_detect_ano = 0  # 检测到的异常数量
    for start_idx, end_idx in anomaly_regions:
        # 为每个检测区域添加填充区域
        x_range = list(range(start_idx, end_idx + 1))
        y_bottom = [ml_bottom] * len(x_range)
        y_top = [ml_top] * len(x_range)
        
        fig.add_trace(go.Scatter(
            x=x_range + x_range[::-1],  # x坐标正向和反向
            y=y_bottom + y_top[::-1],   # y坐标下边界和上边界
            fill='toself',
            fillcolor=f'rgba(255, 0, 0, {detection_alpha})',  # 红色，带透明度
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Detection Results' if num_detect_ano == 0 else '',
            showlegend=True if num_detect_ano == 0 else False,
            hoverinfo='skip'
        ))
        num_detect_ano += 1
    
    # 标记原始标注的异常区间
    if shadow is not None:
        for i in range(len(shadow)):
            start_shadow = shadow[i][0]
            end_shadow = shadow[i][1]
            
            # 找到对应的索引
            shadow_start = bisect.bisect_right(new_time, start_shadow)
            shadow_start_idx = shadow_start - 1
            shadow_end_idx = bisect.bisect_left(new_time, end_shadow)

            # 添加垂直阴影区域
            fig.add_vrect(
                x0=shadow_start_idx,
                x1=shadow_end_idx,
                fillcolor=shadow_color,
                opacity=shadow_alpha,
                layer="below",
                line_width=0
            )

            # 添加图例显示（只在第一个阴影区域时添加）
            if i == 0:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color=shadow_color, opacity=shadow_alpha, size=15, symbol='square'),
                    name='Ground Truth',
                    showlegend=True
                ))

            # 添加标注点
            # shadow_points_x = [shadow_start_idx, shadow_end_idx]
            # shadow_points_y = [y.iloc[shadow_start_idx], y.iloc[shadow_end_idx]]
            # shadow_time = [new_time[shadow_start_idx], new_time[shadow_end_idx]]
            
            # fig.add_trace(go.Scatter(
            #     x=shadow_points_x,
            #     y=shadow_points_y,
            #     mode='markers+text',
            #     marker=dict(color='blue', size=8, symbol='diamond'),
            #     text=[f'({shadow_points_y[0]:.2f}, {shadow_time[0]})', 
            #           f'({shadow_points_y[1]:.2f}, {shadow_time[1]})'],
            #     textposition='top center',
            #     textfont=dict(color='blue', size=10),
            #     name=f'Ground Truth Boundary {i+1}' if i == 0 else '',
            #     showlegend=True if i == 0 else False,
            #     hovertemplate='<b>Ground Truth Point</b><br><b>Time</b>: %{text}<br><b>Value</b>: %{y:.2f}<extra></extra>'
            # ))
    
    # 设置x轴标签
    total_points = len(results_df_original)
    xticks_count = min(10, total_points)  # 最多显示10个刻度
    xticks_pos = np.linspace(0, total_points-1, xticks_count, dtype=int)
    xticks_labels = [new_time[i] for i in xticks_pos]
    
    # 设置y轴范围
    if y_max is not None:
        y_range_setting = [0, y_max]
    else:
        y_min = min(y)
        y_max_calc = max(y)
        y_range_setting = [y_min, y_max_calc]

    # 设置图表布局
    fig.update_layout(
        plot_bgcolor='white',
        title=dict(
            text=fig_name if fig_name else '完整序列中的异常检测结果可视化',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Time',
            tickmode='array',
            tickvals=xticks_pos,
            ticktext=xticks_labels,
            tickangle=45,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            griddash='dot',
            showline=True,
            linewidth=1.2,
            linecolor='black',
            mirror=True,
            autorange=True
        ),
        yaxis=dict(
            # title='Price',
            range=y_range_setting,
            tickfont=dict(size=14),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            griddash='dot',
            showline=True,
            linewidth=1.2,
            linecolor='black',
            mirror=True,
            autorange=True
        ),
        width=width,
        height=height,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12.9)
        ),
        margin=dict(l=50, r=50, t=80, b=100)
    )

    # 配置工具栏
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': fig_name + '_original' if fig_name else 'anomaly_detection_original',
            'height': height,
            'width': width,
            'scale': 1
        }
    }

    # 保存和显示
    if save_fig and save_path and fig_name:
        fig.write_html(f"{save_path}{fig_name}_original.html")

    # fig.show(config=config)
















