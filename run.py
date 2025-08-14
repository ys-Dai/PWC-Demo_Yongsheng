import numpy as np
import pandas as pd

import config as config
from get_features import *
import os

import argparse

own_scores_dict = np.load(r'detect_results/pre_scores.npy')

def main(data_name):
    df_sample_2807 = pd.read_csv(config.Data_path[data_name])

    folder_path = config.Save_path[data_name]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df_price = get_original_price(time_freq=config.Time_freq[data_name], df_sample=df_sample_2807)
    y, ano_sample = v_rushTrans_sub_shadow_outputtest_plotly(
    results_df_original=df_price,
    shadow=config.Shadow[data_name],
    start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name],
    fig_name='01_original_price',
    width=720 * 0.82,  ## 720
    height=460 * 0.86,  ## 450
    save_fig=True,
    save_path=config.Save_path[data_name],
    line_color='#696969')

    df_rushorder_45s = build_features_Trans_S(time_freq=config.Time_freq[data_name], df_sample=df_sample_2807)
    y, ano_sample = v_rushTrans_sub_shadow_outputtest_plotly(
    results_df_original=df_rushorder_45s,
    shadow=config.Shadow[data_name],
    start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name],
    y_max=90,
    fig_name='02_feature_Rushorder_S',
    width=720 * 0.82,  ## 720
    height=460 * 0.86,  ## 450
    save_fig=True,
    save_path=config.Save_path[data_name],
    line_color='#ff964f',
    line_name='Rushorder_S',)

    df_rushorder_45b = build_features_Trans_B(time_freq=config.Time_freq[data_name], df_sample=df_sample_2807)
    y, ano_sample = v_rushTrans_sub_shadow_outputtest_plotly(
    results_df_original=df_rushorder_45b,
    shadow=config.Shadow[data_name],
    start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name],
    y_max=90,
    fig_name='03_feature_Rushorder_B',
    width=720 * 0.82,  ## 720
    height=460 * 0.86,  ## 450
    save_fig=True,
    save_path=config.Save_path[data_name],
    line_color='#d5b60a',
    line_name='Rushorder_B',)

    OCR_df = get_OCR(df_sample_2807)
    OCR_agg_S_df, _, _ = agg_OCR(time_freq='60S', df_sample=df_sample_2807, df_OCR=OCR_df, BorS = 'S')
    y, ano_sample = v_rushTrans_sub_shadow_outputtest_plotly(
    results_df_original=OCR_agg_S_df,
    shadow=config.Shadow[data_name],
    start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name],
    y_max=90,
    fig_name='04_feature_OCR_S',
    width=720 * 0.82,  ## 720
    height=460 * 0.86,  ## 450
    save_fig=True,
    save_path=config.Save_path[data_name],
    line_color='#8f99fb',
    line_name='OCR_S',
    y_column='OCR_sum')

    OCR_agg_B_df, _, _ = agg_OCR(time_freq='60S', df_sample=df_sample_2807, df_OCR=OCR_df, BorS = 'B')
    y, ano_sample = v_rushTrans_sub_shadow_outputtest_plotly(
    results_df_original=OCR_agg_B_df,
    shadow=config.Shadow[data_name],
    start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name],
    y_max=90,
    fig_name='05_feature_OCR_B',
    width=720 * 0.82,  ## 720
    height=460 * 0.86,  ## 450
    save_fig=True,
    save_path=config.Save_path[data_name],
    line_color='#6fc276',
    line_name='OCR_B',
    y_column='OCR_sum')


    new_s_df, merge_s_df=merge_rush_roc_S(df_rushorder_45s, OCR_agg_S_df, max_roc=1.4, min_roc=0.9, start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name])
    new_b_df, merge_b_df=merge_rush_roc_B(df_rushorder_45b, OCR_agg_B_df, max_roc=1.4, min_roc=0.9, start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name])
    df_B_S = merge_B_S(new_b_df, new_s_df)
    df_S_B = merge_S_B(new_b_df, new_s_df)
    if config.IF_BS[data_name]:
        Fea_all = df_B_S
    else:
        Fea_all = df_S_B

    y, ano_sample = v_rushTrans_sub_shadow_outputtest_plotly(
    results_df_original=Fea_all,
    shadow=config.Shadow[data_name],
    start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name],
    y_max=90,
    fig_name='06_feature_all',
    width=720 * 0.82,  ## 720
    height=460 * 0.86,  ## 450
    save_fig=True,
    save_path=config.Save_path[data_name],
    line_color='#d6b4fc',
    line_name='Features_all',
    # y_column='OCR_sum'
    )


    result_scores = own_scores_dict[config.Result_Score_id[data_name]]
    visualize_anomaly_scores_in_original_series_plotly(
    results_df_original=Fea_all,
    result=result_scores,  # 长度为160的异常分数数组
    shadow=config.Shadow[data_name],
    start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name],
    save_fig=True,
    save_path=config.Save_path[data_name],
    fig_name='07_detect_results on Features',
    show_text=True,
    width=720 * 0.82,  ## 720
    height=460 * 0.86,  ## 450
    line_color='#d6b4fc',
    line_name='Features_all',
    )

    df_price_slice = slice_df_bytime(df_price, start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name])
    visualize_anomaly_scores_in_original_series_plotly(
    results_df_original=df_price_slice,
    result=result_scores,  # 长度为160的异常分数数组
    shadow=config.Shadow[data_name],
    start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name],
    save_fig=True,
    save_path=config.Save_path[data_name],
    fig_name='08_detect_results on Price',
    show_text=True,
    width=720 * 0.82,  ## 720
    height=460 * 0.86,  ## 450
    # line_color='#d6b4fc',
    line_name='Price-Norm',
    )

    binary_result = (result_scores >= config.Result_Score_threshold[data_name]).astype(int)
    visualize_anomalies_in_original_series_fewershallow_plotly(
    results_df_original=df_price_slice, 
    result=binary_result, 
    shadow=config.Shadow[data_name],
    start_time=config.Squence_start[data_name], end_time=config.Squence_end[data_name],
    save_fig=True,
    save_path=config.Save_path[data_name],
    fig_name='09_detect_results on Price',
    width=720 * 0.82,  ## 720
    height=460 * 0.86,  ## 450
    # line_color='#d6b4fc',
    line_name='Price',
    )




if __name__ == '__main__':

    # main(
    # data_name = '20170221_002807'
    # )

    parser = argparse.ArgumentParser(description='运行数据处理')
    parser.add_argument('--data_name', type=str, required=True, help='数据文件名（不含扩展名）')
    
    args = parser.parse_args()
    
    main(data_name=args.data_name)
    
 