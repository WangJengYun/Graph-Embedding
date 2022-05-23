import numpy as np 
import pandas as pd
import gc

dtype_low_memory = {'dt':np.int8,
 'chid':np.object,
 'shop_tag':np.object,
 'txn_cnt':np.int8,
 'txn_amt':np.float64,
 'domestic_offline_cnt':np.int8,
 'domestic_online_cnt':np.int8,
 'overseas_offline_cnt':np.int8,
 'overseas_online_cnt':np.int8,
 'domestic_offline_amt_pct':np.float16,
 'domestic_online_amt_pct':np.float16,
 'overseas_offline_amt_pct':np.float16,
 'overseas_online_amt_pct':np.float16,
 'card_1_txn_cnt':np.int8,
 'card_2_txn_cnt':np.int8,
 'card_3_txn_cnt':np.int8,
 'card_4_txn_cnt':np.int8,
 'card_5_txn_cnt':np.int8,
 'card_6_txn_cnt':np.int8,
 'card_7_txn_cnt':np.int8,
 'card_8_txn_cnt':np.int8,
 'card_9_txn_cnt':np.int8,
 'card_10_txn_cnt':np.int8,
 'card_11_txn_cnt':np.int8,
 'card_12_txn_cnt':np.int8,
 'card_13_txn_cnt':np.int8,
 'card_14_txn_cnt':np.int8,
 'card_other_txn_cnt':np.int8,
 'card_1_txn_amt_pct':np.float16,
 'card_2_txn_amt_pct':np.float16,
 'card_3_txn_amt_pct':np.float16,
 'card_4_txn_amt_pct':np.float16,
 'card_5_txn_amt_pct':np.float16,
 'card_6_txn_amt_pct':np.float16,
 'card_7_txn_amt_pct':np.float16,
 'card_8_txn_amt_pct':np.float16,
 'card_9_txn_amt_pct':np.float16,
 'card_10_txn_amt_pct':np.float16,
 'card_11_txn_amt_pct':np.float16,
 'card_12_txn_amt_pct':np.float16,
 'card_13_txn_amt_pct':np.float16,
 'card_14_txn_amt_pct':np.float16,
 'card_other_txn_amt_pct':np.float16,
 'masts':np.object,
 'educd':np.object,
 'trdtp':np.object,
 'naty':np.object,
 'poscd':np.object,
 'cuorg':np.object,
 'slam':np.float64,
 'gender_code':np.object,
 'age':np.object,
 'primary_card':np.object}

def create_embedding_dataframe(data_path):
    # data_path = './dataset/tbrain_cc_training_48tags_hash_final.csv'
    raw_df = pd.read_csv(data_path,dtype = dtype_low_memory,engine = 'c')
    df = raw_df[['chid','dt', 'txn_amt','masts','slam','gender_code','age', 'shop_tag']]
    del raw_df
    gc.collect()

    # Masts Transformation
    df['masts_category'] = 'nan_masts'
    df.loc[df['masts'] == '1', 'masts_category'] = 'masts_1'
    df.loc[(df['masts'] == '2'), 'masts_category'] = 'masts_2'
    df.loc[df['masts'] == '3', 'masts_category'] = 'masts_3'

    # Age Transformation
    df['age_category'] = 'nan_age_group'
    df['age'] = df['age'].astype(np.float16)
    df.loc[df['age'] <= 3.0, 'age_category'] = 'age_group_1'
    df.loc[(df['age'] >= 3.0) & (df['age'] < 7.0), 'gender_category'] = 'age_group_2'
    df.loc[df['age'] < 10.0, 'age_category'] = 'age_group_3'

    # Gender Transformation
    df['gender_category'] = 'nan_gender'
    df.loc[df['gender_code'] == '1', 'gender_category'] = 'gender1'
    df.loc[df['gender_code'] == '0', 'gender_category'] = 'gender2'

    # slam Transformation
    df['slam_category'] = 'nan_slam'
    df.loc[df['slam'] < 50000.0, 'slam_category'] = 'low_slam'
    df.loc[(df['slam'] >= 50000.0) & (df['slam'] < 100000.0) , 'slam_category'] = 'mid_slam'
    df.loc[(df['slam'] >= 100000.0) , 'slam_category'] = 'high_slam'

    # txn_amt Transformation
    df['txn_amt_category'] = 'nan_txn_amt'
    df.loc[df['txn_amt'] < 5000.0, 'txn_amt_category'] = 'low_txn_amt'
    df.loc[(df['txn_amt'] >= 5000.0) & (df['txn_amt'] < 10000.0) , 'txn_amt_category'] = 'mid_txn_amt'
    df.loc[(df['txn_amt'] >= 10000.0) & (df['txn_amt'] < 30000.0) , 'txn_amt_category'] = 'mid_high_txn_amt'
    df.loc[(df['txn_amt'] >= 30000.0) , 'txn_amt_category'] = 'v_high_txn_amt'


    # Create final DataFrame
    final_df = pd.DataFrame()
    final_df['from'] = df['chid'].astype(str)
    df['dt'] = df['dt']%12
    df.loc[df['dt'] == 0, 'dt'] = 12
    # final_df['rel'] = df['dt'].astype(str) + '/' + df['masts_category'].astype(str) + '/' + df['age_category'].astype(str) + '/' + df['gender_category'].astype(str) + '/' + df['slam_category'].astype(str) + '/' + df['txn_amt_category'].astype(str)
    final_df['rel'] = 'month_'+df['dt'].astype(str) 
    final_df['to'] = df['shop_tag'].astype(str)
    print(final_df)

    final_df.to_csv('./dataset/graph_embedding_data_v2.csv',index = False)
    


if __name__ == '__main__':

    # Change data file path here :)
    # data_path = 'sample_data.csv'
    data_path = './dataset/tbrain_cc_training_48tags_hash_final.csv'
    create_embedding_dataframe(data_path)
