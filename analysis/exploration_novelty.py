import pandas as pd
import random
import pickle

from src.preprocess import standardization_pipeline, remove_smiles_duplicates


if __name__ == "__main__":    
    # set random seed, used for error generation & initiation transformer
    SEED = 42
    random.seed(SEED)

    name = 'selective_ki'
    errors_per_molecule = 1000
    error_source = "Data/explore/%s_with_%s_errors_index.csv" % (name, errors_per_molecule)

    folder_raw = "RawData/"
    folder_out = "Data/"
    invalid_type = 'multiple'
    num_errors = 12
    threshold = 200
    data_source = f"PAPYRUS_{threshold}"

    df = pd.read_csv('Data/explore/selective_ki_with_1000_errors_index.csv')
    df_o = pd.read_csv('Data/explore/selective_ki_with_1000_errors_index_fixed.csv')
    df = pd.merge(df, df_o, left_on='SMILES_TARGET', right_on='ORIGINAL')
    print(len(df))
    #df = df.join(df_o)
    #df_val = df.dropna()
    #df_val = df.dropna(subset=['CORRECT'])

    num_original = []
    num_fixed = []
    num_new = []
    news = []
    new_samples = []
    sample_nums = []

    df["STD_CORRECT"] = df.apply(
                lambda row: standardization_pipeline(row["CORRECT"]) if(row["CORRECT"]) else None, axis=1
            )#.dropna()
    df_val = df.dropna(subset=["STD_CORRECT"])
    with open(f"Data/explore/{error_source.split('/')[2].split('.')[0]}.txt", 'w') as f:
            f.write(f'Percentage valid: {len(df_val)/len(df_o) * 100} %\n')

    collect_new = False
    if collect_new:
        original = df['STD_SMILES'].values.tolist()
        fixed = df_val["STD_CORRECT"].values.tolist()
        shared = list(set(original).intersection(fixed))
        new = list(set(fixed) - set(original))
        #print(not_fixed)
        fixed = set(fixed)
        print(len(fixed))
        #print(new)
        print(len(new))
        df_new = pd.DataFrame(new, columns =['SMILES'])
        df_new.to_csv(f"Data/explore/{error_source.split('/')[2].split('.')[0]}_new.csv", index=None)

    print(df['Unnamed: 0'].max())
    for index in range(df['Unnamed: 0'].max()):
        df_i = df_val.loc[df_val['Unnamed: 0'] == index]


        df_new = remove_smiles_duplicates(df_i, subset="STD_CORRECT")
        df_i = remove_smiles_duplicates(df_i, subset="ORIGINAL_SMILES")

        original = df_i['ORIGINAL_SMILES'].values.tolist()
        fixed = df_new['STD_CORRECT'].values.tolist()
        #print(fixed)
        num_original.append(len(original))
        shared = list(set(original).intersection(fixed))
        new = list(set(fixed) - set(original))
        news.extend(new)
        #print(not_fixed)
        fixed = set(fixed)
        #print(fixed)
        num_fixed.append(len(fixed))
        num_new.append(len(new))
        
        df_i = df.loc[df['Unnamed: 0'] == index]
        new_sample = []
        sample_num = []
        for i in range(1,len(df_i)+1):
            #print(i)
            sample_num.append(i)
            df_s = df_i.sample(i, random_state=42)
            #print(len(df_s))
            df_s = df_s.dropna(subset=["STD_CORRECT"])
            df_new = remove_smiles_duplicates(df_s, subset="STD_CORRECT")
            df_s = remove_smiles_duplicates(df_i, subset="ORIGINAL_SMILES")
            original = df_s['ORIGINAL_SMILES'].values.tolist()
            fixed = df_new['STD_CORRECT'].values.tolist()
            new = list(set(fixed) - set(original))
            #print(len(new))
            new_sample.append(len(new))
        new_samples.append(new_sample)
        sample_nums.append(sample_num)

    # print(num_original)
    # print(num_fixed)
    # print(num_new)

    print(new_samples)
    


    # store list in binary file so 'wb' mode
    with open('explore_novelty_1000', 'wb') as fp:
        pickle.dump(new_samples, fp)
        print('Done writing list into a binary file')

    with open('explore_num_1000', 'wb') as fp:
        pickle.dump(sample_nums, fp)
        print('Done writing list into a binary file')


    print(sum(num_fixed))
    print(sum(num_new))
    with open(f"Data/explore/{error_source.split('/')[2].split('.')[0]}.txt", 'a') as f:
            f.write(f'Percentage unique: {sum(num_fixed)/len(df_val) * 100} %\n')
            f.write(f'Percentage novel: {sum(num_new)/len(df_val) * 100} %\n')

    # df = pd.DataFrame(news, columns =['SMILES'])
    # df.to_csv(f"Data/explore/{error_source.split('/')[2].split('.')[0]}_new.csv", index=None)