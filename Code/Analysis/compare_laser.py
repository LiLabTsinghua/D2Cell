import pandas as pd
import math


def clean_numbe_list(number_list):
    if number_list and len(number_list) == 1:
        try:
            if int(number_list[0]) == number_list[0]:
                number_list = [str(int(numbers)) for numbers in number_list]
            else:
                number_list = [str(numbers) for numbers in number_list]
        except ValueError:
            number_list = []
    elif number_list and len(number_list) > 1:
        new_number_list = []
        for numbers in number_list:
            try:
                if int(numbers) == numbers:
                    new_number_list.append(str(int(numbers)))
                else:
                    new_number_list.append(str(numbers))
            except ValueError:
                continue
        number_list = list(set(new_number_list))
    return number_list


def accuracy(gt_list, predict_list, acc):
    predict_number = 0
    gt_list = [entity for entity in gt_list if isinstance(entity, str)]
    predict_list = [entity for entity in predict_list if isinstance(entity, str)]
    for entity in gt_list:
        if any(entity.lower() in pre_entity.lower() for pre_entity in predict_list):
            predict_number += 1
    if gt_list:
        acc += predict_number / len(gt_list)
    else:
        acc += 1
    return acc


def accuracy_product(gt_list, predict_list, acc):
    predict_number = 0
    gt_list = [entity for entity in gt_list if isinstance(entity, str)]
    predict_list = [' ' if pd.isna(x) else x for x in predict_list]
    for entity in gt_list:
        if any(entity.lower() in pre_entity.lower() for pre_entity in predict_list):
            predict_number += 1
        elif any(pre_entity.lower() in entity.lower() for pre_entity in predict_list) and not any(entity.lower() in pre_entity.lower() for pre_entity in predict_list):
            predict_number += 1
    if gt_list:
        acc += predict_number / len(gt_list)
    else:
        acc += 1
    return acc

def compare_d2cell(d2cell_path, laser_data_path):
    df_d2cell = pd.read_csv(d2cell_path)
    df_d2cell = df_d2cell.dropna(subset=['doi'])
    laser_df = pd.read_csv(laser_data_path)
    d2cell_doi = list(set(df_d2cell['doi'].tolist()))

    gene_acc = 0
    tem_acc = 0
    ph_acc = 0
    carbon_acc = 0
    medium_acc = 0
    product_titer_acc = 0
    vessel_acc = 0
    product_acc = 0
    for doi in d2cell_doi:
        laser_df_row = laser_df[laser_df['DOI_replaced'] == doi].iloc[0]
        gt_gene_list = eval(laser_df_row['Knocked out'])
        gt_gene_list.extend(eval(laser_df_row['Overexpress']))
        gt_gene_list.extend(eval(laser_df_row['others gene']))
        gt_product_list = [laser_df_row['TargetMolecule']]
        gt_product_list = [product for product in gt_product_list if isinstance(product, str)]
        gt_product_list = [item.strip() for sublist in gt_product_list for item in sublist.split(';')]

        gt_carbonsource_list = [laser_df_row['CarbonSource']]
        gt_carbonsource_list = [item.strip() for sublist in gt_carbonsource_list for item in sublist.split(',')]
        gt_ph_list = [laser_df_row['pH']]
        gt_ph_list = clean_numbe_list(gt_ph_list)
        gt_tem_list = [laser_df_row['Temperature']]
        gt_tem_list = clean_numbe_list(gt_tem_list)
        gt_medium_list = [laser_df_row['Medium']]
        gt_oxygen_list = [laser_df_row['Oxygen']]
        gt_vessel_list = [laser_df_row['CultureSystem']]

        gt_titer_list = [laser_df_row['FinalTiter']]
        if laser_df_row['FinalYield']:
            gt_titer_list.append(laser_df_row['FinalYield'])
        gt_titer_list = clean_numbe_list(gt_titer_list)

        df_d2cell_row = df_d2cell[df_d2cell['doi'] == doi].iloc[0]
        predict_gene_list = eval(df_d2cell_row['gene'])
        gene_acc = accuracy(gt_gene_list, predict_gene_list, gene_acc)

        predict_temperature_list = eval(df_d2cell_row['temperature'])
        tem_acc = accuracy(gt_tem_list, predict_temperature_list, tem_acc)

        predict_ph_list = eval(df_d2cell_row['ph'])
        ph_acc = accuracy(gt_ph_list, predict_ph_list, ph_acc)

        predict_carbon_list = eval(df_d2cell_row['carbon source'])
        carbon_acc = accuracy(gt_carbonsource_list, predict_carbon_list, carbon_acc)

        predict_medium_list = eval(df_d2cell_row['medium'])
        medium_acc = accuracy(gt_medium_list, predict_medium_list, medium_acc)

        predict_product_titer_list = eval(df_d2cell_row['product titer'])
        product_titer_acc = accuracy(gt_titer_list, predict_product_titer_list, product_titer_acc)

        predict_vessel_list = eval(df_d2cell_row['vessel and feed mode'])
        vessel_acc = accuracy(gt_vessel_list, predict_vessel_list, vessel_acc)

        predict_product_list = eval(df_d2cell_row['product'])
        product_acc = accuracy_product(gt_product_list, predict_product_list, product_acc)

    print('product recall', product_acc / len(d2cell_doi))
    print('titer recall', product_titer_acc / len(d2cell_doi))
    print('gene recall', gene_acc/len(d2cell_doi))
    print('temperature recall', tem_acc/len(d2cell_doi))
    print('ph recall', ph_acc/len(d2cell_doi))
    print('carbon source recall', carbon_acc/len(d2cell_doi))
    print('medium recall', medium_acc/len(d2cell_doi))
    print('vessel recall', vessel_acc/len(d2cell_doi))
    return [product_acc / len(d2cell_doi), product_titer_acc / len(d2cell_doi), gene_acc / len(d2cell_doi),
            tem_acc / len(d2cell_doi),  ph_acc / len(d2cell_doi), carbon_acc / len(d2cell_doi),
            medium_acc / len(d2cell_doi), vessel_acc / len(d2cell_doi)]


def compare_other(predict_path, laser_data_path):
    df_result = pd.read_csv(predict_path)
    laser_df = pd.read_csv(laser_data_path)
    result_doi = list(set(df_result['doi'].tolist()))

    gene_acc = 0
    tem_acc = 0
    ph_acc = 0
    carbon_acc = 0
    medium_acc = 0
    product_titer_acc = 0
    vessel_acc = 0
    product_acc = 0
    for doi in result_doi:
        if isinstance(doi, str):
            laser_df_row = laser_df[laser_df['DOI_replaced'] == doi].iloc[0]
            gt_gene_list = eval(laser_df_row['Knocked out'])
            gt_gene_list.extend(eval(laser_df_row['Overexpress']))
            gt_gene_list.extend(eval(laser_df_row['others gene']))
            gt_product_list = [laser_df_row['TargetMolecule']]
            gt_product_list = [product for product in gt_product_list if isinstance(product, str)]
            gt_product_list = [item.strip() for sublist in gt_product_list for item in sublist.split(';')]

            gt_carbonsource_list = [laser_df_row['CarbonSource']]
            gt_carbonsource_list = [item.strip() for sublist in gt_carbonsource_list for item in sublist.split(',')]
            gt_ph_list = [laser_df_row['pH']]
            gt_ph_list = clean_numbe_list(gt_ph_list)
            gt_tem_list = [laser_df_row['Temperature']]
            gt_tem_list = clean_numbe_list(gt_tem_list)
            gt_medium_list = [laser_df_row['Medium']]
            gt_oxygen_list = [laser_df_row['Oxygen']]
            gt_vessel_list = [laser_df_row['CultureSystem']]

            gt_titer_list = [laser_df_row['FinalTiter']]
            if laser_df_row['FinalYield']:
                gt_titer_list.append(laser_df_row['FinalYield'])
            gt_titer_list = clean_numbe_list(gt_titer_list)

            df_result_row = df_result[df_result['doi'] == doi]

            predict_gene_list = df_result_row['knock out gene'].tolist()
            predict_gene_list.extend(df_result_row['overexpress gene'].tolist())
            predict_gene_list.extend(df_result_row['heterologous gene'].tolist())
            gene_acc = accuracy(gt_gene_list, predict_gene_list, gene_acc)

            predict_temperature_list = df_result_row['temperature'].tolist()
            tem_acc = accuracy(gt_tem_list, predict_temperature_list, tem_acc)

            predict_ph_list = df_result_row['ph'].tolist()
            ph_acc = accuracy(gt_ph_list, predict_ph_list, ph_acc)

            predict_carbon_list = df_result_row['carbon source'].tolist()
            carbon_acc = accuracy(gt_carbonsource_list, predict_carbon_list, carbon_acc)

            predict_medium_list = df_result_row['medium'].tolist()
            medium_acc = accuracy(gt_medium_list, predict_medium_list, medium_acc)

            predict_product_titer_list = df_result_row['product titer'].tolist()
            product_titer_acc = accuracy(gt_titer_list, predict_product_titer_list, product_titer_acc)

            predict_vessel_list = df_result_row['vessel and feed mode'].tolist()
            vessel_acc = accuracy(gt_vessel_list, predict_vessel_list, vessel_acc)

            predict_product_list = df_result_row['product'].tolist()
            product_acc = accuracy_product(gt_product_list, predict_product_list, product_acc)

    print('product recall', product_acc / len(result_doi))
    print('titer recall', product_titer_acc / len(result_doi))
    print('gene recall', gene_acc/len(result_doi))
    print('temperature recall', tem_acc/len(result_doi))
    print('ph recall', ph_acc/len(result_doi))
    print('carbon source recall', carbon_acc/len(result_doi))
    print('medium recall', medium_acc/len(result_doi))
    print('vessel recall', vessel_acc/len(result_doi))
    return [product_acc / len(result_doi), product_titer_acc / len(result_doi), gene_acc/len(result_doi),
            tem_acc/len(result_doi), ph_acc/len(result_doi), carbon_acc/len(result_doi),
            medium_acc/len(result_doi), vessel_acc/len(result_doi)]


if __name__ == '__main__':
    compare_other('laser_direct_110b.xlsx', 'clean_laser_dataset.xlsx')