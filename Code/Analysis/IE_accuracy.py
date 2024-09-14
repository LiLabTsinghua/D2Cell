import json


def is_string_in_list_or_contains(input_str, string_list):
    # Determines if the string is directly in the list (case insensitive)
    if input_str.lower() in [s.lower() for s in string_list]:
        return True
    return False


def is_string_in_list_or_contains_strain(input_str, string_list):
    # Determine if a string is directly in a list (case insensitive)
    if input_str.lower() in [s.lower() for s in string_list]:
        return True
    # Determine if a string is included in one of the strings in the list (case insensitive)
    for string_item in string_list:
        if input_str.lower() in string_item.lower():
            return True
    return False


def strain_name_accuracy(path, print_precision_detail=False):
    with open(path, 'r', encoding="utf-8") as json_file:
        data = json.load(json_file)
    precision_right_number = 0
    precison_all_number = 0
    recall_all_number = 0
    recall_right_number = 0
    final_data = []
    i = 0
    for paper in data['NER result']:
        i+=1
        extra_strain_list = []
        for detail_paper in paper['paper result']:
            text = detail_paper['text']
            predict_strain_name_list = []
            if 'predict_strain' in detail_paper and detail_paper['predict_strain']:
                predict_strain_name_list = detail_paper['predict_strain']
            precison_all_number += len(predict_strain_name_list)
            recall_all_number += len(detail_paper['strain'])
            if detail_paper['strain'] and predict_strain_name_list:
                gt_strain_list = [item for sublist in detail_paper['strain'] for item in sublist]
                for pre_strain in predict_strain_name_list:
                    if print_precision_detail:
                        if is_string_in_list_or_contains_strain(pre_strain, gt_strain_list):
                            precision_right_number += 1
                        else:
                            extra_strain_list.append(pre_strain)
                    else:
                        if is_string_in_list_or_contains_strain(pre_strain, gt_strain_list):
                            precision_right_number += 1
                for gt_strain in detail_paper['strain']:
                    for spe_strain in gt_strain:
                        if is_string_in_list_or_contains_strain(spe_strain, predict_strain_name_list):
                            recall_right_number += 1
                            break
            if extra_strain_list:
                final_data.append({'paper number': i, 'text': text, 'strain': extra_strain_list})
    if print_precision_detail:
        with open('875_strain_need_add.json', 'w') as file:
            json.dump({'paper':final_data}, file, indent=4)
    precision = precision_right_number/precison_all_number
    recall = recall_right_number/recall_all_number
    return precision_right_number, precison_all_number, recall_right_number, recall_all_number


def gene_name_accuracy(path):
    with open(path, 'r', encoding="utf-8") as json_file:
        data = json.load(json_file)
    precision_right_number = 0
    precison_all_number = 0
    recall_all_number = 0
    recall_right_number = 0
    # 计算precision
    for paper in data['NER result']:
        for detail_paper in paper['paper result']:
            predict_gene_name_list = []
            if 'predict_gene' in detail_paper and detail_paper['predict_gene']:
                predict_gene_name_list = detail_paper['predict_gene']
            precison_all_number += len(predict_gene_name_list)
            recall_all_number += len(detail_paper['gene'])
            if detail_paper['gene'] and predict_gene_name_list:
                gt_gene_list = [item for sublist in detail_paper['gene'] for item in sublist]
                for pre_gene in predict_gene_name_list:
                    if pre_gene and is_string_in_list_or_contains(pre_gene, gt_gene_list):
                        precision_right_number += 1
                for gt_gene in detail_paper['gene']:
                    for spe_gene in gt_gene:
                        if spe_gene and is_string_in_list_or_contains(spe_gene, predict_gene_name_list):
                            recall_right_number += 1
                            break
    precision = precision_right_number/precison_all_number
    recall = recall_right_number/recall_all_number
    return precision_right_number, precison_all_number, recall_right_number, recall_all_number


def calculate_accuracy():
    accuracy_array = []
    # qwen 110b result
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = \
        (strain_name_accuracy('../../Data/NER Data/IE-test-110b.json'))
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = \
        (gene_name_accuracy('../../Data/NER Data/IE-test-110b.json'))
    recall = (recall_right_number_strain + recall_right_number_gene) / (
                recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
                precison_all_number_strain + precison_all_number_gene)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy_array.append([precision, recall, f1_score])

    # qwen 14b result
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = \
        (strain_name_accuracy('../../Data/NER Data/IE-test-qwen-14b.json'))
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = \
        (gene_name_accuracy('../../Data/NER Data/IE-test-qwen-14b.json'))
    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy_array.append([precision, recall, f1_score])

    # gemini pro result
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-test-gemini.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-test-gemini.json')
    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy_array.append([precision, recall, f1_score])

    # claude3 result
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-test-claude3.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-test-claude3.json')
    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy_array.append([precision, recall, f1_score])

    #gpt4 result
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-test-gpt4.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-test-gpt4.json')
    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy_array.append([precision, recall, f1_score])

    # llama3 result
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-test-llama3.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-test-llama3.json')
    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy_array.append([precision, recall, f1_score])

    # llama3 lora result
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-test-llama3-lora.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-test-llama3-lora.json')
    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy_array.append([precision, recall, f1_score])

    #qwen lora result
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-test-qwen-lora.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-test-qwen-lora.json')
    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy_array.append([precision, recall, f1_score])
    return accuracy_array


if __name__ == '__main__':
    # print('qwen 110b')
    # precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain =\
    #     (strain_name_accuracy('../../Data/NER Data/IE-test-110b.json'))
    # precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = \
    #     (gene_name_accuracy('../../Data/NER Data/IE-test-110b.json'))
    # gene_precision = precision_right_number_gene / precison_all_number_gene
    # gene_recall = recall_right_number_gene / recall_all_number_gene
    # print('gene precison:', precision_right_number_gene / precison_all_number_gene)
    # print('gene recall:', recall_right_number_gene / recall_all_number_gene)
    # print('gene f1-score:', (2 * gene_precision * gene_recall) / (gene_precision + gene_recall))
    #
    # strain_precision = precision_right_number_strain / precison_all_number_strain
    # strain_recall = recall_right_number_strain / recall_all_number_strain
    # print('strain precison:', strain_precision)
    # print('strain recall:', strain_recall)
    # print('strain f1-score:', (2 * strain_precision * strain_recall) / (strain_precision + strain_recall))
    #
    # recall = (recall_right_number_strain + recall_right_number_gene)/ (recall_all_number_strain+recall_all_number_gene)
    # precision = (precision_right_number_strain+precision_right_number_gene)/(precison_all_number_strain+precison_all_number_gene)
    # print('precision:', precision)
    # print('recall:', recall)
    # print('f1-score:', (2 * precision * recall) / (precision + recall))
    #
    print('\nqwen14b')
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-875_final-qwen14b.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-875_final-qwen14b.json')
    gene_precision = precision_right_number_gene / precison_all_number_gene
    gene_recall = recall_right_number_gene / recall_all_number_gene
    print('gene precison:', precision_right_number_gene / precison_all_number_gene)
    print('gene recall:', recall_right_number_gene / recall_all_number_gene)
    print('gene f1-score:', (2 * gene_precision * gene_recall) / (gene_precision + gene_recall))

    strain_precision = precision_right_number_strain / precison_all_number_strain
    strain_recall = recall_right_number_strain / recall_all_number_strain
    print('strain precison:', strain_precision)
    print('strain recall:', strain_recall)
    print('strain f1-score:', (2 * strain_precision * strain_recall) / (strain_precision + strain_recall))

    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    print('precision:', precision)
    print('recall:', recall)
    print('f1-score:', (2 * precision * recall) / (precision + recall))

    print('\nllama3 8b')
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-875_final-llama3_8b.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-875_final-llama3_8b.json')
    gene_precision = precision_right_number_gene / precison_all_number_gene
    gene_recall = recall_right_number_gene / recall_all_number_gene
    print('gene precison:', precision_right_number_gene / precison_all_number_gene)
    print('gene recall:', recall_right_number_gene / recall_all_number_gene)
    print('gene f1-score:', (2 * gene_precision * gene_recall) / (gene_precision + gene_recall))

    strain_precision = precision_right_number_strain / precison_all_number_strain
    strain_recall = recall_right_number_strain / recall_all_number_strain
    print('strain precison:', strain_precision)
    print('strain recall:', strain_recall)
    print('strain f1-score:', (2 * strain_precision * strain_recall) / (strain_precision + strain_recall))

    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    print('precision:', precision)
    print('recall:', recall)
    print('f1-score:', (2 * precision * recall) / (precision + recall))



    # print('\nGemini')
    # precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy('../../Data/NER Data/IE-test-gemini.json')
    # precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy('../../Data/NER Data/IE-test-gemini.json')
    # gene_precision = precision_right_number_gene / precison_all_number_gene
    # gene_recall = recall_right_number_gene / recall_all_number_gene
    # print('gene precison:', precision_right_number_gene / precison_all_number_gene)
    # print('gene recall:', recall_right_number_gene / recall_all_number_gene)
    # print('gene f1-score:', (2 * gene_precision * gene_recall) / (gene_precision + gene_recall))
    #
    # strain_precision = precision_right_number_strain / precison_all_number_strain
    # strain_recall = recall_right_number_strain / recall_all_number_strain
    # print('strain precison:', strain_precision)
    # print('strain recall:', strain_recall)
    # print('strain f1-score:', (2 * strain_precision * strain_recall) / (strain_precision + strain_recall))
    #
    # recall = (recall_right_number_strain + recall_right_number_gene) / (
    #         recall_all_number_strain + recall_all_number_gene)
    # precision = (precision_right_number_strain + precision_right_number_gene) / (
    #         precison_all_number_strain + precison_all_number_gene)
    # print('precision:', precision)
    # print('recall:', recall)
    # print('f1-score:', (2 * precision * recall) / (precision + recall))
    #
    # print('\nClaude3')
    # precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy('../../Data/NER Data/IE-test-claude3.json')
    # precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy('../../Data/NER Data/IE-test-claude3.json')
    # gene_precision = precision_right_number_gene / precison_all_number_gene
    # gene_recall = recall_right_number_gene / recall_all_number_gene
    # print('gene precison:', precision_right_number_gene / precison_all_number_gene)
    # print('gene recall:', recall_right_number_gene / recall_all_number_gene)
    # print('gene f1-score:', (2 * gene_precision * gene_recall) / (gene_precision + gene_recall))
    #
    # strain_precision = precision_right_number_strain / precison_all_number_strain
    # strain_recall = recall_right_number_strain / recall_all_number_strain
    # print('strain precison:', strain_precision)
    # print('strain recall:', strain_recall)
    # print('strain f1-score:', (2 * strain_precision * strain_recall) / (strain_precision + strain_recall))
    # recall = (recall_right_number_strain + recall_right_number_gene) / (
    #         recall_all_number_strain + recall_all_number_gene)
    # precision = (precision_right_number_strain + precision_right_number_gene) / (
    #         precison_all_number_strain + precison_all_number_gene)
    # print('precision:', precision)
    # print('recall:', recall)
    # print('f1-score:', (2 * precision * recall) / (precision + recall))
    #
    # print('\ngpt4')
    # precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy('../../Data/NER Data/IE-test-gpt4.json')
    # precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy('../../Data/NER Data/IE-test-gpt4.json')
    # gene_precision = precision_right_number_gene / precison_all_number_gene
    # gene_recall = recall_right_number_gene / recall_all_number_gene
    # print('gene precison:', precision_right_number_gene / precison_all_number_gene)
    # print('gene recall:', recall_right_number_gene / recall_all_number_gene)
    # print('gene f1-score:', (2 * gene_precision * gene_recall) / (gene_precision + gene_recall))
    #
    # strain_precision = precision_right_number_strain / precison_all_number_strain
    # strain_recall = recall_right_number_strain / recall_all_number_strain
    # print('strain precison:', strain_precision)
    # print('strain recall:', strain_recall)
    # print('strain f1-score:', (2 * strain_precision * strain_recall) / (strain_precision + strain_recall))
    # recall = (recall_right_number_strain + recall_right_number_gene) / (
    #         recall_all_number_strain + recall_all_number_gene)
    # precision = (precision_right_number_strain + precision_right_number_gene) / (
    #         precison_all_number_strain + precison_all_number_gene)
    # print('precision:', precision)
    # print('recall:', recall)
    # print('f1-score:', (2 * precision * recall) / (precision + recall))
    #
    print('\nllama3 lora')
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-test-llama3-lora.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-test-llama3-lora.json')
    gene_precision = precision_right_number_gene / precison_all_number_gene
    gene_recall = recall_right_number_gene / recall_all_number_gene
    print('gene precison:', precision_right_number_gene / precison_all_number_gene)
    print('gene recall:', recall_right_number_gene / recall_all_number_gene)
    print('gene f1-score:', (2 * gene_precision * gene_recall) / (gene_precision + gene_recall))

    strain_precision = precision_right_number_strain / precison_all_number_strain
    strain_recall = recall_right_number_strain / recall_all_number_strain
    print('strain precison:', strain_precision)
    print('strain recall:', strain_recall)
    print('strain f1-score:', (2 * strain_precision * strain_recall) / (strain_precision + strain_recall))

    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    print('precision:', precision)
    print('recall:', recall)
    print('f1-score:', (2 * precision * recall) / (precision + recall))
    #
    print('\nqwen lora')
    precision_right_number_strain, precison_all_number_strain, recall_right_number_strain, recall_all_number_strain = strain_name_accuracy(
        '../../Data/NER Data/IE-test-qwen-lora.json')
    precision_right_number_gene, precison_all_number_gene, recall_right_number_gene, recall_all_number_gene = gene_name_accuracy(
        '../../Data/NER Data/IE-test-qwen-lora.json')
    gene_precision = precision_right_number_gene / precison_all_number_gene
    gene_recall = recall_right_number_gene / recall_all_number_gene
    print('gene precison:', precision_right_number_gene / precison_all_number_gene)
    print('gene recall:', recall_right_number_gene / recall_all_number_gene)
    print('gene f1-score:', (2 * gene_precision * gene_recall) / (gene_precision + gene_recall))

    strain_precision = precision_right_number_strain / precison_all_number_strain
    strain_recall = recall_right_number_strain / recall_all_number_strain
    print('strain precison:', strain_precision)
    print('strain recall:', strain_recall)
    print('strain f1-score:', (2 * strain_precision * strain_recall) / (strain_precision + strain_recall))

    recall = (recall_right_number_strain + recall_right_number_gene) / (
            recall_all_number_strain + recall_all_number_gene)
    precision = (precision_right_number_strain + precision_right_number_gene) / (
            precison_all_number_strain + precison_all_number_gene)
    print('precision:', precision)
    print('recall:', recall)
    print('f1-score:', (2 * precision * recall) / (precision + recall))
