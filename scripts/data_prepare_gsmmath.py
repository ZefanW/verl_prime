import re
import os
import datasets

# from verl.utils.hdfs_io import copy, makedirs
import argparse

# To extract the solution for each prompts in the dataset
# def extract_solution(solution_str):
# ...
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

instruction_following = "Let's think step by step and output the final answer after \"####\"."

# add a row to each data item that represents a unique id
def make_map_fn(split):

    def process_fn(example, idx):
        question = example.pop('question')

        question = question + ' ' + instruction_following

        answer = example.pop('answer')
        solution = extract_solution(answer)
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data

    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/test/test05/wzf/verl_prime/scripts/dataset/gsmmath')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    train_dataset_list=[]
    test_dataset_list=[]

    num_few_shot = 5
    for data_source in ['openai/gsm8k']:


        dataset = datasets.load_dataset(data_source,'main')

        train_dataset = dataset['train']
        test_dataset = dataset['test']

            # Construct a `def make_map_fn(split)` for the corresponding datasets.
        # ...

        train_dataset_list.append(train_dataset.map(function=make_map_fn('train'), with_indices=True))
        test_dataset_list.append(test_dataset.map(function=make_map_fn('test'), with_indices=True))

    train_dataset=datasets.concatenate_datasets(train_dataset_list)
    test_dataset=datasets.concatenate_datasets(test_dataset_list)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # makedirs(hdfs_dir)
    #
    # copy(src=local_dir, dst=hdfs_dir)