# 评估结果分析
import json
import os
from collections import defaultdict
from functools import reduce

def analyze_single_result(data):
    success_rate = data['success_rate']
    results = data['results']
    task_len = len(results)
    print(f'评估task总数: {task_len}')
    print(f'成功率: {success_rate}')

    res = defaultdict(lambda: [list(), list()])
    for task in results:
        task_id = task['task_id']
        task_type = task_id.split('/')[-2].split('-')[0]
        if task['final_score'] == 1.0:
            res[task_type][0].append(task_id)
        else:
            res[task_type][1].append(task_id)
    for task_type in res:
        print(f"{task_type}成功率: {len(res[task_type][0])}/{len(res[task_type][0])+len(res[task_type][1])}")
    print(f"总成功率: {sum([len(res[task_type][0]) for task_type in res])}/{sum([len(res[task_type][0])+len(res[task_type][1]) for task_type in res])}")
    print("\n")

def analyze_multi_result(files):
    # 分析json_root_dir里所有json的评估结果
    # 1. 评估成功完成任务的交集数量

    all_eval_success_task = []
    all_eval_fail_task = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        print(f"分析文件: {file}")
        analyze_single_result(data)
        success_task = []
        fail_task = []
        results = data['results']
        for task in results:
            if task['final_score'] == 1.0:
                success_task.append(task['task_id'])
            else:
                fail_task.append(task['task_id'])
        all_eval_success_task.append(success_task)
        all_eval_fail_task.append(fail_task)
    
    # 交集
    intersection = reduce(lambda x,y: set(x)&set(y), all_eval_success_task)
    print(f"成功任务交集数量: {len(intersection)}")
    # 求每个列表与intersection的差集
    # 求每个列表与交集的差集
    differences = []
    for i, lst in enumerate(all_eval_success_task):
        diff = set(lst) - intersection
        differences.append(diff)
        print(f"{files[i]} 成功任务差集数量: {len(diff)}")
    print("\n")

    # 失败任务交集
    intersection = reduce(lambda x,y: set(x)&set(y), all_eval_fail_task)
    print(f"失败任务交集数量: {len(intersection)}")
    # 求每个列表与intersection的差集
    # 求每个列表与交集的差集
    differences = []
    for i, lst in enumerate(all_eval_fail_task):
        diff = set(lst) - intersection
        differences.append(diff)
        print(f"{files[i]} 失败任务差集数量: {len(diff)}")
        


    
    


if __name__ == '__main__':
    # json_path = '/home/xh/Desktop/code/eval_result/overall_results_eval_out_of_distribution_llama3_20241114_115813.json'
    # with open(json_path, 'r') as f:
    #     data = json.load(f)
    # analyze_single_result(data)

    files = ['/mnt/tangyehui/code/LLaMA-Factory/saves/qwen2_vl-7b/sft_eval/overall_results_eval_out_of_distribution_llama3_20241115_162047.json',
    '/mnt/tangyehui/code/LLaMA-Factory/saves/qwen2_vl-7b/sft_eval/overall_results_eval_out_of_distribution_llama3_20241118_100418.json',
    '/mnt/tangyehui/code/LLaMA-Factory/saves/qwen2_vl-7b/sft_eval/overall_results_eval_out_of_distribution_llama3_20241120_100117.json']

    analyze_multi_result(files)