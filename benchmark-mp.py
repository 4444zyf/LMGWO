from GWO import MLL_GWO, GWO, AGWO, LGWO, LCGWO, LCGWO2, LCGWO3, LCGWO4, LCGWO5, LCGWO6, LCGWO7, LCGWO8, ACGWO, LFGWO, LFGWO2, LFGWO3, LFGWO4, LFGWO5, LFGWO6, LFGWO7, LFGWO8, LFGWO9, LFGWO10, LMGWO, LMGWO2, LMGWO3, LMGWO4, LRGWO, MELGWO, SOGWO
from PSO import MPSO
from SSA import SSA
from ChOA import ChOA
from QRFS import QRFS
import numpy as np
from cec2019comp100digit import cec2019comp100digit as cec2019
import time
import pickle
from multiprocessing import Pool, Manager

# 评估算法函数
def evaluate_algorithm(algorithm, func_num, dim, lower, upper, max_iter, n_runs=50):
    cec2019.init(func_num, dim)
    results = []
    results_per_iter = []
    solution_per_iter = []
    start_time = time.time()
    for _ in range(n_runs):
        algo = algorithm(cec2019.eval, dim, lower, upper, max_iter=max_iter)
        best_solution, best_value = algo.optimize()
        results.append(best_value)
        results_per_iter.append(algo.get_best_fitness_history())
        solution_per_iter.append(best_solution)
    end_time = time.time()
    cec2019.end()
    return {
        'function': func_num,
        'best_fitness': np.min(results),
        'best_f_index': np.argmin(results),
        'mean': np.mean(results),
        'std': np.std(results),
        'time': (end_time - start_time) / n_runs,
        'results_per_iter': results_per_iter,
        'solution_per_iter': solution_per_iter
    }
from collections import defaultdict

# 修改 evaluate_and_store，使其返回结果而不是修改 shared_results
def evaluate_and_store(algo_name, algo_class, func_num, dim, lower, upper, max_iter):
    result = evaluate_algorithm(algo_class, func_num, dim, lower, upper, max_iter)
    print(f"{algo_name} - Function {func_num}: Best_fitness = {result['best_fitness']}, "
          f"Best_f_index = {result['best_f_index']}, Mean = {result['mean']}, "
          f"Std = {result['std']}, Time = {result['time']}")
    return (algo_name, result)


# 维度设置
dim = {
    1: 9,
    2: 16,
    3: 18,
    4: 10,
    5: 10,
    6: 10,
    7: 10,
    8: 10,
    9: 10,
    10: 10
}
lower = {
    1: -8192,
    2: -16384,
    3: -4,
    4: -100,
    5: -100,
    6: -100,
    7: -100,
    8: -100,
    9: -100,
    10: -100
}
upper = {
    1: 8192,
    2: 16384,
    3: 4,
    4: 100,
    5: 100,
    6: 100,
    7: 100,
    8: 100,
    9: 100,
    10: 100
}
# lower = -100
# upper = 100
max_iter = 500

# 存储算法列表，便于扩展
algorithms = {
    # 'LFGWO': LFGWO,
    # 'LFGWO2': LFGWO2,
    # 'LFGWO3': LFGWO3,
    # 'LFGWO4': LFGWO4,
    # 'LFGWO5': LFGWO5,
    'LFGWO6': LFGWO6,
    'LFGWO7': LFGWO7,
    'LFGWO8': LFGWO8,
    'LFGWO9': LFGWO9,
    'LFGWO10': LFGWO10,
    'LMGWO': LMGWO,
    'LMGWO2': LMGWO2,
    'LMGWO3': LMGWO3,
    'LMGWO4': LMGWO4,
    'LRGWO': LRGWO,
    'GWO': GWO,
    'AGWO': AGWO,
    'DMLLS_GWO': MLL_GWO,
    'QRFS': QRFS,
    'LGWO': LGWO,
    #'LCGWO': LCGWO,
    #'LCGWO2': LCGWO2,
    #'LCGWO3': LCGWO3,
    # 'LCGWO4': LCGWO4,
    #'LCGWO5': LCGWO5,
    'LCGWO6': LCGWO6,
    'LCGWO7': LCGWO7,
    'LCGWO8': LCGWO8,
    'MELGWO': MELGWO,
    'SOGWO': SOGWO,
    'MPSO': MPSO,
    'SSA': SSA,
    'ChOA': ChOA,
    'ACGWO': ACGWO,
}

# 在主代码中
if __name__ == "__main__":

    # 创建进程池
    pool = Pool()

    # 准备任务
    tasks = []
    for func_num in range(1, 11):
        for algo_name, algo_class in algorithms.items():
            tasks.append((algo_name, algo_class, func_num, dim[func_num], lower[func_num], upper[func_num], max_iter))

    # 运行任务并收集结果
    results = pool.starmap(evaluate_and_store, tasks)

    # 关闭并等待进程池结束
    pool.close()
    pool.join()

    # 将结果组织成字典
    organized_results = defaultdict(list)
    for algo_name, result in results:
        organized_results[algo_name].append(result)

    # 保存所有结果
    with open(f'./results/results.pkl', 'wb') as f:
        pickle.dump(dict(organized_results), f)
