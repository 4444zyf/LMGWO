import numpy as np
import copy
from tqdm import tqdm

class SSA:
    def __init__(self, func, dim, lower, upper, n_pop=100, max_iter=100):
        """
        初始化SSA算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界
        :param upper: 搜索空间上界
        :param n_pop: 种群规模（麻雀数量）
        :param max_iter: 最大迭代次数
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.population = np.random.uniform(lower, upper, (n_pop, dim))  # 初始化麻雀位置
        self.personal_best_position = copy.deepcopy(self.population)  # 个人历史最优位置
        self.personal_best_fitness = np.full(n_pop, float('inf'))  # 个人历史最优适应度值
        self.global_best_position = np.ones(self.dim)  # 全局最优位置
        self.global_best_fitness = float('inf')  # 全局最优适应度值
        self.best_fitness_history = []  # 保存每次迭代的全局最优适应度值

    def optimize(self):
        """
        优化过程
        :return: 最优解及其对应的适应度值
        """
        for iteration in tqdm(range(self.max_iter), desc="SSA Progress"):
            # 更新每个麻雀的位置
            for i in range(self.n_pop):
                R2 = np.random.rand()  # 警报值
                ST = 0.8  # 安全阈值
                self.update_position(i, R2, ST)
            
            # 计算每个麻雀的适应度值
            fitness = np.apply_along_axis(self.func, 1, self.population)

            # 更新个人历史最优和全局最优
            self.update_best_positions(fitness)

            # 保存当前全局最优适应度值
            self.best_fitness_history.append(self.global_best_fitness)

        return self.global_best_position, self.global_best_fitness

    def update_position(self, i, R2, ST):
        """
        更新每个麻雀的位置
        :param i: 当前麻雀的索引
        :param R2: 警报值
        :param ST: 安全阈值
        """
        if R2 < ST:  # 无捕食者，进入广泛搜索模式
            self.population[i] = self.population[i] * np.exp(-(i + 1) / self.max_iter)
        else:  # 有捕食者，麻雀逃逸
            self.population[i] = self.population[i] + np.random.normal() * np.ones(self.dim)

        # 保证位置在搜索空间内
        self.population[i] = np.clip(self.population[i], self.lower, self.upper)

    def update_best_positions(self, fitness):
        """
        更新每个麻雀的个人最优位置以及全局最优位置
        :param fitness: 当前种群的适应度值
        """
        for i in range(self.n_pop):
            if fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness[i]
                self.personal_best_position[i] = copy.deepcopy(self.population[i])

            if fitness[i] < self.global_best_fitness:
                self.global_best_fitness = fitness[i]
                self.global_best_position = copy.deepcopy(self.population[i])

    def get_best_fitness_history(self):
        """
        返回全局最优适应度值历史记录
        :return: 历史记录
        """
        return self.best_fitness_history


