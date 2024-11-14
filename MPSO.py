import numpy as np
import copy
from tqdm import tqdm

class MPSO:
    def __init__(self, func, dim, lower, upper, n_pop=100, max_iter=100):
        """
        初始化MPSO算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界
        :param upper: 搜索空间上界
        :param n_pop: 种群规模（粒子数量）
        :param max_iter: 最大迭代次数
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.population = np.random.uniform(lower, upper, (n_pop, dim))  # 初始化粒子位置
        self.velocity = np.zeros_like(self.population)  # 初始化粒子速度
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
        for iteration in tqdm(range(self.max_iter), desc="MPSO Progress"):
            # 计算线性递减惯性权重
            w = self.sigmoid_inertia_weight(iteration)
            position = copy.deepcopy(self.population)
            # 更新每个粒子的位置和速度
            for i in range(self.n_pop):
                self.velocity[i] = self.update_velocity(position[i], self.personal_best_position[i], self.global_best_position, w)
                self.population[i] = self.update_position(position[i])

            # 计算每个粒子的适应度值
            fitness = np.apply_along_axis(self.func, 1, self.population)

            # 更新个人历史最优和全局最优
            self.update_best_positions(fitness)

            # 保存当前全局最优适应度值
            self.best_fitness_history.append(self.global_best_fitness)

            # 计算最大聚焦距离，若陷入局部最优，则应用小波变异
            if self.maximal_focus_distance() < 1e-6:
                self.apply_wavelet_mutation()

        return self.global_best_position, self.global_best_fitness

    def sigmoid_inertia_weight(self, iteration):
        """
        类S形惯性权重的计算
        :param iteration: 当前迭代次数
        :return: 惯性权重
        """
        w_max, w_min = 0.9, 0.4
        alpha = 0.2  # 调整参数
        if iteration <= alpha * self.max_iter:
            return w_max
        else:
            # 限制输入值的范围以避免溢出
            exp_input = np.clip(10 * iteration - 2 * self.max_iter, -700, 700)
            return 1 / (1 + np.exp(exp_input) / self.max_iter) + w_min

    def update_velocity(self, position, personal_best, global_best, w):
        """
        更新粒子的速度
        :param position: 当前粒子的位置
        :param personal_best: 个人最优位置
        :param global_best: 全局最优位置
        :param w: 惯性权重
        :return: 更新后的速度
        """
        c1, c2 = 2.0, 2.0  # 学习因子
        r1, r2 = np.random.rand(), np.random.rand()

        cognitive = c1 * r1 * (personal_best - position)
        social = c2 * r2 * (global_best - position)
        return w * self.velocity[np.where(self.population == position)[0][0]] + cognitive + social

    def update_position(self, position):
        """
        根据当前速度更新粒子的位置
        :param position: 当前粒子的位置
        :return: 更新后的位置
        """
        idx = np.where(self.population == position)[0][0]
        new_position = position + self.velocity[idx]
        return np.clip(new_position, self.lower, self.upper)

    def update_best_positions(self, fitness):
        """
        更新每个粒子的个人最优位置以及全局最优位置
        :param fitness: 当前种群的适应度值
        """
        for i in range(self.n_pop):
            if fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness[i]
                self.personal_best_position[i] = copy.deepcopy(self.population[i])

            if fitness[i] < self.global_best_fitness:
                self.global_best_fitness = fitness[i]
                self.global_best_position = copy.deepcopy(self.population[i])

    def maximal_focus_distance(self):
        """
        计算最大聚焦距离
        :return: 最大聚焦距离
        """
        distances = np.linalg.norm(self.population - self.global_best_position, axis=1)
        return np.max(distances)

    def apply_wavelet_mutation(self):
        """
        对所有粒子应用小波变异
        """
        for i in range(self.n_pop):
            self.population[i] = self.wavelet_mutation(self.population[i])

    def wavelet_mutation(self, position, dilation_param=1):
        """
        小波变异的实现
        :param position: 当前粒子的位置
        :param dilation_param: 伸缩参数
        :return: 变异后的位置
        """
        sigma = (1 / np.sqrt(dilation_param)) * np.exp(-(position / dilation_param) ** 2) * np.cos(5 * np.pi * (position / dilation_param))
        mutated_position = position + sigma * (self.upper - self.lower) * np.random.random(self.dim)
        return np.clip(mutated_position, self.lower, self.upper)

    def get_best_fitness_history(self):
        """
        返回全局最优适应度值历史记录
        :return: 历史记录
        """
        return self.best_fitness_history
