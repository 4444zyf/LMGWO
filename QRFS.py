import numpy as np
from scipy.stats import qmc
import copy
from tqdm import tqdm

class QRFS:
    def __init__(self, func, dim, lower, upper, n_pop=25, n_fractals=15, max_iter=1000):
        """
        初始化QRFS算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界
        :param upper: 搜索空间上界
        :param n_pop: 每个分形的种群规模
        :param n_fractals: 分形数量
        :param max_iter: 最大迭代次数
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.n_pop = n_pop
        self.n_fractals = n_fractals
        self.max_iter = max_iter
        self.popIni = n_pop
        self.popFi = 20
        self.subpopulations = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.best_fitness_history = []
        self.centroids = [None] * n_fractals  # 用于存储每个分形的中心粒子
        self.lower_bounds = [np.copy(lower)] * n_fractals  # 每个分形的下界
        self.upper_bounds = [np.copy(upper)] * n_fractals  # 每个分形的上界

    def optimize(self):
        """
        优化过程
        :return: 最优解及其对应的适应度值
        """
        # 初始化种群
        self.initialize_population()

        for iteration in tqdm(range(self.max_iter), desc="QRFS Progress"):
            # 计算当前迭代下的种群规模
            npop = self.sigmoid_population_decrease(iteration)

            for j in range(self.n_fractals):
                # 重新初始化分型的种群
                newpop = np.ascontiguousarray(self.init_subpopulation(npop, j)) # 根据每个分形的边界生成新种群
                # newpop = self.init_subpopulation(npop, j)
                fitness = np.apply_along_axis(self.func, 1, newpop)
                self.update_subpopulation(j, newpop, fitness)

            # 更新全局最优
            self.update_global_best()

            # 记录全局最优适应度
            self.best_fitness_history.append(self.global_best_fitness)

        return self.global_best_position, self.global_best_fitness

    def initialize_population(self):
        """
        初始化种群的分形结构，并更新每个分形的上下界，统计全局最优值和centroid
        """
        # 使用Halton序列初始化整个种群
        self.population = self.init_pop(self.n_pop * self.n_fractals, 2, self.lower, self.upper)  # Halton初始化
        self.subpopulations = np.split(self.population, self.n_fractals)

        for i in range(self.n_fractals):
            # 计算每个分形内所有粒子的适应度值
            fitness = np.apply_along_axis(self.func, 1, self.subpopulations[i])
            
            # 找到分形内的最优粒子作为centroid
            min_idx = np.argmin(fitness)
            self.centroids[i] = self.subpopulations[i][min_idx]  # 分型内的最优粒子

            # 假设最差粒子用于更新上下界
            max_idx = np.argmax(fitness)
            lub = self.subpopulations[i][max_idx]  # 最差粒子
            
            # 计算距离，并更新每个分形的上下界
            epsilon = 1e-10
            dist = np.abs(lub - self.centroids[i])  # 距离绝对值
            self.lower_bounds[i] = np.clip(self.centroids[i] - dist, self.lower, self.upper)  # 确保不超出全局上下界
            self.upper_bounds[i] = np.clip(self.centroids[i] + dist + epsilon, self.lower, self.upper)
            
            # 如果当前分形中的最优解比全局最优解更优，则更新全局最优解
            if fitness[min_idx] < self.global_best_fitness:
                self.global_best_fitness = fitness[min_idx]
                self.global_best_position = self.centroids[i]


    def init_subpopulation(self, npop, fractal_idx):
        """
        初始化分形中的种群
        :param npop: 种群规模
        :param fractal_idx: 分形索引
        :return: 初始化后的种群
        """
        seq_type = np.random.randint(1, 5)  # 随机选择序列类型
        #seq_type = 2  # Latin Hypercube
        # print("lower")
        # print(self.lower_bounds[fractal_idx])
        # print("upper")
        # print(self.upper_bounds[fractal_idx])   
        return self.init_pop(npop, seq_type, self.lower_bounds[fractal_idx], self.upper_bounds[fractal_idx])

    def adjust_to_power_of_two(self, n):
        """
        将 n 调整为最近的 2 的幂次方
        """
        return 2 ** np.ceil(np.log2(n)).astype(int)
    
    def init_pop(self, pop_size, seq_type, lower, upper):
        """
        根据序列类型初始化种群
        :param pop_size: 种群规模
        :param seq_type: 低差异序列类型
        :param lower: 当前分形的下界
        :param upper: 当前分形的上界
        :return: 初始化后的种群
        """
        if seq_type == 1:
            # Sobol序列，确保种群大小为 2 的幂次方
            pop_size = self.adjust_to_power_of_two(pop_size)
            sampler = qmc.Sobol(d=self.dim)
        elif seq_type == 2:
            # 拉丁超立方序列
            sampler = qmc.LatinHypercube(d=self.dim)
        elif seq_type == 3:
            # Halton序列
            sampler = qmc.Halton(d=self.dim)
        else:
            # Hammersley序列
            return self.hammersley_sequence(pop_size, lower, upper)
        
        # 生成采样点并缩放到指定的上下界
        sample = sampler.random(pop_size)
        return qmc.scale(sample, lower, upper)

    def hammersley_sequence(self, pop_size, lower, upper):
        """
        生成Hammersley序列
        :param pop_size: 种群大小
        :param lower: 下界
        :param upper: 上界
        :return: Hammersley序列的采样点
        """
        seq = np.zeros((pop_size, self.dim))
        
        # 生成第一个维度的序列，使用1/n的线性序列
        seq[:, 0] = np.arange(1, pop_size + 1) / pop_size
        
        # 生成其他维度的Hammersley序列
        for j in range(1, self.dim):
            base = self.get_prime(j)  # 获取维度对应的素数
            seq[:, j] = self.radical_inverse(base, np.arange(1, pop_size + 1))
        
        # 缩放采样点到指定的上下界
        return qmc.scale(seq, lower, upper)

    def radical_inverse(self, base, indices):
        """
        计算radical inverse，用于生成Hammersley序列的一个维度
        :param base: 基数（通常为素数）
        :param indices: 索引数组
        :return: 反向基数序列
        """
        result = np.zeros_like(indices, dtype=float)
        f = 1.0 / base
        i = indices
        while np.any(i > 0):
            result += f * (i % base)
            i = i // base
            f = f / base
        return result

    def get_prime(self, dim):
        """
        获取指定维度的素数
        :param dim: 维度
        :return: 该维度的素数
        """
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        return primes[dim % len(primes)]  # 使用素数列表


    def sigmoid_population_decrease(self, iteration):
        """
        计算种群规模的S形衰减
        :param iteration: 当前迭代次数
        :return: 当前迭代的种群规模
        """
        return int(1 + self.popIni - (1 / ((1 / (self.popIni - self.popFi)) + np.exp(-(self.popFi / self.max_iter) * iteration))))

    def update_subpopulation(self, fractal_idx, newpop, fitness):
        """
        更新分形种群中的粒子信息
        :param fractal_idx: 当前分形索引
        :param newpop: 新生成的种群
        :param fitness: 新生成种群的适应度
        """
        min_idx = np.argmin(fitness)
        self.subpopulations[fractal_idx] = newpop
        self.centroids[fractal_idx] = newpop[min_idx]  # 更新该分形的最优粒子为centroid
        new_centroid = self.centroids[fractal_idx] + 0.3 * (self.global_best_position - self.centroids[fractal_idx])
        self.centroids[fractal_idx] = new_centroid
        
        # 更新分型的搜索边界
        max_idx = np.argmax(fitness)
        lub = newpop[max_idx]  # 最差解
        dist = np.abs(lub - self.centroids[fractal_idx])  # 计算最优和最差粒子的距离
        
        epsilon = 1e-8
        
        # 更新每个维度的上下界
        self.lower_bounds[fractal_idx] = np.clip(self.centroids[fractal_idx] - dist, self.lower, self.upper)
        self.upper_bounds[fractal_idx] = np.clip(self.centroids[fractal_idx] + dist + epsilon, self.lower, self.upper)
        

    def update_global_best(self):
        """
        更新全局最优解
        """
        for fractal in self.subpopulations:
            for particle in fractal:
                fitness = self.func(particle)
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = copy.deepcopy(particle)


    def get_best_fitness_history(self):
        """
        返回全局最优适应度值的历史记录
        :return: 历史记录
        """
        return self.best_fitness_history
