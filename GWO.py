import numpy as np
from tqdm import tqdm
import copy
import math


np.random.seed(42)

class GWO:
    def __init__(self, func, dim, lower, upper, n_pop=150, max_iter=1000):
        """
        初始化GWO算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界
        :param upper: 搜索空间上界
        :param n_pop: 种群规模（灰狼个体数量）
        :param max_iter: 最大迭代次数
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.population = np.random.uniform(lower, upper, (n_pop, dim))
        self.best_fitness_history = []



    def optimize(self):
        """
        优化过程
        :return: 最优解及其对应的适应度值
        """
        # 初始化α、β、δ狼的位置
        alpha, beta, delta = self.initialize_leaders()
        
        for iteration in tqdm(range(self.max_iter), desc="GWO Progress"):
            # 线性递减系数a从2线性递减到0
            a = 2 - 2 * (iteration / (self.max_iter - 1))
            
            # 更新每个个体的位置，包括头狼
            for i in range(self.n_pop):
                self.population[i] = self.update_position(self.population[i], alpha, beta, delta, a)
                
            # 限制个体的位置在搜索空间内
            self.population = np.clip(self.population, self.lower, self.upper)
            
            # 计算每个个体的适应度值
            fitness = np.apply_along_axis(self.func, 1, self.population)
            
            # 更新α、β、δ狼
            alpha, beta, delta = self.update_leaders(fitness, alpha, beta, delta)
            
            # 保存当前迭代的最佳适应度值
            self.best_fitness_history.append(self.func(alpha))
            
        return alpha, self.func(alpha)


    def get_best_fitness_history(self):
        return self.best_fitness_history

    def initialize_leaders(self):
        """
        初始化α、β、δ狼的位置
        :return: α、β、δ狼的位置
        """
        alpha = np.zeros(self.dim)
        beta = np.zeros(self.dim)
        delta = np.zeros(self.dim)
        return alpha, beta, delta

    def update_leaders(self, fitness, alpha, beta, delta):
        """
        更新α、β、δ狼的位置
        :param fitness: 当前种群的适应度值
        :param alpha: 当前α狼的位置
        :param beta: 当前β狼的位置
        :param delta: 当前δ狼的位置
        :return: 更新后的α、β、δ狼的位置
        """
        for i in range(self.n_pop):
            if fitness[i] < self.func(alpha):
                alpha = copy.deepcopy(self.population[i])
            elif fitness[i] < self.func(beta):
                beta = copy.deepcopy(self.population[i])
            elif fitness[i] < self.func(delta):
                delta = copy.deepcopy(self.population[i])
        return alpha, beta, delta

    def update_position(self, position, alpha, beta, delta, a):
        """
        根据α、β、δ狼的位置更新个体位置
        :param position: 当前个体的位置
        :param alpha: α狼的位置
        :param beta: β狼的位置
        :param delta: δ狼的位置
        :param a: 线性递减系数
        :return: 更新后的个体位置
        """
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2

        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2

        D_alpha = abs(C1 * alpha - position)
        D_beta = abs(C2 * beta - position)
        D_delta = abs(C3 * delta - position)

        X1 = alpha - A1 * D_alpha
        X2 = beta - A2 * D_beta
        X3 = delta - A3 * D_delta

        return (X1 + X2 + X3) / 3

class AGWO:
    def __init__(self, func, dim, lower, upper, n_pop=150, max_iter=1000):
        """
        初始化AGWO算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界（标量或数组）
        :param upper: 搜索空间上界（标量或数组）
        :param n_pop: 种群规模（灰狼个体数量）
        :param max_iter: 最大迭代次数
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.population = np.random.uniform(self.lower, self.upper, (n_pop, dim))
        self.best_fitness_history = []
        self.position_history = np.zeros((n_pop, max_iter, dim))
        self.fitness_history = np.zeros((n_pop, max_iter))
    
    def optimize(self):
        """
        优化过程
        :return: 最优解及其对应的适应度值
        """
        # 初始化α、β、δ狼的位置和分数
        Alpha_pos = np.zeros(self.dim)
        Alpha_score = float('inf')  # 对于最小化问题
        Beta_pos = np.zeros(self.dim)
        Beta_score = float('inf')
        Delta_pos = np.zeros(self.dim)
        Delta_score = float('inf')
        
        for iteration in tqdm(range(1, self.max_iter + 1), desc="AGWO Progress"):
            # 更新参数a
            a = np.cos((np.pi / 2) * ((iteration / self.max_iter) ** 4))
            
            for i in range(self.n_pop):
                # 边界检查
                self.population[i] = np.clip(self.population[i], self.lower, self.upper)
                
                # 计算适应度
                fitness = self.func(self.population[i])
                self.fitness_history[i, iteration - 1] = fitness
                
                # 更新α、β、δ狼
                if fitness < Alpha_score:
                    Alpha_score = fitness
                    Alpha_pos = copy.deepcopy(self.population[i])
                elif fitness > Alpha_score and fitness < Beta_score:
                    Beta_score = fitness
                    Beta_pos = copy.deepcopy(self.population[i])
                elif fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                    Delta_score = fitness
                    Delta_pos = copy.deepcopy(self.population[i])
                
                # 保存位置历史
                self.position_history[i, iteration - 1, :] = self.population[i]
            
            # 更新每个个体的位置
            for i in range(self.n_pop):
                self.population[i] = self.update_position(copy.deepcopy(self.population[i]), Alpha_pos, Beta_pos, Delta_pos, a, iteration)
            
            # 保存当前迭代的最佳适应度值
            self.best_fitness_history.append(Alpha_score)
        
        return Alpha_pos, Alpha_score
    
    def update_position(self, position, Alpha_pos, Beta_pos, Delta_pos, a, iteration):
        """
        根据AGWO的规则更新个体位置
        :param position: 当前个体的位置
        :param Alpha_pos: α狼的位置
        :param Beta_pos: β狼的位置
        :param Delta_pos: δ狼的位置
        :param a: 动态调整的参数
        :param iteration: 当前迭代次数
        :return: 更新后的个体位置
        """
        new_position = copy.deepcopy(position)
        for j in range(self.dim):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * Alpha_pos[j] - position[j])
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * Beta_pos[j] - position[j])
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * Delta_pos[j] - position[j])
            
            if np.random.rand() < 0.8:
                new_position[j] = (0.5 * (Alpha_pos[j] - A1 * D_alpha) +
                                   0.3 * (Beta_pos[j] - A2 * D_beta) +
                                   0.2 * (Delta_pos[j] - A3 * D_delta))
            else:
                if np.random.rand() < 0.5:
                    mean_pos_j = np.mean(self.population[:, j])
                    new_position[j] = (Alpha_pos[j] * (1 - iteration / self.max_iter) +
                                       (mean_pos_j - Alpha_pos[j]) * np.random.rand())
                else:
                    # Levy飞行
                    beta_levy = 1.5  # 常用值
                    sigma = (math.gamma(1 + beta_levy) * np.sin(np.pi * beta_levy / 2) /
                             (math.gamma((1 + beta_levy) / 2) * beta_levy * 2 ** ((beta_levy - 1) / 2))) ** (1 / beta_levy)
                    u = np.random.normal(0, sigma)
                    v = np.random.normal(0, 1)
                    step = u / (abs(v) ** (1 / beta_levy))
                    step_size = 0.01 * step * (position[j] - Alpha_pos[j])
                    new_position[j] = position[j] + step_size
        return new_position
    
    def get_best_fitness_history(self):
        return self.best_fitness_history

class LCGWO8:
    # 改进自LCGWO8，增加参数b，先增大后减小。在circle movement中加入随机的小步移动
    def __init__(self, func, dim, lower, upper, n_pop=150, max_iter=1000):
        """
        初始化 MLL_GWO 算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界（标量或数组）
        :param upper: 搜索空间上界（标量或数组）
        :param n_pop: 种群规模（灰狼个体数量）
        :param max_iter: 最大迭代次数
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.n_layers = 3  # 固定层数为三层
        self.population = np.random.uniform(self.lower, self.upper, (n_pop, dim))
        self.best_fitness_history = []
        self.best_fitness = np.inf
        self.best_pos = np.zeros(self.dim)
        self.second_best_fitness = np.inf
        self.second_best_pos = np.zeros(self.dim)
        self.third_best_fitness = np.inf
        self.third_best_pos = np.zeros(self.dim)
        self.theta = np.random.uniform(0, 2 * np.pi, n_pop)  # 初始化每只狼的角度

    def optimize(self):
        """
        优化过程
        :return: 最优解及其对应的适应度值
        """
        for iteration in tqdm(range(self.max_iter), desc="LCGWO8 Progress"):
            a = self.compute_a(iteration)

            # 计算每个个体的适应度值
            fitness = np.apply_along_axis(self.func, 1, self.population)

            # 根据适应度值对种群进行排序，并划分层次
            sorted_fitness_indices = np.argsort(fitness)
            layers = np.array_split(sorted_fitness_indices, self.n_layers)

            # 选择每个层次的领导者
            leaders = [self.population[layer[0]] for layer in layers]

            # 计算种群平均位置
            X_mean = np.mean(self.population, axis=0)

            position = copy.deepcopy(self.population)
            # 层内位置更新
            for i in range(self.n_layers):
                for j in layers[i]:
                    if j != layers[i][0]:  # 跳过领导者自身
                        if i == 0:
                            # 第一层：Levy 飞行更新位置
                            self.population[j] = self.levy_flight(position[j], self.best_pos, iteration)
                        elif i == 1:
                            # 第二层：使用绕圈公式更新位置
                            #self.population[j] = self.circle_movement(position[j], leaders[2], X_mean, a, j)
                            self.population[j] = self.update_position_gwo(position[j], self.best_pos, self.second_best_pos, self.third_best_pos, a)
                        elif i == 2:
                            # 第三层：使用所有层的领导者更新位置
                            #self.population[j] = self.update_position_gwo(position[j], self.best_pos, self.second_best_pos, self.third_best_pos, a)
                            self.population[j] = self.circle_movement(position[j], leaders[2], X_mean, iteration, j)
                        # 边界检查
                        self.population[j] = np.clip(self.population[j], self.lower, self.upper)

            # 保存当前迭代的最佳适应度值，更新全局最佳值
            for indice in (sorted_fitness_indices[0], sorted_fitness_indices[1], sorted_fitness_indices[2]):
                if (fitness[indice] < self.best_fitness):
                    self.best_fitness = fitness[indice]
                    self.best_pos = copy.deepcopy(self.population[indice])
                elif (fitness[indice] < self.second_best_fitness):
                    self.second_best_fitness = fitness[indice]
                    self.second_best_pos = copy.deepcopy(self.population[indice])
                elif (fitness[indice] < self.third_best_fitness):
                    self.third_best_pos = copy.deepcopy(self.population[indice])
            self.best_fitness_history.append(self.best_fitness)

        return self.best_pos, self.best_fitness

    def get_best_fitness_history(self):
        return self.best_fitness_history

    def levy_flight(self, position, leader, iteration):
        """
        使用 Levy 飞行更新位置
        :param position: 当前个体位置
        :param leader: 当前层的领导者位置
        :return: 更新后的个体位置
        """
        alpha = 1.0
        beta_max = 1.9
        k = 0.1
        mid = 0.5
        # beta = 1.5
        beta =  alpha + (beta_max - alpha) / (1 + np.exp(-k * (iteration - mid * self.max_iter)))
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (np.abs(v) ** (1 / beta) + 1e-10)
        step_size = 0.01 * step * (position - leader)
        new_position = position + step_size
        new_position = np.clip(new_position, self.lower, self.upper)
        return new_position
    def compute_a(self, iteration):
        return np.cos((np.pi / 2) * ((iteration / self.max_iter) ** 4))
    def compute_b(self, iteration):
        # b从0.2增大到1，再减小到0.2
        mid_point = self.max_iter / 2
        if iteration < mid_point:
            return 0.2 + 0.8 * (np.sin((iteration / mid_point) * (np.pi / 2)))
        else:
            return 1.0 * (np.cos(((iteration - mid_point) / mid_point) * (np.pi / 2)))

    def circle_movement(self, position, leader, X_mean, iteration, wolf_index):
        X_target = leader
        Delta = X_target - position
        r = np.linalg.norm(Delta)

        # Compute parameter 'a' (cosine-based) and 'b' (increasing then decreasing)
        a = self.compute_a(iteration)
        b = self.compute_b(iteration)

        # Randomly decide to make the movement shorter in the early phase
        if iteration < self.max_iter * 0.3 and np.random.rand() < 0.5:
            alpha = 0.3 * (1 - a)
            beta = 0.3 * b
        else:
            alpha = 1 - a
            beta = b

        omega = np.pi / 6
        self.theta[wolf_index] += omega  # Using a single wolf for demonstration

        # Generate random unit direction u
        u = np.random.randn(self.dim)
        u = u / (np.linalg.norm(u) + 1e-10)

        # Update position
        new_position = position + alpha * Delta + beta * r * u * np.cos(self.theta[wolf_index])

        # 应用小波变换
        
        dilation_param = 2 * b + 1e-7
        sigma = (1 / np.sqrt(dilation_param)) * np.exp(-(new_position / dilation_param) ** 2) * np.cos(5 * np.pi * (new_position / dilation_param))
        new_position = new_position + sigma * (self.upper - self.lower) * np.random.random(self.dim)

        # Ensure new position is within boundaries
        new_position = np.clip(new_position, self.lower, self.upper)

        return new_position


    def update_position_gwo(self, position, leader1, leader2, leader3, a):
        """
        使用所有层的领导者更新位置（类似于 GWO 中的 Alpha、Beta、Delta 狼）
        :param position: 当前个体位置
        :param leader1: 第一层的领导者（最优个体）
        :param leader2: 第二层的领导者
        :param leader3: 第三层的领导者
        :param a: 动态调整的参数
        :return: 更新后的个体位置
        """
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_leader1 = abs(C1 * leader1 - position)
        X1 = leader1 - A1 * D_leader1

        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_leader2 = abs(C2 * leader2 - position)
        X2 = leader2 - A2 * D_leader2

        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_leader3 = abs(C3 * leader3 - position)
        X3 = leader3 - A3 * D_leader3

        # 取三个位置的平均值
        new_position = 0.5 * X1 + 0.3 * X2 + 0.2 * X3
        new_position = np.clip(new_position, self.lower, self.upper)
        return new_position

from scipy.stats import qmc

class LMGWO4: 
    def __init__(self, func, dim, lower, upper, n_pop=150, max_iter=1000):
        """
        初始化 MLL_GWO 算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界（标量或数组）
        :param upper: 搜索空间上界（标量或数组）
        :param n_pop: 种群规模（灰狼个体数量）
        :param max_iter: 最大迭代次数
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.n_layers = 3  # 固定层数为三层
        #self.population = np.random.uniform(self.lower, self.upper, (n_pop, dim))
        self.population = self.init_pop(self.n_pop, 2, self.lower, self.upper)  # Halton序列初始化
        self.best_fitness_history = []
        self.best_fitness = np.inf
        self.best_pos = np.zeros(self.dim)
        self.second_best_fitness = np.inf
        self.second_best_pos = np.zeros(self.dim)
        self.third_best_fitness = np.inf
        self.third_best_pos = np.zeros(self.dim)
        self.theta = np.random.uniform(0, 2 * np.pi, n_pop)  # 初始化每只狼的角度

    def optimize(self):
        """
        优化过程
        :return: 最优解及其对应的适应度值
        """
        for iteration in tqdm(range(self.max_iter), desc="LMGWO Progress"):
            a = np.cos((np.pi / 2) * ((iteration / self.max_iter) ** 4))

            # 计算每个个体的适应度值
            fitness = np.apply_along_axis(self.func, 1, self.population)

            # 根据适应度值对种群进行排序，并划分层次
            sorted_fitness_indices = np.argsort(fitness)
            layers = np.array_split(sorted_fitness_indices, self.n_layers)

            # 选择每个层次的领导者
            leaders = [self.population[layer[0]] for layer in layers]

            # 计算种群平均位置
            X_mean = np.mean(self.population, axis=0)

            position = copy.deepcopy(self.population)
            # 层内位置更新
            for i in range(self.n_layers):
                for j in layers[i]:
                    if j != layers[i][0]:  # 跳过领导者自身
                        if i == 0:
                            # 第一层：Levy 飞行更新位置
                            self.population[j] = self.levy_flight(position[j], self.best_pos, iteration)
                        elif i == 1:
                            self.population[j] = self.update_position_gwo(position[j], self.best_pos, self.second_best_pos, self.third_best_pos, a)
                        elif i == 2:
                            self.population[j] = self.m_flight(position[j], self.best_pos)
                            #self.population[j] = self.circle_movement(position[j], leaders[2], X_mean, a, j)
                        # 边界检查
                        self.population[j] = np.clip(self.population[j], self.lower, self.upper)

            # 保存当前迭代的最佳适应度值，更新全局最佳值
            for indice in (sorted_fitness_indices[0], sorted_fitness_indices[1], sorted_fitness_indices[2]):
                if (fitness[indice] < self.best_fitness):
                    self.best_fitness = fitness[indice]
                    self.best_pos = copy.deepcopy(self.population[indice])
                elif (fitness[indice] < self.second_best_fitness):
                    self.second_best_fitness = fitness[indice]
                    self.second_best_pos = copy.deepcopy(self.population[indice])
                elif (fitness[indice] < self.third_best_fitness):
                    self.third_best_pos = copy.deepcopy(self.population[indice])
            self.best_fitness_history.append(self.best_fitness)

        return self.best_pos, self.best_fitness

    def get_best_fitness_history(self):
        return self.best_fitness_history
    
    import numpy as np

    def m_flight(self, position, leader):
        # 参数设置
        #scale = 0.25  # 用于调整步长，但不再影响 Beta 分布
        dim = self.dim
        left_alpha, left_beta = 2, 10  # 左侧 Beta 分布参数
        right_alpha, right_beta = 10, 2  # 右侧 Beta 分布参数

        # 生成左峰和右峰的数据 (将 Beta 分布的范围映射到 [-1, 1])
        left_peak = 2 * (np.random.beta(left_alpha, left_beta, size=dim // 2 + dim % 2) - 0.5)
        right_peak = 2 * (np.random.beta(right_alpha, right_beta, size=dim // 2) - 0.5)

        # 合并并打乱数据
        random_step = np.concatenate((left_peak, right_peak))
        # 随机选择一半的维度
        half_dim = self.dim // 2
        random_indices = np.random.choice(self.dim, half_dim, replace=False)
        
        # 将选中的维度设置为1
        random_step[random_indices] = 1
        np.random.shuffle(random_step)
        Diff = leader - position
        norm = np.linalg.norm(Diff)
        direction = Diff / norm

        # 随机生成一个向量
        random_direction = np.random.rand(*direction.shape)

        # 去除 random_direction 在 direction 方向上的分量，使其垂直于 direction
        tangent_direction = random_direction - np.dot(random_direction, direction) * direction
        tangent_direction = tangent_direction / np.linalg.norm(tangent_direction)  # 标准化
        # 将数据截断在[-1, 1]范围内
        random_step = np.clip(random_step, -1, 1)
        step = (0.3* random_step * tangent_direction + 5/self.max_iter) * Diff
        new_position = position + step
        # 边界检查
        new_position = np.clip(new_position, self.lower, self.upper)
        return new_position


    def levy_flight(self, position, leader, iteration):
        """
        使用 Levy 飞行更新位置
        :param position: 当前个体位置
        :param leader: 当前层的领导者位置
        :return: 更新后的个体位置
        """
        alpha = 1.0
        beta_max = 1.9
        k = 0.1
        mid = 0.5
        # beta = 1.5
        beta =  alpha + (beta_max - alpha) / (1 + np.exp(-k * (iteration - mid * self.max_iter)))
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (np.abs(v) ** (1 / beta) + 1e-10)
        step_size = 0.02 * step * (position - leader)
        new_position = position + step_size
        new_position = np.clip(new_position, self.lower, self.upper)
        return new_position

    def circle_movement(self, position, leader, X_mean, a, wolf_index):
        """
        使用小波变异和绕圈公式更新位置
        :param position: 当前个体位置
        :param leader: 当前层的领导者位置
        :param X_mean: 种群平均位置
        :param a: 当前控制参数 a
        :param wolf_index: 当前狼的索引
        :return: 更新后的个体位置 new_position
        """
        # 计算目标点
        # X_target = a * X_mean + (1 - a) * leader
        # X_target = (X_mean + leader) / 2
        X_target = leader
        # 计算径向向量和距离
        Delta = X_target - position
        r = np.linalg.norm(Delta)

        # 动态调整参数 alpha 和 beta
        #k = iteration / self.max_iter
        alpha = 1 - a  # alpha 从 0 递增到 1
        beta_max = 1
        beta = beta_max * a  # beta 从 beta_max 递减到 0

        # 更新角速度 omega
        omega = np.pi / 6
        self.theta[wolf_index] += omega  # 更新当前狼的角度

        # 生成随机单位向量 u
        u = np.random.randn(self.dim)
        u = u / (np.linalg.norm(u) + 1e-10)  # 防止除以零

        # 更新位置 (包含靠近目标和绕圈)
        new_position = position + alpha * Delta + beta * r * u * np.cos(self.theta[wolf_index])

        # 应用小波变异增加多样性
        dilation_param = 2 * a + 1e-7  # 控制小波变异的伸缩参数
        sigma = (1 / np.sqrt(dilation_param)) * np.exp(-(new_position / dilation_param) ** 2) * np.cos(5 * np.pi * (new_position / dilation_param))
        mutated_position = new_position + sigma * (self.upper - self.lower) * np.random.random(self.dim)

        # 确保新位置在边界内
        mutated_position = np.clip(mutated_position, self.lower, self.upper)

        return mutated_position  # 返回新的粒子位置
                   
    def adjust_to_power_of_two(self, n):
        """
        将 n 调整为最近的 2 的幂次方
        """
        return 2 ** np.ceil(np.log2(n)).astype(int)

    def init_pop(self, pop_size, seq_type, lower, upper):
        """
        根据序列类型初始化种群，并进行缓存。
        :param pop_size: 种群规模
        :param seq_type: 低差异序列类型（1 = Sobol, 2 = 拉丁超立方, 3 = Halton, 4 = Hammersley）
        :param lower: 当前分形的下界
        :param upper: 当前分形的上界
        :return: 初始化后的种群
        """
        
        # 确保有缓存字典
        if not hasattr(self, 'cached_sequences'):
            self.cached_sequences = {}
        
        # 生成序列的唯一键，包含序列类型、维度、种群规模
        cache_key = (seq_type, self.dim, pop_size)
        
        # 检查是否已经缓存了这个序列
        if cache_key not in self.cached_sequences:
            if seq_type == 1:
                # Sobol序列，确保种群大小为 2 的幂次方
                pop_size = self.adjust_to_power_of_two(pop_size)
                sampler = qmc.Sobol(d=self.dim)
                self.cached_sequences[cache_key] = sampler.random(pop_size)
            elif seq_type == 2:
                # 拉丁超立方序列
                sampler = qmc.LatinHypercube(d=self.dim)
                self.cached_sequences[cache_key] = sampler.random(pop_size)
            elif seq_type == 3:
                # Halton序列
                sampler = qmc.Halton(d=self.dim)
                self.cached_sequences[cache_key] = sampler.random(pop_size)
            elif seq_type == 4:
                # Hammersley序列生成 [0,1] 区间的采样点
                self.cached_sequences[cache_key] = self.hammersley_sequence(pop_size)
        
        # 从缓存中读取序列，并缩放到 [lower, upper] 区间
        sequence = self.cached_sequences[cache_key]
        return qmc.scale(sequence, lower, upper)
    def hammersley_sequence(self, pop_size):
        """
        生成 [0,1] 区间内的 Hammersley 序列。
        :param pop_size: 种群大小
        :return: Hammersley 序列在 [0,1] 区间的采样点
        """
        seq = np.zeros((pop_size, self.dim))
        
        # 生成第一个维度的序列，使用 1/n 的线性序列
        seq[:, 0] = np.arange(1, pop_size + 1) / pop_size
        
        # 生成其他维度的 Hammersley 序列
        for j in range(1, self.dim):
            base = self.get_prime(j)  # 获取维度对应的素数
            seq[:, j] = self.radical_inverse(base, np.arange(1, pop_size + 1))
        
        return seq  # 返回生成的 [0,1] 区间序列

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
        return primes[dim % len(primes)]  # 使用素数列表中的值
    

    def update_position_gwo(self, position, leader1, leader2, leader3, a):
        """
        使用所有层的领导者更新位置（类似于 GWO 中的 Alpha、Beta、Delta 狼）
        :param position: 当前个体位置
        :param leader1: 第一层的领导者（最优个体）
        :param leader2: 第二层的领导者
        :param leader3: 第三层的领导者
        :param a: 动态调整的参数
        :return: 更新后的个体位置
        """
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_leader1 = abs(C1 * leader1 - position)
        X1 = leader1 - A1 * D_leader1

        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_leader2 = abs(C2 * leader2 - position)
        X2 = leader2 - A2 * D_leader2

        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_leader3 = abs(C3 * leader3 - position)
        X3 = leader3 - A3 * D_leader3

        # 取三个位置的平均值
        new_position = 0.5 * X1 + 0.3 * X2 + 0.2 * X3
        new_position = np.clip(new_position, self.lower, self.upper)
        return new_position

class MELGWO:
    def __init__(self, func, dim, lower, upper, n_pop=150, max_iter=1000, max_fe=50000):
        """
        初始化MELGWO算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界
        :param upper: 搜索空间上界
        :param n_pop: 种群规模（灰狼个体数量）
        :param max_iter: 最大迭代次数
        :param max_fe: 最大函数评估次数
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.max_fe = max_fe
        self.population = np.random.uniform(lower, upper, (n_pop, dim))
        self.best_fitness_history = []
        self.Fmin = 0.10
        self.Fmax = 2.00
        self.pCR = 0.60
        self.min_agents_no = 10
        self.max_agents_no = n_pop
        self.fe = 0  # Function evaluation counter

    def optimize(self):
        """
        优化过程
        :return: 最优解及其对应的适应度值
        """
        # 初始化α、β、δ狼的位置和适应度值
        alpha, beta, delta = self.initialize_leaders()
        alpha_fitness, beta_fitness, delta_fitness = float('inf'), float('inf'), float('inf')
        
        memory_positions = copy.deepcopy(self.population)
        memory_fitness = np.apply_along_axis(self.func, 1, self.population)
        self.fe += self.n_pop
        
        for iteration in tqdm(range(self.max_iter), desc="MELGWO Progress"):
            # 线性递减系数a从2线性递减到0
            a = 2 - 2 * (iteration / self.max_iter)
            
            # 更新每个个体的位置
            for i in range(self.n_pop):
                self.population[i] = self.update_position(self.population[i], alpha, beta, delta, a)
            
            # 限制个体的位置在搜索空间内
            self.population = np.clip(self.population, self.lower, self.upper)
            
            # 计算每个个体的适应度值
            fitness = np.apply_along_axis(self.func, 1, self.population)
            self.fe += self.n_pop
            
            # 更新α、β、δ狼
            alpha, beta, delta, alpha_fitness, beta_fitness, delta_fitness = self.update_leaders(fitness, alpha, beta, delta, alpha_fitness, beta_fitness, delta_fitness)

            # 差分进化操作
            self.differential_evolution(memory_positions, memory_fitness, alpha, a)

            # 更新记忆群体
            for i in range(self.n_pop):
                if fitness[i] < memory_fitness[i]:
                    memory_positions[i] = copy.deepcopy(self.population[i])
                    memory_fitness[i] = fitness[i]

            # 局部搜索操作
            self.local_search(memory_positions, memory_fitness, alpha, beta, delta)

            # 种群缩减策略
            self.population_size_reduction(iteration)

            # 保存当前迭代的最佳适应度值
            self.best_fitness_history.append(alpha_fitness)
            
            if self.fe >= self.max_fe:
                break
        
        return alpha, alpha_fitness

    def get_best_fitness_history(self):
        return self.best_fitness_history
    
    def initialize_leaders(self):
        """
        初始化α、β、δ狼的位置
        :return: α、β、δ狼的位置
        """
        alpha = np.zeros(self.dim)
        beta = np.zeros(self.dim)
        delta = np.zeros(self.dim)
        return alpha, beta, delta

    def update_leaders(self, fitness, alpha, beta, delta, alpha_fitness, beta_fitness, delta_fitness):
        """
        更新α、β、δ狼的位置和适应度值
        :param fitness: 当前种群的适应度值
        :param alpha: 当前α狼的位置
        :param beta: 当前β狼的位置
        :param delta: 当前δ狼的位置
        :param alpha_fitness: 当前α狼的适应度值
        :param beta_fitness: 当前β狼的适应度值
        :param delta_fitness: 当前δ狼的适应度值
        :return: 更新后的α、β、δ狼的位置和适应度值
        """
        for i in range(self.n_pop):
            if fitness[i] < alpha_fitness:
                delta_fitness = beta_fitness
                delta = copy.deepcopy(beta)
                beta_fitness = alpha_fitness
                beta = copy.deepcopy(alpha)
                alpha_fitness = fitness[i]
                alpha = copy.deepcopy(self.population[i])
            elif fitness[i] < beta_fitness:
                delta_fitness = beta_fitness
                delta = copy.deepcopy(beta)
                beta_fitness = fitness[i]
                beta = copy.deepcopy(self.population[i])
            elif fitness[i] < delta_fitness:
                delta_fitness = fitness[i]
                delta = copy.deepcopy(self.population[i])
        return alpha, beta, delta, alpha_fitness, beta_fitness, delta_fitness

    def update_position(self, position, alpha, beta, delta, a):
        """
        根据α、β、δ狼的位置更新个体位置
        :param position: 当前个体的位置
        :param alpha: α狼的位置
        :param beta: β狼的位置
        :param delta: δ狼的位置
        :param a: 线性递减系数
        :return: 更新后的个体位置
        """
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2

        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2

        D_alpha = abs(C1 * alpha - position)
        D_beta = abs(C2 * beta - position)
        D_delta = abs(C3 * delta - position)

        X1 = alpha - A1 * D_alpha
        X2 = beta - A2 * D_beta
        X3 = delta - A3 * D_delta

        return (X1 + X2 + X3) / 3

    def differential_evolution(self, memory_positions, memory_fitness, alpha, a):
        """
        差分进化操作
        :param memory_positions: 记忆群体的位置
        :param memory_fitness: 记忆群体的适应度值
        :param alpha: α狼的位置
        :param a: 线性递减系数
        """
        for i in range(self.n_pop):
            F = self.Fmin + (self.Fmax - self.Fmin) * ((self.max_iter - a) / self.max_iter)
            x = memory_positions[i]
            y = x + F * (alpha - x)

            # 交叉操作
            z = np.zeros_like(x)
            j0 = np.random.randint(self.dim)
            for j in range(self.dim):
                if j == j0 or np.random.rand() <= self.pCR:
                    z[j] = y[j]
                else:
                    z[j] = x[j]

            new_position = np.clip(z, self.lower, self.upper)
            new_fitness = self.func(new_position)
            self.fe += 1

            # 贪婪选择
            if new_fitness < memory_fitness[i]:
                memory_positions[i] = copy.deepcopy(new_position)
                memory_fitness[i] = new_fitness
                
    def local_search(self, memory_positions, memory_fitness, alpha, beta, delta):
        """
        局部搜索操作
        :param memory_positions: 记忆群体的位置
        :param memory_fitness: 记忆群体的适应度值
        :param alpha: α狼的位置
        :param beta: β狼的位置
        :param delta: δ狼的位置
        """
        for k in range(self.n_pop // 2):
            distances = np.linalg.norm(memory_positions - memory_positions[k], axis=1)
            nearest_idx = np.argsort(distances)[1]
            if memory_fitness[nearest_idx] < memory_fitness[k]:
                nbest1 = memory_positions[nearest_idx]
                nbest2 = memory_positions[k]
            else:
                nbest1 = memory_positions[k]
                nbest2 = memory_positions[nearest_idx]
            
            temp_pbest = memory_positions[k] + 2 * np.random.rand(self.dim) * (nbest1 - nbest2)
            temp_pbest = np.clip(temp_pbest, self.lower, self.upper)
            temp_fitness = self.func(temp_pbest)
            self.fe += 1
            
            if temp_fitness < memory_fitness[k]:
                memory_positions[k] = copy.deepcopy(temp_pbest)
                memory_fitness[k] = temp_fitness

    def population_size_reduction(self, iteration):
        """
        种群大小缩减策略
        :param iteration: 当前迭代次数
        """
        planned_agents_no = int(self.min_agents_no + (self.max_agents_no - self.min_agents_no) * (self.max_iter - iteration) / self.max_iter)
        if self.n_pop > planned_agents_no:
            reduction = self.n_pop - planned_agents_no
            self.n_pop -= reduction
            sorted_indices = np.argsort(np.apply_along_axis(self.func, 1, self.population))
            self.population = self.population[sorted_indices[:self.n_pop]]
    
class SOGWO:
    def __init__(self, func, dim, lower, upper, n_pop=150, max_iter=1000):
        """
        初始化SOGWO算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界
        :param upper: 搜索空间上界
        :param n_pop: 种群规模（灰狼个体数量）
        :param max_iter: 最大迭代次数
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.population = np.random.uniform(lower, upper, (n_pop, dim))
        self.best_fitness_history = []
        self.upper_bound = np.full(dim, upper)
        self.lower_bound = np.full(dim, lower)

    def optimize(self):
        """
        优化过程
        :return: 最优解及其对应的适应度值
        """
        # 初始化α、β、δ狼的位置
        alpha, beta, delta = self.initialize_leaders()
        
        for iteration in tqdm(range(self.max_iter), desc="SOGWO Progress"):
            # 线性递减系数a从2线性递减到0
            a = 2 - 2 * (iteration / (self.max_iter - 1))
            
            # 对最差个体执行选择性反向学习
            self.population = self.opposition_learning(self.population, a)
            
            # 更新每个个体的位置，包括头狼
            for i in range(self.n_pop):
                self.population[i] = self.update_position(self.population[i], alpha, beta, delta, a)
                
            # 限制个体的位置在搜索空间内
            self.population = np.clip(self.population, self.lower, self.upper)
            
            # 计算每个个体的适应度值
            fitness = np.apply_along_axis(self.func, 1, self.population)
            
            # 更新α、β、δ狼
            alpha, beta, delta = self.update_leaders(fitness, alpha, beta, delta)
            
            # 保存当前迭代的最佳适应度值
            self.best_fitness_history.append(self.func(alpha))
            
        return alpha, self.func(alpha)

    def get_best_fitness_history(self):
        return self.best_fitness_history

    def initialize_leaders(self):
        """
        初始化α、β、δ狼的位置
        :return: α、β、δ狼的位置
        """
        alpha = np.zeros(self.dim)
        beta = np.zeros(self.dim)
        delta = np.zeros(self.dim)
        return alpha, beta, delta

    def update_leaders(self, fitness, alpha, beta, delta):
        """
        更新α、β、δ狼的位置
        :param fitness: 当前种群的适应度值
        :param alpha: 当前α狼的位置
        :param beta: 当前β狼的位置
        :param delta: 当前δ狼的位置
        :return: 更新后的α、β、δ狼的位置
        """
        for i in range(self.n_pop):
            if fitness[i] < self.func(alpha):
                alpha = copy.deepcopy(self.population[i])
            elif fitness[i] < self.func(beta):
                beta = copy.deepcopy(self.population[i])
            elif fitness[i] < self.func(delta):
                delta = copy.deepcopy(self.population[i])
        return alpha, beta, delta

    def update_position(self, position, alpha, beta, delta, a):
        """
        根据α、β、δ狼的位置更新个体位置
        :param position: 当前个体的位置
        :param alpha: α狼的位置
        :param beta: β狼的位置
        :param delta: δ狼的位置
        :param a: 线性递减系数
        :return: 更新后的个体位置
        """
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2

        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2

        D_alpha = abs(C1 * alpha - position)
        D_beta = abs(C2 * beta - position)
        D_delta = abs(C3 * delta - position)

        X1 = alpha - A1 * D_alpha
        X2 = beta - A2 * D_beta
        X3 = delta - A3 * D_delta

        return (X1 + X2 + X3) / 3

    def opposition_learning(self, population, a):
        """
        对适应度较差的个体执行选择性反向学习
        :param population: 当前种群的位置
        :param a: 线性递减系数，用于决定阈值
        :return: 反向学习后的位置
        """
        threshold = a  # 基于a设置阈值
        for i in range(self.n_pop):
            for j in range(self.dim):
                diff = self.upper_bound[j] - self.lower_bound[j]
                if np.random.rand() < threshold:
                    population[i, j] = self.upper_bound[j] + self.lower_bound[j] - population[i, j]
        return population

