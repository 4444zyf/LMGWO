import numpy as np
import copy
from tqdm import tqdm

class ChOA:
    # 黑猩猩优化算法（Chimpanzee Optimization Algorithm, ChOA）
    def __init__(self, func, dim, lower, upper, n_pop=100, max_iter=100):
        """
        初始化ChOA算法参数
        :param func: 优化目标函数
        :param dim: 问题的维度
        :param lower: 搜索空间下界
        :param upper: 搜索空间上界
        :param n_pop: 种群规模（搜索代理个体数量）
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
        # 初始化攻击者、屏障者、追逐者和驱动者的位置
        attacker, barrier, chaser, driver = self.initialize_leaders()
        
        for iteration in tqdm(range(self.max_iter), desc="ChOA Progress"):
            # 线性递减系数f从2线性递减到0
            f = 2 - iteration * (2 / self.max_iter)
            
            # 更新每个个体的位置
            for i in range(self.n_pop):
                self.population[i] = self.update_position(self.population[i], attacker, barrier, chaser, driver, f)
            
            # 限制个体的位置在搜索空间内
            self.population = np.clip(self.population, self.lower, self.upper)
            
            # 计算每个个体的适应度值
            fitness = np.apply_along_axis(self.func, 1, self.population)
            
            # 更新攻击者、屏障者、追逐者和驱动者
            attacker, barrier, chaser, driver = self.update_leaders(fitness, attacker, barrier, chaser, driver)
            
            # 保存当前迭代的最佳适应度值
            self.best_fitness_history.append(self.func(attacker))
        
        return attacker, self.func(attacker)

    def get_best_fitness_history(self):
        return self.best_fitness_history

    def initialize_leaders(self):
        """
        初始化攻击者、屏障者、追逐者和驱动者的位置
        :return: 攻击者、屏障者、追逐者和驱动者的位置
        """
        attacker = np.ones(self.dim)
        barrier = np.ones(self.dim)
        chaser = np.ones(self.dim)
        driver = np.ones(self.dim)
        return attacker, barrier, chaser, driver

    def update_leaders(self, fitness, attacker, barrier, chaser, driver):
        """
        更新攻击者、屏障者、追逐者和驱动者的位置
        :param fitness: 当前种群的适应度值
        :param attacker: 当前攻击者的位置
        :param barrier: 当前屏障者的位置
        :param chaser: 当前追逐者的位置
        :param driver: 当前驱动者的位置
        :return: 更新后的攻击者、屏障者、追逐者和驱动者的位置
        """
        for i in range(self.n_pop):
            if fitness[i] < self.func(attacker):
                attacker = copy.deepcopy(self.population[i])
            elif fitness[i] < self.func(barrier):
                barrier = copy.deepcopy(self.population[i])
            elif fitness[i] < self.func(chaser):
                chaser = copy.deepcopy(self.population[i])
            elif fitness[i] < self.func(driver):
                driver = copy.deepcopy(self.population[i])
        return attacker, barrier, chaser, driver

    def update_position(self, position, attacker, barrier, chaser, driver, f):
        """
        根据攻击者、屏障者、追逐者和驱动者的位置更新个体位置
        :param position: 当前个体的位置
        :param attacker: 攻击者的位置
        :param barrier: 屏障者的位置
        :param chaser: 追逐者的位置
        :param driver: 驱动者的位置
        :param f: 线性递减系数
        :return: 更新后的个体位置
        """
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * f * r1 - f
        C1 = 2 * r2
        
        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * f * r1 - f
        C2 = 2 * r2
        
        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * f * r1 - f
        C3 = 2 * r2
        
        r1, r2 = np.random.rand(), np.random.rand()
        A4 = 2 * f * r1 - f
        C4 = 2 * r2

        D_attacker = abs(C1 * attacker - position)
        D_barrier = abs(C2 * barrier - position)
        D_chaser = abs(C3 * chaser - position)
        D_driver = abs(C4 * driver - position)

        X1 = attacker - A1 * D_attacker
        X2 = barrier - A2 * D_barrier
        X3 = chaser - A3 * D_chaser
        X4 = driver - A4 * D_driver

        return (X1 + X2 + X3 + X4) / 4
