import numpy as n
import numpy as np

class JSO:
    def __init__(self, fitness_fn, dim, min_bound, max_bound):
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.min_bound = min_bound
        self.max_bound = max_bound

        # jSO parameters
        self.memory_size = 5
        self.p_best_rate = 0.25
        self.arc_rate = 1.0
        self.epsilon = 1e-8
        self.optimum = 0.0
        self.max_evals = dim * 10000
        self.pop_size = int(round(np.sqrt(dim) * np.log(dim) * 25))
        self.min_pop_size = 4
        self.archive_size = int(round(self.pop_size * self.arc_rate))

        # Population
        self.population = self.initialize_population(self.pop_size)
        self.fitness = np.array([self.fitness_fn(ind) for ind in self.population])
        self.archive = []
        self.nfes = self.pop_size

        # Historical memories
        self.M_F = np.full(self.memory_size, 0.3)
        self.M_CR = np.full(self.memory_size, 0.8)
        self.memory_pos = 0

        # Best solution so far
        best_idx = np.argmin(self.fitness)
        self.best = self.population[best_idx].copy()
        self.best_fit = self.fitness[best_idx]

    def initialize_population(self, size):
        return [np.random.uniform(self.min_bound, self.max_bound, self.dim) for _ in range(size)]

    def bound(self, vec, parent):
        return np.where(vec < self.min_bound, (self.min_bound + parent) / 2,
                        np.where(vec > self.max_bound, (self.max_bound + parent) / 2, vec))

    def cauchy_g(self, mu, gamma):
        return mu + gamma * np.tan(np.pi * (np.random.rand() - 0.5))

    def gauss(self, mu, sigma):
        return mu + sigma * np.sqrt(-2.0 * np.log(np.random.rand())) * np.sin(2.0 * np.pi * np.random.rand())

    def run(self):
        arc = []
        pop = self.population
        fit = self.fitness
        dim = self.dim
        nfes = self.nfes
        max_evals = self.max_evals
        memory_size = self.memory_size
        best = self.best
        best_fit = self.best_fit
        M_F = self.M_F
        M_CR = self.M_CR
        memory_pos = self.memory_pos

        archive = []
        success_sf = []
        success_cr = []
        delta_f = []

        while nfes < max_evals:
            sorted_idx = np.argsort(fit)
            p_num = max(2, int(round(self.p_best_rate * len(pop))))
            new_pop = []
            new_fit = []

            for i in range(len(pop)):
                r = np.random.randint(memory_size)
                mu_sf = 0.9 if r == memory_size - 1 else M_F[r]
                mu_cr = 0.9 if r == memory_size - 1 else M_CR[r]

                CR = np.clip(self.gauss(mu_cr, 0.1), 0.0, 1.0)
                if nfes < 0.25 * max_evals and CR < 0.7:
                    CR = 0.7
                elif nfes < 0.5 * max_evals and CR < 0.6:
                    CR = 0.6

                while True:
                    F = self.cauchy_g(mu_sf, 0.1)
                    if F > 0:
                        break
                F = min(F, 1.0)
                if nfes < 0.6 * max_evals and F > 0.7:
                    F = 0.7

                pbest_idx = sorted_idx[np.random.randint(p_num)]
                while nfes < 0.5 * max_evals and pbest_idx == i:
                    pbest_idx = sorted_idx[np.random.randint(p_num)]

                xi = pop[i]
                xp = pop[pbest_idx]
                r1 = np.random.randint(len(pop))
                while r1 == i:
                    r1 = np.random.randint(len(pop))

                r2_total = len(pop) + len(archive)
                r2 = np.random.randint(r2_total)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(r2_total)

                xr1 = pop[r1]
                xr2 = archive[r2 - len(pop)] if r2 >= len(pop) else pop[r2]

                F_j = F * (0.7 if nfes < 0.2 * max_evals else 0.8 if nfes < 0.4 * max_evals else 1.2)
                vi = xi + F_j * (xp - xi) + F * (xr1 - xr2)
                vi = self.bound(vi, xi)

                jrand = np.random.randint(dim)
                ui = np.array([vi[j] if (np.random.rand() < CR or j == jrand) else xi[j] for j in range(dim)])
                fi = self.fitness_fn(ui)
                nfes += 1

                if fi < fit[i]:
                    new_pop.append(ui)
                    new_fit.append(fi)
                    archive.append(pop[i])
                    archive = archive[-self.archive_size:]
                    success_sf.append(F)
                    success_cr.append(CR)
                    delta_f.append(abs(fit[i] - fi))
                    if fi < best_fit:
                        best = ui
                        best_fit = fi
                else:
                    new_pop.append(pop[i])
                    new_fit.append(fit[i])

            # Update memory
            if success_sf:
                weights = np.array(delta_f)
                weights /= weights.sum()
                mean_sf = np.sum(weights * np.array(success_sf)**2) / np.sum(weights * np.array(success_sf))
                mean_cr = np.sum(weights * np.array(success_cr)**2) / np.sum(weights * np.array(success_cr))
                M_F[memory_pos] = 0.5 * (M_F[memory_pos] + mean_sf)
                M_CR[memory_pos] = 0.5 * (M_CR[memory_pos] + mean_cr)
                memory_pos = (memory_pos + 1) % memory_size
                success_sf.clear()
                success_cr.clear()
                delta_f.clear()

            # Population size reduction
            plan_size = int(round(((self.min_pop_size - self.pop_size) / max_evals) * nfes + self.pop_size))
            if len(new_pop) > plan_size:
                losses = np.argsort(new_fit)[::-1]
                keep = sorted(losses[reduction:] if (reduction := len(new_pop) - plan_size) > 0 else [])
                new_pop = [new_pop[i] for i in range(len(new_pop)) if i not in keep]
                new_fit = [new_fit[i] for i in range(len(new_fit)) if i not in keep]

            pop = new_pop
            fit = np.array(new_fit)

        return best_fit


def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

jso = JSO(fitness_fn=rastrigin, dim=30, min_bound=-5.12, max_bound=5.12)
best = jso.run()
print("Best fitness:", best)