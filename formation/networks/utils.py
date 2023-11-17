import numpy as np

def add_list_elements(list1, list2):
    return [l1+l2 for l1, l2 in zip(list1, list2)]


class Step:
    # Time-dependent steps
    @staticmethod
    def step_t(f, t, x, h=1):
        raise NotImplementedError()

    # Time-independent steps
    @staticmethod
    def step(f, x, h=1):
        raise NotImplementedError()


class RK4(Step):
    @staticmethod
    def step_t(f, t, x, h=1):
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t, x + h * k3)

        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def step(f, x, h=1):
        dims_x = [len(x_i) for x_i in x]
        k1 = f(x)
        k2 = f(add_list_elements(x, [h / 2 * k1[sum(dims_x[:i]):sum(dims_x[:i+1])].flatten() for i in range(len(dims_x))]))
        k3 = f(add_list_elements(x, [h / 2 * k2[sum(dims_x[:i]):sum(dims_x[:i+1])].flatten() for i in range(len(dims_x))]))
        k4 = f(add_list_elements(x, [h * k3[sum(dims_x[:i]):sum(dims_x[:i+1])].flatten() for i in range(len(dims_x))]))
        
        x_vec = np.zeros((sum(dims_x), 1))
        for i, x_i in enumerate(x):
            x_vec[sum(dims_x[:i]):sum(dims_x[:i+1])] = np.atleast_2d(x_i).T

        return x_vec + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
