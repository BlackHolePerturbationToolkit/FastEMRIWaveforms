import unittest
from few.utils.spline import CubicSpline, BicubicSpline, TricubicSpline
from few.utils.utility import get_separatrix
import numpy as np
import time
def test_func(x, y, z):
    return np.sin(x)*np.cos(y)/(1-z**2)

class TestSplines(unittest.TestCase):
    def test_tricubic(self):
        x = np.linspace(0.5, 1, 100)
        y = np.linspace(-1, 1, 100)
        z = np.linspace(2, 3, 100)
        XYZ = np.meshgrid(x, y, z, indexing='ij')
        fvec = test_func(*XYZ)
        trispl = TricubicSpline(x, y, z, fvec)
        self.assertTrue(np.allclose(trispl(0.51, 0.4, 2.4),test_func(0.51, 0.4, 2.4),1e-10))
        # # create a TricubicSpline to interpolate get_separatrix
        # a = np.linspace(0.0, 0.998, 100)
        # e = np.linspace(0.0, 0.99, 100)
        # x = np.linspace(-1.0, 1.0, 100)
        # XYZ = np.meshgrid(a, e, x, indexing='ij')
        # # time computation
        # tic = time.perf_counter()
        # sep = ( get_separatrix(XYZ[0].flatten(), XYZ[1].flatten(), XYZ[2].flatten())).reshape((100,100,100))
        # toc = time.perf_counter()
        # print("Time to compute separatrix: ", (toc-tic)/100**3)
        # trispl = TricubicSpline(a, e, x, sep)
        # # test precision by randomly drawing 100 points
        # rel_prec = []
        # for i in range(1000):
        #     a = np.random.uniform(0.0, 0.998)
        #     e = np.random.uniform(0.0, 0.99)
        #     x = np.random.uniform(-1.0, 1.0)
        #     rel_prec.append(1-trispl(a, e, x)/get_separatrix(a, e, x)) 
        # print(np.mean(rel_prec), np.std(rel_prec), np.max(rel_prec))
        # # self.assertTrue(np.allclose(trispl(a, e, x),get_separatrix(a, e, x),1e-10))
        
        
    def test_bicubic(self):
        x = np.linspace(0.5, 1, 100)
        y = np.linspace(-1, 1, 100)
        fvec = np.array([[test_func(x[i], y[j], 0.5) for j in range(100)] for i in range(100)])
        bispl = BicubicSpline(x, y, fvec)
        self.assertTrue(np.allclose(bispl(0.51, 0.4),test_func(0.51, 0.4, 0.5),1e-10))
    def test_cubic(self):
        x = np.linspace(0.5, 1, 100)
        fvec = np.array([test_func(x[i], 0, 0) for i in range(100)])
        spl = CubicSpline(x, fvec)
        self.assertTrue(np.allclose(spl(0.51),test_func(0.51, 0, 0),1e-10))
        
        
# import time

# def time_tricubic(n_points,Neval=5000):
#     # Generate some test data
#     x = np.linspace(0.5, 1, n_points)
#     y = np.linspace(-1, 1, n_points)
#     z = np.linspace(2, 3, n_points)
#     XYZ = np.meshgrid(x, y, z, indexing='ij')
#     fvec = test_func(*XYZ)

#     # Measure the time to create the TricubicSpline
#     start_time = time.time()
#     trispl = TricubicSpline(x, y, z, fvec)
#     creation_time = time.time() - start_time

#     # Measure the time to evaluate the TricubicSpline at a single point
#     # randomly chosen from the grid
#     xtest = np.random.uniform(0.5, 1, size=Neval)
#     ytest = np.random.uniform(-1, 1, size=Neval)
#     ztest = np.random.uniform(2, 3, size=Neval)
    
#     start_time = time.perf_counter()
#     [trispl(xtest[ii],ytest[ii],ztest[ii]) for ii in range(Neval)]
#     evaluation_time = (time.perf_counter() - start_time)/Neval

#     return creation_time, evaluation_time

# import matplotlib.pyplot as plt

# n_points_values = [10, 20, 40, 80, 160, 200]
# evaluation_times = []

# for n_points in n_points_values:
#     _, evaluation_time = time_tricubic(n_points)
#     evaluation_times.append(evaluation_time)

# plt.figure()
# plt.loglog(n_points_values, evaluation_times,'-o')
# plt.xlabel('Number of Points')
# plt.ylabel('Evaluation Time (seconds)')
# plt.title('Evaluation Time vs Number of Points')
# plt.tight_layout()
# plt.show()
