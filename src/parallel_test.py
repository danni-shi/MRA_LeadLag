import multiprocessing
import scipy.io as spio
import pickle
def read_data(x,y=10):
    print(x+y)

def wrapper(sigma, k):
  read_data(**kwargs)
def main():

  inputs = [(0.1, 2), (0.1, 3), (0.1, 4), (0.2, 2), (0.2, 3), (0.2, 4)]
  inputs = [(inputs[i][0],inputs[i][1],i) for i in range(len(inputs))]
  inputs = range(5)
  with multiprocessing.Pool() as pool:
    # use the pool to apply the worker function to each input in parallel
    results = pool.starmap(wrapper(), inputs)

if __name__ == "__main__":
  main()