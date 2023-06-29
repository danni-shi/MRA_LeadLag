import multiprocessing
from tqdm import tqdm
import time


# def foo(my_number):
#     square = my_number * my_number
#     time.sleep(1)
#     progress_bar.update(1)
#     return square
#
# if __name__ == '__main__':
#     start_indices = [i for i in range(6)]
#     progress_bar = tqdm(total=len(start_indices))
#     # map inputs to functions
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#         # use the pool to apply the worker function to each input in parallel
#         tqdm(pool.imap(foo, start_indices), total=6)
#         pool.close()
#         pool.join()
#     progress_bar.close()

def _foo(my_number):
    square = my_number * my_number
    time.sleep(1)
    return square

if __name__ == '__main__':
    with multiprocessing.Pool(processes=2) as pool:
        r = list(tqdm(pool.imap(_foo, range(10)), total=10))
        pool.close()
        pool.join()
    print(r)