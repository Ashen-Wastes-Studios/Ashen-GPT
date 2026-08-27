import time

start_time = time.time()

for i in range(10000):
    print(i*2)

end_time = time.time()

total_time = end_time - start_time

print(f"time taken: {total_time}")