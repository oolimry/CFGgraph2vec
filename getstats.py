import statistics

results = []
with open("result.txt", "r") as f:
    S = f.read()
    results = S.split(",")
    results.pop()
    results = [float(x) for x in results]

print(f'Mean: {statistics.mean(results)}')
print(f'Standard Deviation: {statistics.variance(results)**0.5}')
