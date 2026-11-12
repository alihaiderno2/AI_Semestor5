import random
random.seed(42)
P_train = {"on_time": 0.70, "delayed": 0.30}
P_attend_given_train = {"on_time": 0.90, "delayed": 0.60}
# Task 1 & 2: Exact probabilities
p_attend_exact = (P_train["on_time"] * P_attend_given_train["on_time"] +
P_train["delayed"] * P_attend_given_train["delayed"])
P_miss_given_train = {"on_time": 0.10, "delayed": 0.40}
p_miss_exact = (P_train["on_time"] * P_miss_given_train["on_time"] +
P_train["delayed"] * P_miss_given_train["delayed"])
print(f"Exact P(Attend): {p_attend_exact:.4f}")
print(f"Exact P(Miss): {p_miss_exact:.4f}")
print(f"Total Sum: {p_attend_exact + p_miss_exact:.4f}")
# Task 3: Sampling simulation
def simulate(n_samples):
    attend_count = 0
    for _ in range(n_samples):
        train = "on_time" if random.random() < P_train["on_time"] else "delayed"
        if random.random() < P_attend_given_train[train]:
            attend_count += 1
    return attend_count / n_samples
    for N in [10000, 100000]:
        est = simulate(N)
        print(f"N={N:<6d}-> Estimated P(Attend) = {est:.4f}")