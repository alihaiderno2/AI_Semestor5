import pomegranate
from pomegranate import *
from collections import Counter

train = Node(DiscreteDistribution({
    "on time": 0.7,
    "delayed": 0.3
}), name="train")

appointment = Node(ConditionalProbabilityTable([
    ["on time", "attend", 0.90],
    ["on time", "miss", 0.10],
    ["delayed", "attend", 0.60],
    ["delayed", "miss", 0.40]
], [train.distribution]), name="appointment")

model = BayesianNetwork()
model.add_states(train, appointment)
model.add_edge(train, appointment)
model.bake()

exact_prob = model.probability([["on time", "attend"]])+ model.probability([["delayed", "attend"]])

exact_miss = model.probability([["on time", "miss"]]) + model.probability([["delayed", "miss"]])

print("=== Tasks 1 & 2: Exact Probabilities ===")

print(f"Exact probability of attending the appointment: {exact_prob:.4f}")
print(f"Exact probability of missing the appointment: {exact_miss:.4f}")
print("Sum = ", exact_prob + exact_miss)

def generate_samples():
    sample = {}
    parents = {}

    for state in model.states:
        if isinstance(state.distribution, ConditionalProbabilityTable):
            sample[state.name] = state.distribution.sample(parent_values=parents)
        else:
            sample[state.name] = state.distribution.sample()

        parents[state.distribution] = sample[state.name]
    
    return sample

N = 1000
data = []
for i in range(N):
    sample = generate_samples()
    data.append(sample["appointment"])
counts = Counter(data)
simulated_prob_attend = counts["attend"] / N

print("\n=== Task 3 & 4: Sampling Simulation ===")
print("Sample counts:", counts)
print(f"Simulated probability of attending: {simulated_prob_attend:.4f}")
print(f"Difference (Exact vs Simulated): {abs(exact_prob - simulated_prob_attend):.4f}")