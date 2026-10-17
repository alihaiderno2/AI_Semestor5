from logic import *

rain = Symbol("It is raining")
wet_road = Symbol("The road is wet")
umbrella = Symbol("A student should carry an umbrella")
slow_traffic = Symbol("Traffic is slow")
late_arrival = Symbol("The student is likely to arrive late")

knowledge_base = And(
    Implication(rain, wet_road),
    Implication(rain, umbrella),
    Implication(wet_road, slow_traffic),
    Implication(slow_traffic, late_arrival),
    rain
)

queries = [
    wet_road,
    umbrella,
    slow_traffic,
    late_arrival,
    rain,
    Not(rain),
    Not(wet_road)
    ]

for query in queries:
    result = model_check(knowledge_base, query)
    print(query, "=>", result)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
kb_simple = And(P, Q)
print(f"Simplified KB: {kb_simple.formula()}")
print("-" * 50)

task4_queries = [
    ("1. P", P),
    ("2. Q", Q),
    ("3. P ∧ Q", And(P, Q)),
    ("4. ¬P", Not(P)),
    ("5. R", R)
]

for label, q in task4_queries:
    entailed = model_check(kb_simple, q)
    print(f"Query {label:<28} => KB |= Query is {entailed}")

print("\n" + "-" * 50)
print("TASK 5: EXTENDED KNOWLEDGE-BASED SYSTEM")
print("=" * 70)

exam_tomorrow = Symbol("exam_tomorrow")
leave_early = Symbol("leave_early")
exam_rule = Implication(And(late_arrival, exam_tomorrow), leave_early)

# Test 1: With exam_tomorrow included as a fact
kb_with_exam = And(
    Implication(rain, wet_road),
    Implication(rain, umbrella),
    Implication(wet_road, slow_traffic),
    Implication(slow_traffic, late_arrival),
    exam_rule,
    rain,
    exam_tomorrow
)
res_with = model_check(kb_with_exam, leave_early)
print("Test 1 With 'exam_tomorrow' included as a fact:")
print(f"Does KB entail 'leave_early'? => {res_with}")

print("-" * 50)

# Test 2: Without exam_tomorrow included as a fact
kb_without_exam = And(
    Implication(rain, wet_road),
    Implication(rain, umbrella),
    Implication(wet_road, slow_traffic),
    Implication(slow_traffic, late_arrival),
    exam_rule,
    rain
)
res_without = model_check(kb_without_exam, leave_early)
print("Test 2 With 'exam_tomorrow' removed from facts:")
print(f"Does KB entail 'leave_early'? => {res_without}")