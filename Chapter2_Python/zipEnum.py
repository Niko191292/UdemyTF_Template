list_a = [10, 20, 30]
list_b = ["Jan", "Peter", "Max"]
list_c = [True, False]

for a, b, c in zip(list_a, list_b, list_c):
    print(a, b, c)

A = False
B = True

if not (A and B):
    print("True")

