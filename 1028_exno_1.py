import csv

# Training data (you can also load from CSV)
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

concepts = [row[:-1] for row in data]
target = [row[-1] for row in data]

# Initialize S and G
S = concepts[0].copy()
G = [['?' for _ in range(len(S))]]

# Candidate Elimination Algorithm
for i, example in enumerate(concepts):
    if target[i] == 'Yes':  # Positive example
        for j in range(len(S)):
            if example[j] != S[j]:
                S[j] = '?'
        G = [g for g in G if all(g[j] == '?' or g[j] == example[j] for j in range(len(S)))]
    
    else:  # Negative example
        new_G = []
        for g in G:
            for j in range(len(S)):
                if g[j] == '?' and S[j] != example[j]:
                    new_hypothesis = g.copy()
                    new_hypothesis[j] = S[j]
                    new_G.append(new_hypothesis)
        G = new_G

    print(f"\nAfter Example {i+1}:")
    print("S =", S)
    print("G =", G)

print("\nFinal Version Space:")
print("S =", S)
print("G =", G)
