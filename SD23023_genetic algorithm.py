import numpy as np
import streamlit as st
import pandas as pd

# GA Parameters
POP_SIZE = 300
CHROM_LEN = 80
TARGET_ONES = 50
MAX_FITNESS = 80
GENERATIONS = 50
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN   # ~1 bit flip per chromosome
ELITISM = 2


def fitness(ind):
    ones = int(ind.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)   # peak at 50 ones



def init_population():
    return np.random.randint(0, 2, size=(POP_SIZE, CHROM_LEN), dtype=np.int8)

def tournament_selection(fits):
    idx = np.random.randint(0, POP_SIZE, size=TOURNAMENT_K)
    best = idx[np.argmax(fits[idx])]
    return best

def one_point_crossover(a, b):
    if np.random.rand() > CROSSOVER_RATE:
        return a.copy(), b.copy()
    point = np.random.randint(1, CHROM_LEN)
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

def mutate(ind):
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    out = ind.copy()
    out[mask] = 1 - out[mask]
    return out



def run_ga():
    pop = init_population()
    history = {"Best": [], "Avg": [], "Worst": []}

    for gen in range(GENERATIONS):
        fits = np.array([fitness(ind) for ind in pop])

        # Log history
        history["Best"].append(np.max(fits))
        history["Avg"].append(np.mean(fits))
        history["Worst"].append(np.min(fits))

        # Elitism
        elite_idx = np.argpartition(fits, -ELITISM)[-ELITISM:]
        elites = pop[elite_idx].copy()

        new_pop = []
        while len(new_pop) < POP_SIZE - ELITISM:
            p1 = pop[tournament_selection(fits)]
            p2 = pop[tournament_selection(fits)]
            c1, c2 = one_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])

        pop = np.vstack([new_pop[:POP_SIZE - ELITISM], elites])

    # Final best
    final_fits = np.array([fitness(ind) for ind in pop])
    best_idx = np.argmax(final_fits)
    best_ind = pop[best_idx]
    best_fit = final_fits[best_idx]

    return best_ind, best_fit, pd.DataFrame(history)



st.set_page_config(page_title="GA — Bit Pattern (80 bits)", page_icon="🧬", layout="wide")

st.title("🧬 Genetic Algorithm — Bit Pattern (Fixed Assignment Version)")
st.caption("Population = 300 | Chromosome = 80 bits | Target = 50 ones | Max Fitness = 80")

seed = st.number_input("Random seed", min_value=0, value=42)
run = st.button("Run GA", type="primary")

if run:
    np.random.seed(seed)

    best_ind, best_fit, hist = run_ga()

    ones = int(best_ind.sum())
    zeros = CHROM_LEN - ones
    bitstring = "".join(map(str, best_ind.tolist()))

    st.subheader("🏁 Best Individual Found")
    st.write(f"Fitness = **{best_fit}**")
    st.write(f"Ones = **{ones}**, Zeros = **{zeros}**")

    st.code(bitstring)

    st.subheader("📉 Convergence")
    st.line_chart(hist)

    if ones == TARGET_ONES:
        st.success("Perfect result: 50 ones achieved! 🎉")
    else:
        st.info("Near-optimal solution. Try different seeds.")