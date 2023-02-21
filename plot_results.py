import pickle
import matplotlib.pyplot as plt


with open("./results/sgd_lr=-2.5_beta=0.pkl", "rb") as f:
    samples = pickle.load(f)
    m = pickle.load(f)
    l = pickle.load(f)
    u = pickle.load(f)

with open("./results/art_lr=-2.5_beta=-0.5.pkl", "rb") as f:
    samples_ = pickle.load(f)
    m_ = pickle.load(f)
    l_ = pickle.load(f)
    u_ = pickle.load(f)

with open("./results/pop-art_lr=-2.5_beta=-0.5.pkl", "rb") as f:
    samples__ = pickle.load(f)
    m__ = pickle.load(f)
    l__ = pickle.load(f)
    u__ = pickle.load(f)

with open("./results/normalized-sgd_lr=-2.5_beta=-0.5.pkl", "rb") as f:
    samples___ = pickle.load(f)
    m___ = pickle.load(f)
    l___ = pickle.load(f)
    u___ = pickle.load(f)

# Comparison plot
fig, spls = plt.subplots(2, 1, figsize=(12, 16))
spl = spls[0]

spl.plot(samples__, m__, color="C2", label="PopArt")
spl.fill_between(samples__, u__, l__, facecolor="C2", alpha=0.5)

spl.plot(samples_, m_, color="C0", label="ART")
spl.fill_between(samples_, u_, l_, facecolor="C0", alpha=0.5)

spl.plot(samples, m, color="C3", label="SGD")
spl.fill_between(samples, u, l, facecolor="C3", alpha=0.5)

spl.set_yscale("log")
spl.set_xlabel("# samples")
spl.set_ylabel("RMSE (log scale)")
spl.set_xlim((0, 5000))
spl.legend(loc="lower left", frameon=False)

# Solo NormalizedSGD plot
spl = spls[1]
spl.plot(samples___, m___, color="C4", label="NormalizedSGD")
spl.fill_between(samples___, u___, l___, facecolor="C4", alpha=0.5)

spl.set_yscale("log")
spl.set_xlabel("# samples")
spl.set_ylabel("RMSE (log scale)")
spl.set_xlim((0, 5000))
spl.legend(loc="lower left", frameon=False)

plt.savefig("./results/results.png", bbox_inches="tight")
