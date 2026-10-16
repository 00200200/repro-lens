# Synthetic fixture

`raw/synthetic.csv` is a generated, 240-row binary-classification fixture with two
numeric features. It contains no personal or externally licensed data.

Generation: Python `random.Random(1729)`; x1 and x2 uniformly sampled in [-2, 2];
target is 1 when `x1 + 0.7*x2 + gaussian_noise(0, 0.35) > 0`. Features are stored
with eight decimal places. The committed CSV bytes define the dataset version;
the verification report records their SHA-256. Regenerating with another runtime
is unnecessary to reproduce the training experiment.

Real projects should replace this fixture with an explicitly versioned dataset,
document access and preprocessing, and update the verification input patterns.
