import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from pathlib import Path
import sys

# awful hack so I can keep this script in bin directory in repo and import from directory above
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from src.info_spread import Simulation
from src.population_density import Plot


def extract_values_from_filename(file: Path, extension: str):
    vals = file.name.split("-")
    N = int(vals[0][:-1])
    M = int(vals[1][:-1])
    L = int(vals[2][:-1 - len(extension)])
    return N, M, L


def extract_death_rate(simulation: Simulation) -> float:
    death_rate = simulation.ts_dead[-1] / simulation.M
    return death_rate

def extract_iterations(simulation: Simulation) -> int:
    raise NotImplementedError


# load results
def load_data(path: Path, N: int, L: int, datatype: str):

    X, y = [], []

    glob_string = "{}N-*M-{}L.csv".format(N, L)
    matching_files = Path.glob(path, glob_string)

    for file_name in sorted(matching_files, key=lambda name: extract_values_from_filename(name, ".csv")[1]):

        # TODO: get average death rate / iteration count
        N, M, L = extract_values_from_filename(file_name, ".csv")
        results = Simulation.load_results_from_csv(path, N, M, L)
        buffer = []
        for simulation in results:
            if datatype == "dr":
                extractor = extract_death_rate
            elif datatype == "iter":
                extractor = extract_iterations
            value = extractor(simulation)
            buffer.append(value)
        avg = np.average(buffer) if len(buffer) else 0

        X.append([N, M, L])
        y.append(avg)

    return np.array(X), np.array(y)

# train regressions to predict death rate / iterations for specific values of N, M and L
def main():
    path = Path("C:\\Users\\janek\\Desktop\\PG\\magisterka\\csv")

    for N in [128, 256]:
        for L in [1,2,4]:
            print("{}N, {}L".format(N, L))

            X, y = load_data(path, N, L, "dr")
            X = X[:, 1].reshape(-1, 1) # extract M column
            rounded_y = np.array([int(round(100 * yi)) for yi in y])
            X_train, X_test, y_train, y_test = train_test_split(X, rounded_y, test_size=0.2, random_state=42)

            clf = LogisticRegression(random_state=4, max_iter=10000).fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # rank accuracy
            accuracy = accuracy_score(y_test, y_pred)
            MSE = mean_squared_error(y_test, y_pred)
            print("accuracy:", accuracy)
            print("MSE:", MSE)

            plt.figure()
            plt.title("{}N, {}L".format(N, L))
            #plt.plot(X, y, "o")
            #plt.plot(X, clf.predict(X), "o")
            plt.plot(X, y)
            plt.plot(X, clf.predict(X))
            plt.legend(["Original", "Regression"])

    plt.show()


if __name__ == "__main__":
    main()