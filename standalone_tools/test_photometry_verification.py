'''
test_photometry_varification.py
'''

from matplotlib import pyplot as plt
import numpy as np


def generate_pseudo_error(num_repetitions: int = 10, ideal_light_dir: np.ndarray = None):
    num_lights, num_dimensions = ideal_light_dir.shape
    repetitions = []
    for i in range(num_repetitions):
        repetitions.append(ideal_light_dir + np.random.randn(num_lights, num_dimensions) * 3)
    return np.array(repetitions)

def plot_repetitions(repetitions: np.ndarray, dimension_names: list):
    num_rep, num_lights, num_dimensions = repetitions.shape
    fig, ax = plt.subplots(num_dimensions, num_lights)
    for i in range(num_lights):
        for j in range(num_dimensions):
            ax[j, i].scatter(range(num_rep), repetitions[:, i, j])
            ax[j, i].set_title(f"Light {i+1}, {dimension_names[j]}")
            ax[j, i].set_xlabel("Repetition")
            ax[j, i].set_ylabel(f"{dimension_names[j]} (deg)")
            ax[j, i].grid(True)
            ax[j, i].set_xlim(0, num_rep)
            ax[j, i].set_xticks(range(num_rep))
            ax[j, i].set_ylim(0, 360)
            ax[j, i].set_yticks(range(0, 360, 60))
            ax[j, i].set_yticklabels(range(0, 360, 60))
            

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ideal_light_dir = np.array([[1.68, 50.73],
                                [179.92, 51.63],
                                [270.39, 51.03],
                                [90.39, 51.27]]) # shape (4, 2(azimuth, elevation))
    dimension_names = ["azimuth", "elevation"]
    repetitions = generate_pseudo_error(num_repetitions=10, ideal_light_dir=ideal_light_dir)
    print(repetitions)

    plot_repetitions(repetitions, dimension_names)

