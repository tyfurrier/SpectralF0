import logging

from interface import spread_file_naming, OUTPUT_DIR, Decomposition, get_spread_sound
import pandas as pd
import plotly.express as px
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns

logging.basicConfig(level=logging.INFO)

def visualize_stimuli(cents: float,
                      decomposition: Decomposition):
    """
    Visualize the stimuli for a given spread and decomposition.
    :param cents: The radius in each formant in cents DO NOT DO MORE THAN TWO DECIMALS.
    :param decomposition: The decomposition.
    """
    sound, sr = get_spread_sound(folder_name="radius_50",
                                 cents=cents,
                                 part=decomposition)
    sound = sound[len(sound)//4:]
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(0)

    dt = 1 / sr  # sampling interval (used to be 0.1)
    Fs = sr  # sampling frequency
    length_in_seconds = len(sound) / sr
    t = np.arange(0, length_in_seconds, dt)

    s = sound  # the signal

    title_suffix = f"for {cents} cents spread"

    fig = plt.figure(figsize=(7, 7), layout='constrained')
    mosaic_list = []
    signal = False
    magnitude = False
    log_mag = True
    phase = False
    angle = False
    if signal:
        mosaic_list.append(["signal", "signal"])
    if magnitude or log_mag:
        mosaic_list.append([])
        if magnitude:
            mosaic_list[-1].append("magnitude")
        if log_mag:
            mosaic_list[-1].append("log_magnitude")
    if phase or angle:
        mosaic_list.append([])
        if phase:
            mosaic_list[-1].append("phase")
        if angle:
            mosaic_list[-1].append("angle")
    axs = fig.subplot_mosaic(mosaic_list)

    if signal:
        # plot time signal:
        axs["signal"].set_title(f"Signal {title_suffix}")
        axs["signal"].plot(t, s, color='C0')
        axs["signal"].set_xlabel("Time (s)")
        axs["signal"].set_ylabel("Amplitude")

    if magnitude:
        # plot different spectrum types:
        axs["magnitude"].set_title(f"Magnitude Spectrum {title_suffix}")
        axs["magnitude"].magnitude_spectrum(s, Fs=Fs, color='C1')

    if log_mag:
        axs["log_magnitude"].set_title(f"Log. Magnitude Spectrum {title_suffix}")
        axs["log_magnitude"].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

    if phase:
        axs["phase"].set_title(f"Phase Spectrum {title_suffix}")
        axs["phase"].phase_spectrum(s, Fs=Fs, color='C2')

    if angle:
        axs["angle"].set_title("Angle Spectrum")
        axs["angle"].angle_spectrum(s, Fs=Fs, color='C2')

    plt.show()

def results_df(drop_jack: bool = True):
    df = pd.read_csv("spreadsheet.csv")
    df.reset_index(inplace=True)
    df["ratio"] = np.maximum(df["sound1"] / df["sound2"], df["sound2"] / df["sound1"])
    df["different"] = df["decision"]
    df["different"].replace(["Same", "Different"], [0, 1], inplace=True)
    df["linear_difference"] = np.abs(df["sound1"] - df["sound2"])
    df["average"] = np.average([df["sound1"], df["sound2"]])
    df["size"] = 1
    df["ln_ratio"] = np.log(df["ratio"])
    df["log10_ratio"] = np.log10(df["ratio"])
    df["log2_ratio"] = np.log2(df["ratio"])
    if drop_jack:
        jack_index = df[(df['subject'] == "Jack")].index
        df.drop(jack_index, inplace=True)
    # df["subject_category"] = df["subject"].astype("category")

    return df

def scatter_plot(df: pd.DataFrame, x: str = "ratio", y: str = "different"):
    fig = px.scatter(df, x=x, y=y, color="subject",
                     hover_data=["sound1", "sound2", "decision"])
    fig.show()

def logistic_regression(df: pd.DataFrame, variable: str, plot: bool = True):
    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit
    model = LogisticRegression(C=1e5, solver='lbfgs')
    X = df[variable].values.reshape(-1, 1)
    Y = df["different"].values.reshape(-1, 1)
    model.fit(X, Y)
    if plot:
        x_test = np.linspace(0.0, np.max(X), 100)
        # predict dummy y_test data based on the logistic model
        y_test = x_test * model.coef_ + model.intercept_
        halfway_point = ((0.5 - model.intercept_[0]) / model.coef_[0])[0]

        sigmoid = expit(y_test)
        plt.scatter(df[variable], df["different"], label=variable)

        plt.scatter(X, Y, color="black", label="data")
        # plt.scatter(X, model.predict(X), color="blue", label="regression prediction")
        # ravel to convert the 2-d array to a flat array
        plt.plot(x_test, sigmoid.ravel(), c="green", label="logistic fit")
        plt.yticks([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.xticks([10, np.max(X), halfway_point])
        plt.axhline(.5, color="red", label="cutoff")
        plt.legend(loc="lower right")
        plt.show()
    possible_independents = ["log10_ratio",
                             "ratio",
                             "log2_ratio",
                             "ln_ratio",
                             # "linear_difference", "average", "size",
                             # "subject_category"
                             ]
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = results_df()
    plot = False
    if plot:
        scatter_plot(df, x="linear_difference")
        scatter_plot(df)
        scatter_plot(df, x="log_ratio")
    stats = False
    if stats:
        for v in ["log2_ratio", "ratio", "linear_difference"]:
            model = logistic_regression(df, variable=v)
            print(model.score(df[v].values.reshape(-1, 1), df["different"].values.reshape(-1, 1)))

    # sns.regplot(x=df["log2_ratio"], y=df["different"], data=df, logistic=True, ci=None)
    for cents in [0.1, 1, 10, 49]:
        visualize_stimuli(cents=cents, decomposition=Decomposition.FULL)