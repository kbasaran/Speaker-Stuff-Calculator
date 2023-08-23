# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:18:12 2022

@author: kerem.basaran
"""

import os
from signal_tools import parse_klippel_data
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
import pyperclip
import matplotlib.pyplot as plt
import winsound
import pickle


def make_import_dict():
    try:
        if "import_dict" not in globals() or input("Delete 'import_dict'??!! ").lower() == "y":
            import_dict = dict()
        else:
            raise KeyboardInterrupt

        while True:
            try:
                print("Waiting for new import in clipboard..")
                klippel_import = parse_klippel_data(pyperclip.waitForNewPaste())
                import_name = klippel_import.ObjectTitle
                import_dict[import_name] = klippel_import
                winsound.Beep(880, 100)
                print(f"Added '{import_name}'.\n")
            except ValueError:
                continue

    except KeyboardInterrupt:
        return import_dict


def simplify_curve(klippel_import, freq_array, sigma=3):
    x = klippel_import.x
    y = klippel_import.y
    y_new = np.interp(freq_array, x, y)
    y_filt = gaussian_filter(y_new, sigma)
    return dict(zip(freq_array, y_filt))


def test_smoothing(N=100, sigma=3):
    x = np.arange(N)
    y = np.random.randint(10, size=N) + 70
    plt.plot(x, y)

    y_filt = gaussian_filter(y, sigma)

    plt.plot(x, y_filt, "r")
    plt.ylim(50, 90)


# Define frequencies
ppo = 96
freq_start = 1000
freq_end = 15000
freq_array = np.around(
    2**np.arange(np.log2(freq_start), np.log2(freq_end * 2**(1/ppo)), 1/ppo),
    1
)


if __name__ == "__main__":

    # Define how to group and list
    # dict key: friendly name, value: list of keywords to look for in import name
    group_keywords = {
        "Mass production": ["MP"],
        "SSE cone": ["SSE"],
    }

    # dictionary key=imported name, value=desired name
    single_entries = {
        "1715-018-011 Lm 85.1 - D.01.01": 'G01 D sample - 1715-018-011',
        "7148 PSS 41 57485 DZ D.01.01": 'G08 D sample - 41.57485',
        "1701-025-004 DZ": 'G01 2018 Q-lab reference',
    }

    if True:
        # Read existing dataframe
        df = pd.read_excel(os.path.join(os.getcwd(), "export_df.xlsx"), index_col=0)
    else:
        # Or make a new one through reading each curve from clipboard
        df = pd.DataFrame(columns=freq_array)
        import_dict = make_import_dict()
        # and save this new one to a file
        try:
            with open("import_dict.pickle", "wb") as f:
                pickle.dump(import_dict, f)
        except Exception as e:
            print("Failed save of imports to pickle: " + str(e))

        for klippel_import_name, klippel_import in import_dict.items():
            df.loc[klippel_import_name, freq_array] = simplify_curve(klippel_import, freq_array)

        # Calculate the means and medians and add to df
        for group_name, keywords in group_keywords.items():
            df_group = df[np.prod([df.index.str.contains(keyword)
                                  for keyword in keywords], axis=0, dtype='bool')]
            # checks each keywrod
            if df_group.shape[0] > 0:
                n_curve = df_group.shape[0]
                for freq in freq_array:
                    # df.loc[group_name + " - mean", freq] = 10 * np.log10(np.average(
                    #     10**(df_group.loc[:, freq] / 10)
                    # ))
                    df.loc[group_name + f" - median of {n_curve}", freq] = 10 * np.log10(np.median(
                        10**(df_group.loc[:, freq] / 10)
                    ))

        # Rename some curves. It canb be any of the imported curves.
        df.rename(single_entries, inplace=True)

        # write to file
        base_file_name = "export_df"
        xlsx_to_write = os.path.join(os.getcwd(), f"{base_file_name}.xlsx")
        df.to_excel(xlsx_to_write)
        print(f"Written to Excel file to:\n'{xlsx_to_write}'.")

    def calc_mean_val(f_start, f_end, freq_array, SPL_array):
        return SPL_array[(freq_array >= f_start) & (freq_array < f_end)].mean()

    # plot settings
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["figure.figsize"] = [9, 5]

    # plot
    fig, ax = plt.subplots()
    for row_name in reversed(df.index):
        if (any([key in str(row_name) for key in single_entries.values()]) or
            any([key in str(row_name) for key in list(group_keywords.keys())])
            ) and (
                True
                # ("Wet" in row_name or "WZ" in row_name)
                # ("Dry" in row_name or "DZ" in row_name)
        ):

            # Voltage correction
            if "1715-018-011" in row_name or "41.57485" in row_name:
                ax.semilogx(freq_array, df.loc[row_name, :] + 3.05, label=row_name)
            else:
                ax.semilogx(freq_array, df.loc[row_name, :], label=row_name)

    # Print mean values and plot
    print("\nMean values:")
    sens_list = {}
    for i, row_name in enumerate(reversed(df.index)):
        mean_val = calc_mean_val(300, 1200, freq_array, df.loc[row_name, :])
        if all([word not in row_name for word in ["mean", "median"]]):
            sens_list[row_name] = mean_val
            # plt.scatter(i, sens_list[row_name], label=row_name)
        print(f"{i} - {row_name}: {mean_val:.2f}")

    ax.set_title("MM100DZ G08 - Mass production vs. test group")
    ax.set_ylabel("dB")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylim(75, 105)
    ax.set_xlim(freq_start, freq_end)
    ax.legend(loc="upper left")  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    # ax.grid(which="minor", axis="x")
