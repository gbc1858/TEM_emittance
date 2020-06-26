import settings
import pandas as pd
from scipy.optimize import curve_fit


def linear_function(x, a):
    return a * x


def quadratic(data, a, b, c):
    return a * data ** 2 + b * data + c


def get_sol_field(v_sol_list):
    """

    :param v_sol_list:
    :return: B field list in Gauss
    """
    sol_file = pd.read_excel(settings.TEM_SOL_VOLTAGE_LOC)
    # current = sol_file[settings.SOL_CURRENT_COLUMN].tolist()
    voltage = sol_file[settings.SOL_VOLTAGE_COLUMN].tolist()
    b_field = sol_file[settings.SOL_B_FIELD_COLUMN].tolist()
    popt, pcov = curve_fit(linear_function, voltage, b_field, maxfev=100000)
    return [i * popt[0] for i in v_sol_list]

if __name__ == '__main__':
    print(get_sol_field([11.1]))
    print(get_sol_field([11]))
