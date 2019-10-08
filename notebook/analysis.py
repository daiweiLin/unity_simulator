import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, sys
import glob


def is_inside_area(ir, point):
    signs = []
    for j in range(4):
        corner = ir[j * 2:j * 2 + 2]
        vector = corner - point
        edge = ir[8 + j * 2:8 + j * 2 + 2]
        signs.append(np.sign(np.cross(vector, edge)))

    #     plt.plot(ir[:8][::2], ir[:8][1::2])
    if np.all(signs[0] == signs):
        # print("Inside")
        return True
    else:
        # print("Not inside")
        return False


def calculate(visitor_data, section_num, ir_detection):
    columns = []
    for i in range(5):
        for j in range(24):
            colname = "ir{}_v{}".format(j, i)
            columns.append(colname)
    ir_detection_df = pd.DataFrame(columns=columns)

    section_length = int(625000 / 10000)

    for index, row in visitor_data[
                      section_num * section_length:section_num * section_length + section_length].iterrows():
        is_inside = np.array([])
        for i in range(5):
            position = [row["Location_x_{}".format(i + 1)], row["Location_z_{}".format(i + 1)]]
            visitor_is_inside = np.zeros((24,))
            for j in range(24):
                visitor_is_inside[j] = is_inside_area(ir_detection.iloc[j], position)
            is_inside = np.append(is_inside, visitor_is_inside)
        ir_detection_df.loc[index] = is_inside

    return ir_detection_df


def main(section_num, mode):

    file_date = "2019-08-30-003453"
    visitor_data = pd.read_csv("../SHARCNET/Results/multi/big_collision_model/original/SARA_LED_Multi/" + file_date + "/visitor_log.csv",sep=',')
    ir_detection = pd.read_csv("IR_detection_area.csv")

    save_dir = os.path.join(os.path.abspath('.'), 'Result', mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Calculating section {}".format(section_num))
    ir_detection_df = calculate(visitor_data, section_num, ir_detection)

    ir_detection_df.to_csv(os.path.join(save_dir,"visitor_IR_detection_{}.csv".format(section_num)), index=False)

if __name__ == "__main__":
    section_num = 0
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        section_num = int(sys.argv[2])
    main(section_num, mode)
