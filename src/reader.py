import pandas as pd
from matplotlib import pyplot as plt

def read_data(file_path):
    file = pd.read_csv(file_path, header=None)
    # plot 2 graphs
    rad: plt.Axes
    fig, ((rad, txt), (per, area)) = plt.subplots(2, 2)
    rad.scatter(file[1].to_list(), file[2].to_list())
    rad.set_title('diagnostic vs radius')
    txt.scatter(file[1].to_list(), file[3].to_list())
    txt.set_title('diagnostic vs texture')
    per.scatter(file[1].to_list(), file[4].to_list())
    per.set_title('diagnostic vs perimeter')
    area.scatter(file[1].to_list(), file[5].to_list())
    area.set_title('diagnostic vs area')
    plt.show()

if __name__ == '__main__':
    wrong_path = True
    while wrong_path:
        try:
            file_name = input("Enter the file name from data directory: ")
            read_data(f"../data/{file_name}")
            wrong_path = False
        except Exception as e:
            print(e)
            wrong_path = True