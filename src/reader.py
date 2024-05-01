from data_spliter import split_data

if __name__ == '__main__':
    wrong_path = True
    while wrong_path:
        try:
            file_name = input("Enter the file name from data directory: ")
            split_data(f"../data/{file_name}")
            wrong_path = False
        except Exception as e:
            print(e)
            wrong_path = True