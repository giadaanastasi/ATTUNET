


#%% Call the parser to receive the input path
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the dataset folder",
                type=str)
    args = parser.parse_args()
except:
    e = sys.exc_info()[0]
