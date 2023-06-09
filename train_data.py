'''
This Module is for Training
'''
# import packages
import pickle
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC



# File paths
def train():
    DEFAULT_VECTOR = (r"C:\Users\DELL\AppData\Local\Programs\Python\Python311"
                    r"\SemIIProject\Sample2\Final_Project\output"
                    r"\ExtractedData.PICKLE")
    DEFAULT_RECOGNITION = (r"C:\Users\DELL\AppData\Local\Programs\Python\Python311"
                        r"\SemIIProject\Sample2\Final_Project\output"
                        r"\Recognitions.pickle")
    DEFAULT_LABELS = (r"C:\Users\DELL\AppData\Local\Programs\Python\Python311"
                    r"\SemIIProject\Sample2\Final_Project\output"
                    r"\LabelData.pickle")

    # Adding arguments
    arg_parser = argparse.ArgumentParser()  # Creating an ArgumentParser instance

    arg_parser.add_argument("-vf", "--vector_file", default=DEFAULT_VECTOR)
    arg_parser.add_argument("-rd", "--recognition_data",
                            default=DEFAULT_RECOGNITION)
    arg_parser.add_argument("-ld", "--label_data", default=DEFAULT_LABELS)

    arg_dict = vars(arg_parser.parse_args())

    # Face data loading from extract file
    #facial_data = pickle.loads(open(arg_dict["vector_file"], "rb").read())
    with open(arg_dict["vector_file"], "rb") as file:
        facial_data = pickle.load(file)
    print("<< Face extractions loading complete >>")

    # Encoding labels as integers
    encodings = LabelEncoder()
    labels = encodings.fit_transform(facial_data["names"])
    print("<< Encoding labels complete >>")

    # train the model
    print("<< Training model >>")
    data_recognize = SVC(C=1.0, kernel="linear", probability=True)
    data_recognize.fit(facial_data["vector_file"], labels)

    # Saving the Recognition model data

    with open(arg_dict["recognition_data"], "wb") as file_rec:
        # Write the pickled data to the file
        pickle.dump(data_recognize, file_rec)

    # write the label encoder to disk

    with open(arg_dict["label_data"], "wb") as file_label:
        # Write the pickled data to the file
        pickle.dump(encodings, file_label)
    print("<< Saving Data >>")
    print("<< Training Complete >>")
