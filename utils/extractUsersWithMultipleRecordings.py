import pandas as pd
import argparse


def read_metadata(filename):
    df = pd.read_csv(filename, usecols=['RecordingId', 'ParticipantID', 'Gender', 'Age', 'Country', 'Continent'])
    return df


def extractUsers(userList):
    metadata = read_metadata(userList)
    ids = metadata["ParticipantID"]
    data = metadata[ids.duplicated(keep=False)]
    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--userList', type=str, help="Full metadata file (result of concatenation of all the csv files")
    args = parser.parse_args()
    user_list = args.userList
    data = extractUsers(user_list)

    data.to_csv("usersWithMultipleEntries.csv", index=False)
