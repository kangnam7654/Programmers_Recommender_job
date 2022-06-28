import pandas as pd
from tqdm import tqdm


def preprocess(train_csv, user_tags_csv, job_tags_csv, tags_csv, test_csv):
    # 중복제거
    train_csv.drop_duplicates(["userID", "jobID", "applied"], inplace=True)
    user_tags_csv.drop_duplicates(["userID", "tagID"], inplace=True)
    job_tags_csv.drop_duplicates(["jobID", "tagID"], inplace=True)
    tags_csv.drop_duplicates(["tagID", "keyword"], inplace=True)

    # 인덱스 리셋
    train_csv.reset_index(drop=True, inplace=True)
    user_tags_csv.reset_index(drop=True, inplace=True)
    job_tags_csv.reset_index(drop=True, inplace=True)
    tags_csv.reset_index(drop=True, inplace=True)

    # maps
    user_list = sorted(list(set(list(user_tags_csv.userID)))) # 중복제거
    job_list = sorted(list(set(list(job_tags_csv.jobID))))

    user_map = {}
    job_map = {}

    for idx, U in enumerate(user_list):
        user_map[U] = idx

    for idx, J in enumerate(job_list):
        job_map[J] = idx

    # train encoding
    train_csv["userID_encoding"] = train_csv["userID"].map(
        lambda x: user_map[x]
    )
    train_csv["jobID_encoding"] = train_csv["jobID"].map(lambda x: job_map[x])

    # test encoding
    test_csv["userID_encoding"] = test_csv["userID"].map(lambda x: user_map[x])
    test_csv["jobID_encoding"] = test_csv["jobID"].map(lambda x: job_map[x])

    # tag 벡터화
    tagID = sorted(list(tags_csv.tagID))
    tmp_df = pd.DataFrame(columns=tagID) # tagID를 컬럼으로 갖는 df 생성

    train_csv = pd.concat([train_csv, tmp_df], axis=1, join="outer")
    test_csv = pd.concat([test_csv, tmp_df], axis=1, join="outer")

    # tag 채우기
    train_csv.fillna(0, inplace=True)
    test_csv.fillna(0, inplace=True)

    # user_tag, job_tag를 참조하여 tag 해당되는 경우 +1
    for i in tqdm(range(len(train_csv))):
        user = train_csv.loc[i, "userID"]
        user_tags = list(user_tags_csv.loc[user_tags_csv["userID"] == user].tagID)
        job = train_csv.loc[i, "jobID"]
        job_tags = list(job_tags_csv.loc[job_tags_csv["jobID"] == job].tagID)
        train_csv.loc[i, user_tags] += 1
        train_csv.loc[i, job_tags] += 1

    for i in tqdm(range(len(test_csv))):
        user = test_csv.loc[i, "userID"]
        user_tags = list(user_tags_csv.loc[user_tags_csv["userID"] == user].tagID)
        job = test_csv.loc[i, "jobID"]
        job_tags = list(job_tags_csv.loc[job_tags_csv["jobID"] == job].tagID)
        test_csv.loc[i, user_tags] += 1
        test_csv.loc[i, job_tags] += 1

    train_desc = train_csv.describe()
    test_desc = test_csv.describe()

    train_zeros = train_desc.loc["max", :] == 0
    test_zeros = test_desc.loc["max", :] == 0
    zeros = pd.concat([train_zeros, test_zeros], axis=1, join="outer")
    zeros_ = (zeros == True).all(axis=1)
    zeros_index = list(zeros_[zeros_ == True].index)

    train_csv.drop(zeros_index, axis=1, inplace=True)
    test_csv.drop(zeros_index, axis=1, inplace=True)
    # 저장
    train_csv.to_csv("train_data.csv", index=False, encoding="utf-8")
    test_csv.to_csv("test_data.csv", index=False, encoding="utf-8")
