import pandas as pd


class EDA:
    def __init__(self):
        self.train_file, self.test_file, self.job_tag, self.user_tag, self.job_company = self.load_csv()
        self.user_map, self.job_map, self.company_map = self.maps()
        self.job_company_process()
        self.train_csv = self.preprocess(self.train_file)
        self.test_csv = self.preprocess(self.test_file)

    def load_csv(self):
        train_file = pd.read_csv('./data/train.csv')
        test_file = pd.read_csv('./data/test_job.csv')
        job_tag = pd.read_csv('./data/job_tags.csv')
        user_tag = pd.read_csv('./data/user_tags.csv')
        job_company = pd.read_csv('./data/job_companies.csv')
        return train_file, test_file, job_tag, user_tag, job_company

    def job_company_process(self):
        self.job_company = self.job_company.drop(columns=['companySize'], axis=0)

    def calculate_tags_ratio(self, userID, jobID):
        user_tag_list = []
        job_tag_list = []
        val = []
        user_tag_list = self.user_tag[self.user_tag['userID'] == userID]['tagID'].tolist()
        job_tag_list = self.job_tag[self.job_tag['jobID'] == jobID]['tagID'].tolist()

        for i in job_tag_list:
            if i in user_tag_list:
                val.append(1)

        return len(val) / len(job_tag_list)

    def preprocess(self, df_):
        df = pd.merge(df_, self.job_company, how='left', on='jobID')

        df['user_tag_count'] = 0
        for i in range(len(df)):
            df['user_tag_count'].iloc[i] = len(self.user_tag[self.user_tag['userID'] == df['userID'].iloc[i]])
        df['user_tag_count'] = df['user_tag_count']/max(df['user_tag_count'])

        df['job_tag_count'] = 0
        for i in range(len(df)):
            df['job_tag_count'].iloc[i] = len(self.job_tag[self.job_tag['jobID'] == df['jobID'].iloc[i]])
        df['job_tag_count'] = df['job_tag_count']/max(df['job_tag_count'])

        df['user_job_count'] = 0
        for i in range(len(df)):
            df['user_job_count'].iloc[i] = self.calculate_tags_ratio(df['userID'].iloc[i], df['jobID'].iloc[i])

        df['userID'].replace(list(self.user_map.keys()), list(self.user_map.values()), inplace=True)
        df['jobID'].replace(list(self.job_map.keys()), list(self.job_map.values()), inplace=True)
        df['companyID'].replace(list(self.company_map.keys()), list(self.company_map.values()), inplace=True)
        df['userID'] = df['userID']/len(self.user_map)
        df['jobID'] = df['jobID']/len(self.job_map)
        df['companyID'] = df['jobID']/len(self.company_map)
        return df

    def maps(self):
        user_list = list(set(self.train_file['userID']))
        user_list.sort()

        job_list = list(set(self.job_tag['jobID']))
        job_list.sort()

        company_list = list(set(self.job_company['companyID']))
        company_list.sort()

        user_map = {}
        for idx, x in enumerate(user_list):
            user_map[x] = idx

        job_map = {}
        for idx, x in enumerate(job_list):
            job_map[x] = idx

        company_map = {}
        for idx, x in enumerate(company_list):
            company_map[x] = idx
        return user_map, job_map, company_map

# a = len(user_list)
# b = list(np.linspace(0, len(user_list), num=len(user_list), endpoint=False, dtype=int))

# train_file['userID'].replace(user_list, b, inplace=True)


 # b = pd.merge(a, user_tag, on='userID', how='outer')
# c = pd.merge(b, job_tag, on='jobID', how='outer')



# train_dummy = pd.get_dummies(train_file)

if __name__ == '__main__':
    eda = EDA().train_file