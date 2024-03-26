from cohortbayes import DataLoad, Data, BayesModel

dl = DataLoad(path="./eval_data.csv")
train, test = dl.prepare(split_date="2024-04-01")

model = BayesModel(training_data=train, add_eps=False)
model.build()
out = model.fit()
print(out)
