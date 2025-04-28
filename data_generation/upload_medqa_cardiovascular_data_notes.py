import pandas as pd
import json
from datasets import load_dataset, DatasetDict, Dataset

# from lighteval.tasks.default_tasks

med_qa = load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)

data = pd.read_csv("data/benchmarks/medqa_4opt_with_labels.csv")

dictionary = eval(data["options"].values[0])

items = list(dictionary.items())

correct_format = [{"key": x[0], "value": x[1]} for x in items]


def convert_options_to_medqa_format(option):
    dictionary = eval(option)

    items = list(dictionary.items())

    correct_format = [{"key": x[0], "value": x[1]} for x in items]

    return correct_format


data["options"] = data["options"].apply(lambda x: convert_options_to_medqa_format(x))

cardiovascular_data = data[data["category"] == "Cardiovascular"].reset_index(drop=True)

print(cardiovascular_data["question"].values[2])

print(cardiovascular_data["options"].values[21])

print(cardiovascular_data["answer_idx"].values[21])

print(cardiovascular_data["question"].values[32])


## QUESTION 0:
# Has actual values in there, very similar to what we give. However, we don't really have the *cause* of the patient's symptoms in our data, which might be a problem.

## QUESTION 1:
# Is definitely about cardiovascular patients and the model might understand the patient better - but again the outcome is how the patient should be managed, which is not in the data.

## QUESTION 2:
# Same thing - this time with medication. However, this might be something the model would be more likely to pick up. It could come about in the counterfactual examples

## QUESTION 3:
# This is clearly outside the information in the data

## QUESTION 4:
# Some of the information is in there, but the outcome is different (which therapy should be chosen)

## QUESTION 5:
## There's nothing in the data that would help with this

## QUESTION 6:
# Nothing in the data

## QUESTION 7:
# Nope

## QUESTION 8:
# No I don't think it would be good at this either

## QUESTION 9:
# This might work. Again, the outcome is different but this is damn close. I think it could work on this

## QUESTION 10:
# It's not really anything to do with cardiovascular stuff though is it?

## QUESTION 11:
# Absolutely not cardiovascular

## QUESTION 12:
# It's about what surgery to perform. So this would only work if the model has some sort of better representation of the patient or something like that.

## QUESTION 13:
# YES! This is what we want - this is the sort of thing that it should excel at.

## QUESTION 14:
# Not really.

## QUESTION 15:
# maaaaybe but probably not

## QUESTION 16:
# Not really!

## QUESTION 17:
# Would also be very surprising if it got better at this

## QUESTION 18:
# I don't think so - they have a lot of focus on vitals and diagnoses from vitals

## QUESTION 19:
# No

## QUESTION 21
# Not cardiovascular either

## QUESTION 22:
# There is a trillion of these vitals + sounds -> which diagnoses? Don't think we'll get better on them though

## QUESTION 23
# I don't think so - I think the only thing should be that due to exposure to more medical lingo in finetuning, it might make those words more likely --> increased likelihood of nearby words subsequently. But other than that, got nothing.

## QUESTION 24:
# Most likely not.

## QUESTION 25:
# maaaaybe - the outcome is different though so not certain.

## QUESTION 26:
# No I don't think so.

## QUESTION 27:
# Wrong outcome

## QUESTION 28:
# question concerns management

## QUESTION 29:
# not cardiovascular

## QUESTION 30:
# Unlikely

## QUESTION 31:
#

dataset = DatasetDict({"test": Dataset.from_pandas(data)})

dataset.push_to_hub("mikkel-werling/medqa_4opt_test")

dataset_test = load_dataset("mikkel-werling/medqa_cardiovascular")
