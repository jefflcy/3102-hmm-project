# Group Name: MarkovMonke

#test
# 2.1a
def naive_output_probs(training_data_filename):

    # constant for smoothing - need to try: 0.01, 0.1, 1, 10
    DELTA = 10

    # token - tag count
    token_tag_count = {}

    # tag count
    tag_count = {}

    # token - tag probability
    token_tag_prob = {}

    # list of all tokens
    tokens = []

    # number of unique words
    num_words = 0

    # read from file
    with open(training_data_filename) as file:
        for l in file.readlines():
            if len(l.strip()) != 0:
                token, tag = l.strip().split("\t")
                token_tag_count[(token, tag)] = token_tag_count.get((token, tag), 0) + 1
                tag_count[tag] = tag_count.get(tag, 0) + 1

                if token not in tokens:
                    tokens.append(token)
                    num_words += 1

    for (token, tag), count in token_tag_count.items():
        token_tag_prob[(token, tag)] = (count + DELTA) / (
            tag_count[tag] + DELTA * (num_words) + 1
        )

    # write to file
    with open("naive_output_probs.txt", "w") as file:
        for (token, tag), prob in token_tag_prob.items():
            file.write(f"{token}\t{tag}\t{prob}\n")


# 2.1b
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    # Load output probabilities from file
    output_probs = {}
    with open(in_output_probs_filename) as f:
        for line in f:
            token, tag, prob = line.strip().split("\t")
            output_probs[(token, tag)] = float(prob)

    # Predict tags for test data
    with open(in_test_filename) as f_in, open(out_prediction_filename, "w") as f_out:
        for line in f_in:
            token = line.strip()
            if token:  # Non-empty line
                max_prob = -1
                predicted_tag = None
                for tag in set(tag for (t, tag) in output_probs.keys() if t == token):
                    prob = output_probs.get((token, tag), 0)
                    if prob > max_prob:
                        max_prob = prob
                        predicted_tag = tag
                if predicted_tag is not None:
                    f_out.write(predicted_tag + "\n")
                else:
                    # Handle case when no tags are found for the token
                    f_out.write("UNKNOWN\n")
            else:  # Empty line (end of tweet)
                f_out.write("\n")


# 2.1c
# Naive prediction accuracy:     908/1378 = 0.6589259796806967


# 2.2a


# 2.2b
def naive_predict2(
    in_output_probs_filename,
    in_train_filename,
    in_test_filename,
    out_prediction_filename,
):
    # Load output probabilities from file
    output_probs = {}
    with open(in_output_probs_filename, "r", encoding="utf-8") as f:
        for line in f:
            token, tag, prob = line.strip().split("\t")
            output_probs[(token, tag)] = float(prob)

    # Load tag probabilities from training data
    tag_probs = {}
    total_tags = 0
    with open(in_train_filename) as f:
        for line in f:
            line = line.strip()
            if line:
                _, tag = line.split("\t")
                tag_probs[tag] = tag_probs.get(tag, 0) + 1
                total_tags += 1

    # Normalize tag probabilities
    for tag in tag_probs:
        tag_probs[tag] /= total_tags

    # Predict tags for test data
    with open(in_test_filename) as f_in, open(out_prediction_filename, "w") as f_out:
        for line in f_in:
            token = line.strip()
            if token:  # Non-empty line
                max_prob = -1
                predicted_tag = None
                for tag in set(tag for (_, tag) in output_probs.keys()):
                    prob = output_probs.get((token, tag), 0) * tag_probs.get(tag, 0)
                    if prob > max_prob:
                        max_prob = prob
                        predicted_tag = tag
                f_out.write(predicted_tag + "\n")
            else:  # Empty line (end of tweet)
                f_out.write("\n")


# 2.2c


def viterbi_predict(
    in_tags_filename,
    in_trans_probs_filename,
    in_output_probs_filename,
    in_test_filename,
    out_predictions_filename,
):
    pass


# 2.2c
# Naive2 prediction accuracy:


def viterbi_predict2(
    in_tags_filename,
    in_trans_probs_filename,
    in_output_probs_filename,
    in_test_filename,
    out_predictions_filename,
):
    pass


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth:
            correct += 1
    return correct, len(predicted_tags), correct / len(predicted_tags)


def run():
    """
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    """

    ddir = "/Users/jeff/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y2S2/BT3102/Group Proj/project_files"  # your working dir

    in_train_filename = f"{ddir}/twitter_train.txt"
    # naive_output_probs(in_train_filename)  ######################################### added this, rmb to remove
    naive_output_probs_filename = f"{ddir}/naive_output_probs.txt"

    in_test_filename = f"{ddir}/twitter_dev_no_tag.txt"
    in_ans_filename = f"{ddir}/twitter_dev_ans.txt"
    # naive_prediction_filename = f"{ddir}/naive_predictions.txt"
    # naive_predict(
    #     naive_output_probs_filename, in_test_filename, naive_prediction_filename
    # ) ######################################### run this first before running evaluate
    # correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    # print(f"Naive prediction accuracy:     {correct}/{total} = {acc}")

    naive_prediction_filename2 = f"{ddir}/naive_predictions2.txt"
    naive_predict2(
        naive_output_probs_filename,
        in_train_filename,
        in_test_filename,
        naive_prediction_filename2,
    )  ######################################### run this first before running evaluate
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f"Naive prediction2 accuracy:    {correct}/{total} = {acc}")

    # trans_probs_filename = f"{ddir}/trans_probs.txt"
    # output_probs_filename = f"{ddir}/output_probs.txt"

    # in_tags_filename = f"{ddir}/twitter_tags.txt"
    # viterbi_predictions_filename = f"{ddir}/viterbi_predictions.txt"
    # viterbi_predict(
    #     in_tags_filename,
    #     trans_probs_filename,
    #     output_probs_filename,
    #     in_test_filename,
    #     viterbi_predictions_filename,
    # ) ######################################### run this first before running evaluate
    # correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    # print(f"Viterbi prediction accuracy:   {correct}/{total} = {acc}")

    # trans_probs_filename2 = f"{ddir}/trans_probs2.txt"
    # output_probs_filename2 = f"{ddir}/output_probs2.txt"

    # viterbi_predictions_filename2 = f"{ddir}/viterbi_predictions2.txt"
    # viterbi_predict2(
    #     in_tags_filename,
    #     trans_probs_filename2,
    #     output_probs_filename2,
    #     in_test_filename,
    #     viterbi_predictions_filename2,
    # ) ######################################### run this first before running evaluate
    # correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    # print(f"Viterbi2 prediction accuracy:  {correct}/{total} = {acc}")


if __name__ == "__main__":
    run()
