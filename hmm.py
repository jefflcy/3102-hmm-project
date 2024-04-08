# Group Name: MarkovMonke

# 2.1a
def compute_output_probabilities(training_data_filename, smoothing_delta, output_probs_filename):

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
        for line in file.readlines():
            if len(line.strip()) != 0:
                # stripping off whitespaces and splitting via tab character
                token, tag = line.strip().split("\t")

                # increment the count of token and tag pair
                if ((token, tag) in token_tag_count):
                    token_tag_count[(token, tag)] += 1
                else:
                    token_tag_count[(token, tag)] = 1
                    
                #increment the count of tag                
                if (tag in tag_count):
                    tag_count[tag] += 1
                else:
                    tag_count[tag] = 1
                
                if token not in tokens:
                    tokens.append(token)
                    num_words += 1

    # smoothed probability of token given tag
    for (token, tag), count in token_tag_count.items():
        token_tag_prob[(token, tag)] = (count + smoothing_delta) / (
            tag_count[tag] + smoothing_delta * (num_words) + 1
        )

    # write to file
    with open(output_probs_filename, "w") as file:
        for (token, tag), prob in token_tag_prob.items():
            file.write(f"{token}\t{tag}\t{prob}\n")


# 2.1b
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    # Load output probabilities from file into local variable
    output_probs = {}
    with open(in_output_probs_filename) as f:
        for line in f:
            # stripping off whitespaces and splitting via tab character
            token, tag, prob = line.strip().split("\t")
            output_probs[(token, tag)] = float(prob)

    # Predict tags for test data
    with open(in_test_filename) as f_in, open(out_prediction_filename, "w") as f_out:
        for line in f_in:
            # stripping off whitespaces
            token = line.strip()
            
            # Non-empty line
            if token:

                # initiate max probability with an impossible value
                max_prob = -1
                # initiate predicted tag with an empty string
                predicted_tag = ""

                # get a distinct list of tags that is paired with token using set()
                matching_tags = set()
                for (t, tag) in output_probs.keys():
                    if t == token:
                        matching_tags.add(tag)
                
                # find tag with highest probabilty 
                for tag in matching_tags:
                    prob = output_probs.get((token, tag), 0)
                    if prob > max_prob:
                        max_prob = prob
                        predicted_tag = tag

                # update output file with predicted tag or UNKOWN
                if predicted_tag != "":
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
    # Load output probabilities from file into local variable
    output_probs = {}
    with open(in_output_probs_filename) as f:
        for line in f:
            # stripping off whitespaces and splitting via tab character
            token, tag, prob = line.strip().split("\t")
            output_probs[(token, tag)] = float(prob)

    # Load tag probabilities from training data into local variable
    # count total number of tags
    tag_probs = {}
    total_tags = 0
    with open(in_train_filename) as f:
        for line in f:
            # stripping off whitespaces
            line = line.strip()
            if line:
                _, tag = line.split("\t")

                # increment the count of tag
                if (tag in tag_probs):
                    tag_probs[tag] += 1
                else:
                    tag_probs[tag] = 1

                total_tags += 1

    # Normalize tag probabilities
    for tag in tag_probs:
        tag_probs[tag] /= total_tags

    # Predict tags for test data
    with open(in_test_filename) as f_in, open(out_prediction_filename, "w") as f_out:
        for line in f_in:
            # stripping off whitespaces
            token = line.strip()
            if token:  # Non-empty line

                # initiate max probability with an impossible value
                max_prob = -1
                # initiate predicted tag with an empty string
                predicted_tag = ""

                # get a distinct list of tags using set()
                filtered_tags = set()
                for (_, tag) in output_probs.keys():
                    filtered_tags.add(tag)

                # find tag with highest probabilty 
                for tag in filtered_tags:
                    # probability of token given tag
                    token_given_tag_prob = output_probs.get((token, tag), 0)
                    # normalized probability of tag
                    tag_prob = tag_probs.get(tag, 0)

                    prob = token_given_tag_prob * tag_prob

                    if prob > max_prob:
                        max_prob = prob
                        predicted_tag = tag
                f_out.write(predicted_tag + "\n")
            else:  # Empty line (end of tweet)
                f_out.write("\n")


# 2.2c
# Naive2 prediction accuracy: 

# 3a
def compute_transition_probabilities(training_file, smoothing_delta, trans_probs_filename):
    # transition count
    transition_count = {}

    # tag count
    tag_count = {}
    
    # Initialize tag_count to account for START tag
    tag_count['*'] = 0

    with open(training_file) as f:
        prev_tag = '*'
        for line in f:
            line = line.strip()
            if line:
                _, tag = line.split('\t')

                # increment the count of transition
                if ((prev_tag, tag) in transition_count):
                    transition_count[(prev_tag, tag)] += 1
                else:
                    transition_count[(prev_tag, tag)] = 1
                    
                #increment the count of tag                
                if (tag in tag_count):
                    tag_count[tag] += 1
                else:
                    tag_count[tag] = 1

                # update previous tag to be the current tag
                prev_tag = tag
            else:  # Empty line (end of tweet)
                tag_count['*'] += 1  # Increment count for STOP tag
                prev_tag = '*'  # Reset prev_tag for next tweet

    # calculate transition probabilities
    num_tags = len(tag_count)
    for prev_tag in tag_count:
        for tag in tag_count:
            transition_count[(prev_tag, tag)] = (transition_count.get((prev_tag, tag), 0) + smoothing_delta) / \
                                                (tag_count.get(prev_tag, 0) + smoothing_delta * num_tags)

    # Write transition probabilities to file
    with open(trans_probs_filename, 'w') as f:
        for (prev_tag, tag), prob in transition_count.items():
            f.write(f"{prev_tag}\t{tag}\t{prob}\n")


# 3b
def viterbi_predict(
    in_tags_filename,
    in_trans_probs_filename,
    in_output_probs_filename,
    in_test_filename,
    out_predictions_filename,
):
    # Load tags
    tags = []
    with open(in_tags_filename) as f:
        for line in f:
            tag = line.strip()
            tags.append(tag)

    # Load transition probabilities
    trans_probs = {}
    with open(in_trans_probs_filename) as f:
        for line in f:
            prev_tag, tag, prob = line.strip().split('\t')
            trans_probs[(prev_tag, tag)] = float(prob)

    # Load output probabilities
    output_probs = {}
    with open(in_output_probs_filename) as f:
        for line in f:
            token, tag, prob = line.strip().split('\t')
            output_probs[(token, tag)] = float(prob)

    # Viterbi algorithm
    with open(in_test_filename) as f_in, open(out_predictions_filename, 'w') as f_out:
        for line in f_in:
            tokens = line.strip().split()
            if tokens:  # Non-empty line
                n = len(tokens)
                best_scores = {}
                back_pointers = {}
                for tag in tags:
                    # Initialization
                    best_scores[(0, tag)] = trans_probs.get(('*', tag), 0) * output_probs.get((tokens[0], tag), 0)
                    back_pointers[(0, tag)] = ""

                for i in range(1, n):
                    for tag in tags:                            


                        # loop to get highest probability with corressponding tag                        
                        check = []
                        for prev_tag in tags:
                            score =  best_scores[(i - 1, prev_tag)] \
                                            * trans_probs.get((prev_tag, tag), 0) \
                                            * output_probs.get((tokens[i], tag), 0)
                            check.append((score, prev_tag))
                        best_score, back_pointer = max(check)

                        # best_score, back_pointer = max(
                        #     ((best_scores[(i - 1, prev_tag)] * trans_probs.get((prev_tag, tag), 0) *
                        #       output_probs.get((tokens[i], tag), 0), prev_tag) for prev_tag in tags)
                        # )


                        best_scores[(i, tag)] = best_score
                        back_pointers[(i, tag)] = back_pointer

                # Find the best final tag
                best_final_tag = max(tags, key=lambda tag: best_scores[(n - 1, tag)])

                # Trace back to find the best tag sequence
                predicted_tags = [best_final_tag]
                prev_tag = best_final_tag
                for i in range(n - 1, 0, -1):
                    prev_tag = back_pointers[(i, prev_tag)]
                    predicted_tags.insert(0, prev_tag)

                # Write predicted tags to output file
                for token, tag in zip(tokens, predicted_tags):
                    f_out.write(tag + '\n')
            else:  # Empty line (end of tweet)
                f_out.write('\n')

# 3c
# Viterbi prediction accuracy: 68.87%


# 4a
def preprocess_data(training_file, processed_training_file):
    with open(training_file) as f_in, open(processed_training_file, "w") as f_out:
        for line in f_in:
            token, tag = line.strip().split('\t')
            processed_token = ""
            if (not token.startswith('@')) and token != '-':
                processed_token = token.lower()
            else:
                processed_token = token
            
            processed_line = processed_token + '\t' + tag
            f_out.write(processed_line + '\n')


# 4b
def viterbi_predict2(
    in_tags_filename,
    in_trans_probs_filename,
    in_output_probs_filename,
    in_test_filename,
    out_predictions_filename,
):
    # Load tags
    tags = []
    with open(in_tags_filename) as f:
        for line in f:
            tag = line.strip()
            tags.append(tag)

    # Load transition probabilities
    trans_probs = {}
    with open(in_trans_probs_filename) as f:
        for line in f:
            prev_tag, tag, prob = line.strip().split('\t')
            trans_probs[(prev_tag, tag)] = float(prob)

    # Load output probabilities
    output_probs = {}
    with open(in_output_probs_filename) as f:
        for line in f:
            token, tag, prob = line.strip().split('\t')
            output_probs[(token, tag)] = float(prob)

    # Viterbi algorithm
    with open(in_test_filename) as f_in, open(out_predictions_filename, 'w') as f_out:
        for line in f_in:
            tokens = line.strip().split()
            if tokens:  # Non-empty line
                n = len(tokens)
                best_scores = {}
                back_pointers = {}
                for tag in tags:
                    # Initialization
                    best_scores[(0, tag)] = trans_probs.get(('*', tag), 0) * output_probs.get((tokens[0], tag), 0)
                    back_pointers[(0, tag)] = ""

                for i in range(1, n):
                    for tag in tags:                            


                        # loop to get highest probability with corressponding tag                        
                        check = []
                        for prev_tag in tags:
                            score =  best_scores[(i - 1, prev_tag)] \
                                            * trans_probs.get((prev_tag, tag), 0) \
                                            * output_probs.get((tokens[i], tag), 0)
                            check.append((score, prev_tag))
                        best_score, back_pointer = max(check)

                        # best_score, back_pointer = max(
                        #     ((best_scores[(i - 1, prev_tag)] * trans_probs.get((prev_tag, tag), 0) *
                        #       output_probs.get((tokens[i], tag), 0), prev_tag) for prev_tag in tags)
                        # )


                        best_scores[(i, tag)] = best_score
                        back_pointers[(i, tag)] = back_pointer

                # Find the best final tag
                best_final_tag = max(tags, key=lambda tag: best_scores[(n - 1, tag)])

                # Trace back to find the best tag sequence
                predicted_tags = [best_final_tag]
                prev_tag = best_final_tag
                for i in range(n - 1, 0, -1):
                    prev_tag = back_pointers[(i, prev_tag)]
                    predicted_tags.insert(0, prev_tag)

                # Write predicted tags to output file
                for token, tag in zip(tokens, predicted_tags):
                    f_out.write(tag + '\n')
            else:  # Empty line (end of tweet)
                f_out.write('\n')

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

    ddir = "/Users/102al/Desktop/Y2/Y2S2/BT3102/Project/3102-hmm-project"  # your working dir

    in_train_filename = f"{ddir}/twitter_train.txt"
    # naive_output_probs(in_train_filename)  ######################################### added this, rmb to remove
    naive_output_probs_filename = f"{ddir}/naive_output_probs.txt"

    in_test_filename = f"{ddir}/twitter_dev_no_tag.txt"
    in_ans_filename = f"{ddir}/twitter_dev_ans.txt"


    naive_prediction_filename = f"{ddir}/naive_predictions.txt"
    naive_predict(
        naive_output_probs_filename, in_test_filename, naive_prediction_filename
    ) ######################################### run this first before running evaluate
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f"Naive prediction accuracy:     {correct}/{total} = {acc}")



    naive_prediction_filename2 = f"{ddir}/naive_predictions2.txt"
    naive_predict2(
        naive_output_probs_filename,
        in_train_filename,
        in_test_filename,
        naive_prediction_filename2,
    )  ######################################### run this first before running evaluate
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f"Naive prediction2 accuracy:    {correct}/{total} = {acc}")

    '''
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
    '''

if __name__ == "__main__":
    run()