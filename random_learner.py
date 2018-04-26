import time
import random
from active_learner import ActiveLearner


class Random(ActiveLearner):
    def run(self):
        """ Runs the active learner with random data updates.
        :return: The lists of testing metrics from each iteration.
        """

        # Defines the lists to store the metrics.
        accuracies, recalls, precisions, f1_scores, losses = [], [], [], [], []
        start_time = time.clock()

        # Loops until all the data has been read.
        while len(self.data.data_y) != 0:
            self.log('\nCycle ' + str(len(accuracies) + 1))

            # Trains a model with the training data.
            accuracy, recall, precision, f1_score, loss = self.train()

            # Adds the metrics to the lists.
            accuracies.append(accuracy)
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1_score)
            losses.append(loss)

            # Randomly adds data to the training data.
            self.data.set_training_data(random.sample(list(range(len(self.data.data_y))), self.config.update_size))
            self.log('\n\n')

        # Trains the model with all the data.
        accuracy, recall, precision, f1_score, loss = self.train()

        # Adds the metrics to the lists.
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1_score)
        losses.append(loss)

        # Logs the metrics.
        self.log('Accuracies: ' + str(accuracies) + '\n')
        self.log('Recalls: ' + str(recalls) + '\n')
        self.log('Precisions: ' + str(precisions) + '\n')
        self.log('F1 Scores: ' + str(f1_scores) + '\n')
        self.log('Losses: ' + str(losses))
        self.log('Cycles: ' + str(len(accuracies)) + ' Time: ' + str(time.clock() - start_time))

        # Returns the list of metrics.
        return accuracies, recalls, precisions, f1_scores, losses
