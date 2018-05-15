import time
import random
import numpy as np
from active_learner import ActiveLearner


class RandomLearner(ActiveLearner):
    def run(self):
        """ Runs the active learner with random data updates.
        :return: The lists of testing metrics from each iteration.
        """

        # Defines the lists to store the metrics.
        accuracies, mean_accuracies, recalls, precisions, f1_scores, losses = [], [], [], [], [], []
        start_time = time.clock()

        # Loops until all the data has been read.
        while len(self.data.data_y) != 0:
            self.log("\nCycle " + str(len(accuracies) + 1))

            # Trains a model with the training data.
            accuracy, mean_accuracy, recall, precision, f1_score, loss = self.model.train(self.data)

            # Adds the metrics to the lists.
            accuracies.append(accuracy)
            mean_accuracies.append(mean_accuracy)
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1_score)
            losses.append(loss)

            # Randomly adds data to the training data.
            if self.config.update_size * self.config.cell_patches< len(self.data.data_y):
                update = self.config.update_size
            else:
                update = len(self.data.data_x) // self.config.cell_patches

            cell_patches = random.sample(list(range(len(self.data.data_y) // self.config.cell_patches)), update)
            self.data.set_training_data(cell_patches)

            if self.config.pseudo_labels:
                predictions = self.model.predict(self.data, np.average)
                self.data.add_pesudo_labels(predictions)

            self.log("\n\n")

        # Trains the model with all the data.
        accuracy, mean_accuracy, recall, precision, f1_score, loss = self.model.train(self.data)

        # Adds the metrics to the lists.
        accuracies.append(accuracy)
        mean_accuracies.append(mean_accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1_score)
        losses.append(loss)

        # Logs the metrics.
        self.log("Accuracies: " + str(accuracies) + "\n")
        self.log("Mean Class Accuracies: " + str(mean_accuracies) + "\n")
        self.log("Recalls: " + str(recalls) + "\n")
        self.log("Precisions: " + str(precisions) + "\n")
        self.log("F1 Scores: " + str(f1_scores) + "\n")
        self.log("Losses: " + str(losses))
        self.log("Cycles: " + str(len(accuracies)) + " Time: " + str(time.clock() - start_time))

        # Returns the list of metrics.
        return accuracies, mean_accuracies, recalls, precisions, f1_scores, losses
