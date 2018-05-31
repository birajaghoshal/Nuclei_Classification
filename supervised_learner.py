import time
import numpy as np
from active_learner import ActiveLearner


class SupervisedLearner(ActiveLearner):
    def run(self):
        """ Runs the learner with random data updates.
        :return: The lists of testing metrics from each iteration.
         """

        # Defines the lists to store the metrics.
        accuracies, mean_accuracies, recalls, precisions, f1_scores, losses = [], [], [], [], [], []
        start_time = time.clock()

        update = self.config.update_size

        self.data.set_training_data(np.random.choice(list(range(len(self.data.data_x) // self.config.cell_patches)),
                                                     update, replace=False))

        # Loops until all the data has been read.
        while len(self.data.data_y) != 0 and len(accuracies) < self.config.max_updates:
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

            if self.config.update_size * self.config.cell_patches < len(self.data.data_y):
                update += self.config.update_size
            else:
                update += len(self.data.data_x) // self.config.cell_patches

            self.data.data_x = np.append(self.data.data_x, np.append(self.data.train_x, self.data.val_x))
            self.data.data_y = np.append(self.data.data_y, np.append(self.data.train_y, self.data.val_y))
            self.data.train_x, self.data.train_y = np.array([]), np.array([])
            self.data.val_x, self.data.val_y = np.array([]), np.array([])

            self.data.set_training_data(np.random.choice(list(range(len(self.data.data_x) // self.config.cell_patches)),
                                                         update, replace=False))

            self.log("\n\n")

        # Trains a model with the training data.
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
