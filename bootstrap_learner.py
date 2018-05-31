import math
import time
import numpy as np
from active_learner import ActiveLearner


class BootStrap_Learner(ActiveLearner):
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

            predictions, labels = self.model.predict(self.data, np.average)

            if self.config.pseudo_labels and len(self.data.data_y) != 0:
                self.data.add_pesudo_labels(predictions, labels)
            else:
                self.data.pseudo_indices = []

            train_x = np.append(self.data.train_x, self.data.data_x[self.data.pseudo_indices])
            train_y = np.append(self.data.train_y, self.data.data_y[self.data.pseudo_indices])

            if self.config.shortlist < len(self.data.data_y):
                update = self.config.update_size
            else:
                update = len(self.data.data_x) // self.config.cell_patches

            uncertainties = []
            for prediction in predictions:
                uncertainty = max(prediction)
                for i in range(self.config.num_classes):
                    if prediction[i] != 0.0:
                        uncertainty -= prediction[i] * math.log(prediction[i])
                uncertainties.append(uncertainty)

            indices = [i[1] for i in sorted(((value, index) for index, value in enumerate(uncertainties)),
                                            reverse=True)[:update]]

            bootstraps = self.data.get_bootstraps(train_x, train_y, indices)
            cell_predictions = []
            for i in range(len(bootstraps)):
                self.log("\nBootstrap " + str(i + 1))
                predictions, _ = self.model.train(bootstraps[i], test=False)
                cell_predictions.append(predictions)
            cell_predictions = np.average(cell_predictions, axis=0)

            if self.config.update_size * self.config.cell_patches < len(self.data.data_y):
                update = self.config.update_size
            else:
                update = len(self.data.data_x) // self.config.cell_patches

            uncertainties = []
            for prediction in cell_predictions:
                uncertainty = max(prediction)
                for i in range(self.config.num_classes):
                    uncertainty -= prediction[i] * math.log(prediction[i])
                uncertainties.append(uncertainty)

            indices = [i[1] for i in sorted(((value, index) for index, value in enumerate(uncertainties)),
                                            reverse=True)[:update]]

            self.data.set_training_data(indices)

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
