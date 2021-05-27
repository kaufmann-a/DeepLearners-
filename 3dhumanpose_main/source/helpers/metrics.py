from source.logcreator.logcreator import Logcreator


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def log(self, writer, comet, epoch, dataset_name="train", metric_name="loss"):
        avg = self.avg

        Logcreator.info(dataset_name, "avg.", metric_name, f"{avg:.5f}")

        # log values
        writer.add_scalar(metric_name + "/" + dataset_name, avg, epoch)
        if comet is not None:
            comet.log_metric(dataset_name + '_' + metric_name, avg, epoch=epoch)


class AverageMeterDict(object):
    """
    Handles a dictionary of values.
    Tracks the average of all values in the dictionary.
    """

    def __init__(self, dictionary_keys):
        """
        Instantiate the average meters.

        Args:
            nr_values: number of values in the dictionary
        """
        self.dict_keys = dictionary_keys
        self.avg_meters = dict()
        self.at_least_once_updated = False
        self.reset()

    def reset(self):
        """
        Reset all average meters.
        """
        self.at_least_once_updated = False
        for key in self.dict_keys:
            self.avg_meters[key] = AverageMeter()

    def update(self, dictionary: dict, n=1):
        """
        Updates the stored values for every key in dictionary_keys.
        Args:
            dictionary:
        """
        self.at_least_once_updated = True
        for key in self.dict_keys:
            if key in dictionary:
                val = dictionary[key]
                self.avg_meters[key].update(val, n)

    def log(self, writer, comet, epoch, dataset_name="train"):
        if self.at_least_once_updated:
            for key in self.dict_keys:
                avg = self.avg_meters[key].avg

                Logcreator.info(dataset_name, "avg.", key, f"{avg:.5f}")

                # log values
                writer.add_scalar(key + "/" + dataset_name, avg, epoch)
                if comet is not None:
                    comet.log_metric(dataset_name + '_' + key, avg, epoch=epoch)
