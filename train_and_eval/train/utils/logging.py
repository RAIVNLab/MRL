import abc
import tqdm

from torch.utils.tensorboard import SummaryWriter


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, tqdm_writer=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if not tqdm_writer:
            print("\t".join(entries))
        else:
            tqdm.tqdm.write("\t".join(entries))

    def write_to_tensorboard(
        self, writer: SummaryWriter, prefix="train", global_step=None
    ):
        for meter in self.meters:
            avg = meter.avg
            val = meter.val
            if meter.write_val:
                writer.add_scalar(
                    f"{prefix}/{meter.name}_val", val, global_step=global_step
                )

            if meter.write_avg:
                writer.add_scalar(
                    f"{prefix}/{meter.name}_avg", avg, global_step=global_step
                )

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class Meter(object):
    @abc.abstractmethod
    def __init__(self, name, fmt=":f"):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class AverageMeter(Meter):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f", write_val=True, write_avg=True):
        self.name = name
        self.fmt = fmt
        self.reset()

        self.write_val = write_val
        self.write_avg = write_avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class VarianceMeter(Meter):
    def __init__(self, name, fmt=":f", write_val=False):
        self.name = name
        self._ex_sq = AverageMeter(name="_subvariance_1", fmt=":.02f")
        self._sq_ex = AverageMeter(name="_subvariance_2", fmt=":.02f")
        self.fmt = fmt
        self.reset()
        self.write_val = False
        self.write_avg = True

    @property
    def val(self):
        return self._ex_sq.val - self._sq_ex.val ** 2

    @property
    def avg(self):
        return self._ex_sq.avg - self._sq_ex.avg ** 2

    def reset(self):
        self._ex_sq.reset()
        self._sq_ex.reset()

    def update(self, val, n=1):
        self._ex_sq.update(val ** 2, n=n)
        self._sq_ex.update(val, n=n)

    def __str__(self):
        return ("{name} (var {avg" + self.fmt + "})").format(
            name=self.name, avg=self.avg
        )