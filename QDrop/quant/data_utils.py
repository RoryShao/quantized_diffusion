import torch
import torch.nn.functional as F
from .quant_layer import QuantModule, Union
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock


def save_inp_oup_data(
    model: QuantModel,
    layer: Union[QuantModule, BaseQuantBlock],
    cali_data: torch.Tensor,
    wq: bool = False,
    aq: bool = False,
    batch_size: int = 32,
    keep_gpu: bool = True,
    input_prob: bool = False,
):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param weight_quant: use weight_quant quantization
    :param act_quant: use act_quant quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(
        model, layer, device=device, wq=wq, aq=aq, input_prob=input_prob
    )
    cached_batches = []
    print('cali_data:', cali_data[0].size(0))
    print('batch_size:', batch_size)
    for i in range(int(cali_data[0].size(0) / batch_size)):
        if input_prob:
            cur_inp, cur_out, cur_sym = get_inp_out(
                [_[i * batch_size : (i + 1) * batch_size] for _ in cali_data]
            )
            cached_batches.append((cur_inp.cpu(), cur_out.cpu(), cur_sym.cpu()))
        else:
            cur_inp, cur_out = get_inp_out(
                [_[i * batch_size : (i + 1) * batch_size] for _ in cali_data]
            )
            cached_batches.append((cur_inp.cpu(), cur_out.cpu()))
    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    if input_prob:
        cached_sym = torch.cat([x[2] for x in cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
        if input_prob:
            cached_sym = cached_sym.to(device)
    if input_prob:
        return (cached_inps, cached_sym), cached_outs
    return (cached_inps,), cached_outs


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """

    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(
        self,
        model: QuantModel,
        layer: Union[QuantModule, BaseQuantBlock],
        device: torch.device,
        wq: bool = False,
        aq: bool = False,
        input_prob: bool = False,
    ):
        self.model = model
        self.layer = layer
        self.device = device
        self.wq = wq
        self.aq = aq
        self.data_saver = DataSaverHook(
            store_input=True, store_output=True, stop_forward=True
        )
        self.input_prob = input_prob

    def __call__(self, model_input):
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(*[_.to(self.device) for _ in model_input])
            except StopForwardException:
                pass
            if self.input_prob:
                input_sym = self.data_saver.input_store[0].detach()
            if self.wq or self.aq:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=self.wq, act_quant=self.aq)
                try:
                    _ = self.model(*[_.to(self.device) for _ in model_input])
                except StopForwardException:
                    pass

            self.data_saver.store_output = True
        handle.remove()

        if self.input_prob:
            return (
                self.data_saver.input_store[0].detach(),
                self.data_saver.output_store.detach(),
                input_sym,
            )
        return (
            self.data_saver.input_store[0].detach(),
            self.data_saver.output_store.detach(),
        )


class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(
        self,
        model: QuantModel,
        layer: Union[QuantModule, BaseQuantBlock],
        device: torch.device,
        act_quant: bool = False,
    ):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(
                    F.log_softmax(out_q, dim=1),
                    F.softmax(out_fp, dim=1),
                    reduction="batchmean",
                )
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(
    model: QuantModule,
    layer: Union[QuantModule, BaseQuantBlock],
    act_quant: bool = False,
):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break


def get_train_samples(args, sample_data, custom_steps=None):
    num_samples, num_st = args.cali_n, args.cali_st
    custom_steps = args.custom_steps if custom_steps is None else custom_steps
    if num_st == 1:
        xs = sample_data[:num_samples]
        ts = (torch.ones(num_samples) * 800)
    else:
        # get the real number of timesteps (especially for DDIM)
        nsteps = len(sample_data["ts"])
        assert(nsteps >= custom_steps)
        timesteps = list(range(0, nsteps, nsteps//num_st))
        # logger.info(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')
        xs_lst = [sample_data["xs"][i][:num_samples] for i in timesteps]
        ts_lst = [sample_data["ts"][i][:num_samples] for i in timesteps]
        if args.cond:
            xs_lst += xs_lst
            ts_lst += ts_lst
            conds_lst = [sample_data["cs"][i][:num_samples] for i in timesteps] + [sample_data["ucs"][i][:num_samples] for i in timesteps]
        xs = torch.cat(xs_lst, dim=0)
        ts = torch.cat(ts_lst, dim=0)
        if args.cond:
            conds = torch.cat(conds_lst, dim=0)
            return xs, ts, conds
    return xs, ts