# from torchsummary import summary
from torch.autograd import Variable
from lib.model.faster_rcnn.vgg16 import vgg16
import argparse
import torch
import numpy as np
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=300, help="size of each image dimension")
parser.add_argument('--net', dest='net',default='vgg16', type=str, help='vgg16, res101')
parser.add_argument('--ls', dest='large_scale',action='store_true',help='whether use large imag scale')
args = parser.parse_args()
# print(opt)


def print_model_parm_flops(model,im_data,im_info,gt_boxes,num_boxes):


    multiply_adds = True  # FLOPs include multiply ops and add ops
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    # resnet = models.alexnet()
    # resnet = models.resnet152()

    model = model
    foo(model)

    # input = Variable(torch.rand(3, size, size).unsqueeze(0), requires_grad=True).cuda()
    out = model(im_data, im_info, gt_boxes, num_boxes)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
        print(cfg['POOLING_MODE'])
    # Set up model
    classes = ('__background__',  # always index 0
                     'crazing', 'inclusion', 'patches',
                     'pitted_surface', 'rolled-in_scale', 'scratches')
    fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()
    fasterRCNN.cuda()
    fasterRCNN.eval()

    im_data = torch.FloatTensor(1).to(device)
    im_info = torch.FloatTensor(1).to(device)
    num_boxes = torch.LongTensor(1).to(device)
    gt_boxes = torch.FloatTensor(1).to(device)

    input = np.random.random((1,args.img_size, args.img_size,3))
    im_blob = input
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], 1.5]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)
    # summary(net1, input_size=(3,300,300))
    im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.resize_(1, 1, 5).zero_()
    num_boxes.resize_(1).zero_()

    total = sum([param.nelement() for param in fasterRCNN.parameters()])

    print('  + fasterRCNN Number of params: %.2fM' % (total / 1e6))


    print_model_parm_flops(fasterRCNN,im_data,im_info,gt_boxes,num_boxes)