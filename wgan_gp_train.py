from model.wgan_64 import Discriminator, Generator
import configuration as conf
from utils import build_dirs, time2str
from init import gaussian_weights_init

import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.autograd import grad

import os
from os.path import join
import argparse
import tensorboardX

# Parameter setting
parser = argparse.ArgumentParser(description='A2I Training')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=conf.NUM_EPOCHS, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=conf.TRAIN_BATCH_SIZE,
                    type=int, metavar='N',
                    help='mini-batch size (default: {})'.format(conf.TRAIN_BATCH_SIZE), dest='batch_size')
parser.add_argument('--lr', '--learning-rate', default=conf.LEARNING_RATE,
                    type=float, metavar='LR', help='initial learning rate',
                    dest='learning_rate')
parser.add_argument('--mi', '--max-iterations', default=conf.MAX_ITERATIONS,
                    type=int, metavar='MI', help='Maximum iteration number',
                    dest='max_iterations')
parser.add_argument('--resume', default=0, type=int, metavar='check',
                    help='0 set as no resume and 1 set as resume')
parser.add_argument('--n_critic', default=5, type=int, metavar='NC',
                    help='number of critic training', dest='number_critic')
parser.add_argument('--gp-lambda', default=5, type=int, metavar='gp',
                    help='weight of gradient penalty', dest='gp_lambda')


# Main control script
def main():
    args = parser.parse_args()

    # initialize global path
    global_path = os.path.dirname(os.path.realpath(__file__))
    conf.global_path = global_path
    print('the global path: '.format(global_path))

    # configure the logging path.
    conf.time_id = time2str()
    conf.logging_path = join(global_path, './logs', conf.time_id)
    conf.writting_path = join(conf.logging_path, './logging')

    # configure checkpoint for images and models.
    conf.image_directory = join(
        conf.logging_path, conf.IMAGE_SAVING_DIRECTORY)
    conf.model_directory = join(
        conf.logging_path, conf.MODEL_SAVEING_DIRECTORY)
    build_dirs(conf.image_directory)
    build_dirs(conf.model_directory)
    build_dirs(conf.writting_path)
    conf.writer = tensorboardX.SummaryWriter(conf.writting_path)

    # Setting parameters
    conf.max_epochs = args.epochs
    print('number epochs: {}'.format(conf.max_epochs))
    conf.num_data_workers = args.workers
    print('number of workers: {}'.format(conf.num_data_workers))
    conf.lr = args.learning_rate
    print('learning rate: {}'.format(conf.lr))
    conf.batch_size = args.batch_size
    print('batch size: {}'.format(conf.batch_size))
    conf.max_iterations = args.max_iterations
    print('max number of iterations: {}'.format(conf.max_iterations))
    conf.n_critic = args.number_critic
    print('number of critic training: {}'.format(conf.n_critic))
    conf.gp_lambda = args.gp_lambda
    print('gradient penalty weight: {}'.format(conf.gp_lambda))

    train(conf)


# Dataset loading
def loading_data(conf):
    print("Loading dataset....")
    # Dataset = AWA_Dataset(join(conf.global_path, conf.DATA_ROOT))
    transformation = torchvision.transforms.Compose(
        [transforms.Resize(64),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    Dataset = torchvision.datasets.CIFAR10('./dataset/cifar10', train=True,
                                           transform=transformation,
                                           download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=Dataset, batch_size=conf.batch_size,
        shuffle=True, num_workers=conf.num_data_workers)

    print("Dataset length: {}".format(len(Dataset)))

    conf.data_set_len = len(Dataset)

    return train_loader


# Training initialization(Model Initialization)
def init_models(conf):
    # Setup Models
    print("Setup Models ......")
    Generator_Model = Generator()
    Discriminator_Model = Discriminator()

    # Model weight initialization
    print("Weight initialization......")
    Generator_Model.apply(gaussian_weights_init)
    Discriminator_Model.apply(gaussian_weights_init)
    model = Generator_Model, Discriminator_Model

    # Uploading the model into GPU
    if torch.cuda.is_available():
        Generator_Model.cuda()
        Discriminator_Model.cuda()

    # Setup the optimizers
    print("Optimization Setup.......")
    GeneratorOptimizor = torch.optim.Adam(
        Generator_Model.parameters(), lr=conf.lr, betas=(0.5, 0.999))

    DiscriminatorOptimizor = torch.optim.Adam(
        Discriminator_Model.parameters(), lr=conf.lr, betas=(0.5, 0.999))

    optimizer = GeneratorOptimizor, DiscriminatorOptimizor

    # Fixing noise
    fix_noise = torch.randn(conf.batch_size, conf.NOISE_SIZE)

    return model, optimizer, fix_noise


def train(conf):
    # Loading datasets
    train_loader = loading_data(conf)

    # Initialize Model
    model, optimizer, fix_noise = init_models(conf)

    Generator_Model, Discriminator_Model = model
    GeneratorOptimizor, DiscriminatorOptimizor = optimizer

    conf.iterations = 0

    # Begin training Process
    print("Begin Training Process........\n")

    for epoch in range(0, conf.max_epochs):
        Discriminator_Model.train()

        for it, (image, _) in enumerate(train_loader):
            Generator_Model.train()
            if image.size(0) != conf.batch_size:
                continue

            conf.iterations += 1
            if conf.iterations >= conf.max_iterations:
                return

            stop_flag = train_one_iteration(conf, image,
                                            Generator_Model,
                                            GeneratorOptimizor,
                                            Discriminator_Model,
                                            DiscriminatorOptimizor,
                                            fix_noise)
            if (it + 1) % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (epoch, it + 1, len(train_loader)))

            if stop_flag:
                return


def train_one_iteration(conf, image, Generator_Model, GeneratorOptimizor,
                        Discriminator_Model,
                        DiscriminatorOptimizor, fix_noise):
    # Copy data to the gpu
    if torch.cuda.is_available():
        image = Variable(image.cuda())
        fix_noise = Variable(fix_noise.cuda())
    else:
        image = Variable(image)
        fix_noise = Variable(fix_noise)

    # Discriminator Update
    dis_update(conf, Generator_Model, Discriminator_Model,
               DiscriminatorOptimizor, image)

    # Generator Update
    if conf.iterations % conf.n_critic == 0:
        gen_update(conf, Generator_Model, Discriminator_Model,
                   GeneratorOptimizor, DiscriminatorOptimizor)

    if (conf.iterations + 1) % 100 == 0:
        Generator_Model.eval()
        gen_image = Generator_Model(fix_noise)
        # Save the output images
        img_name = conf.image_directory + "/gen_image_" + str(conf.iterations + 1) + ".jpg"
        torchvision.utils.save_image((gen_image.data + 1) / 2, img_name)
        # # Save the input images
        # img_name = conf.image_directory + "/input_image.jpg"
        # torchvision.utils.save_image((image.data), img_name)


# Generator update
def dis_update(conf, Generator_Model, Discriminator_Model,
               DiscriminatorOptimizor, image):
    # Discriminator Update function
    DiscriminatorOptimizor.zero_grad()

    # Generate random noise
    noise = torch.randn(conf.batch_size, conf.NOISE_SIZE)

    if torch.cuda.is_available():
        noise = Variable(noise.cuda())
    else:
        noise = Variable(noise)

    # Get output from Generator
    output = Generator_Model(noise)

    # Feed forward Discriminator
    fake_image_output = Discriminator_Model(output.detach())
    real_image_output = Discriminator_Model(image)

    # Calculate Wasserstein-1 Distance
    w_distance = real_image_output.mean() - fake_image_output.mean()

    # Calculate Gradient Penalty
    g_penalty = gradient_penalty(image.data,
                                 output.data,
                                 Discriminator_Model)

    dis_loss = -w_distance + g_penalty * conf.gp_lambda

    # loss backprobagation
    dis_loss.backward(retain_graph=True)
    DiscriminatorOptimizor.step()

    conf.writer.add_scalar('D/wd', w_distance.data.cpu().numpy(), global_step=conf.iterations)
    conf.writer.add_scalar('D/gp', g_penalty.data.cpu().numpy(), global_step=conf.iterations)
    conf.writer.add_scalar('D/total', w_distance.data.cpu().numpy() +
                           g_penalty.data.cpu().numpy(),
                           global_step=conf.iterations)


# Generator update
def gen_update(conf, Generator_Model, Discriminator_Model,
               GeneratorOptimizor, DiscriminatorOptimizor):
    # Generator Update function
    # Optimizor setup
    GeneratorOptimizor.zero_grad()
    DiscriminatorOptimizor.zero_grad()

    # Generate random noise
    noise = torch.randn(conf.batch_size, conf.NOISE_SIZE)

    if torch.cuda.is_available():
        noise = Variable(noise.cuda())
    else:
        noise = Variable(noise)

    # Get output from Generator
    output = Generator_Model(noise)

    # Feed Forward to Discriminator
    fake_image_output = Discriminator_Model(output)

    # Generator loss Calculation
    gen_loss = -fake_image_output.mean()

    # Loss backprobagation
    gen_loss.backward()
    GeneratorOptimizor.step()

    conf.writer.add_scalars('G',
                            {"g_loss": gen_loss.data.cpu().numpy()},
                            global_step=conf.iterations)


# Gradient Penalty Calculation
def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    if torch.cuda.is_available():
        alpha = torch.rand(shape).cuda()
    else:
        alpha = torch.rand(shape)
    z = x + alpha * (y - x)

    # gradient penalty
    if torch.cuda.is_available():
        z = Variable(z, requires_grad=True).cuda()
    else:
        z = Variable(z, requires_grad=True)
    o = f(z)
    if torch.cuda.is_available():
        g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    else:
        g = grad(o, z, grad_outputs=torch.ones(o.size()), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


# Main Running control
if __name__ == '__main__':
    main()
