import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable


# def image_loader(oads, image_name, index, transform, device):
#     image, label = oads.load_crop_from_image(image_name=str(image_name), index=int(index))
#     # image = Image.open(image_name)
#     # fake batch dimension required to fit network's input dimensions
#     image = transform(image).unsqueeze(0)
#     return image.to(device, torch.float), label['classId']


# class ContentLoss(nn.Module):

#     def __init__(self, target,):
#         super(ContentLoss, self).__init__()
#         # we 'detach' the target content from the tree used
#         # to dynamically compute the gradient: this is a stated value,
#         # not a variable. Otherwise the forward method of the criterion
#         # will throw an error.
#         self.target = target.detach()

#     def forward(self, input):
#         self.loss = F.mse_loss(input, self.target)
#         return input

# def gram_matrix(input):
#     a, b, c, d = input.size()  # a=batch size(=1)
#     # b=number of feature maps
#     # (c,d)=dimensions of a f. map (N=c*d)

#     features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

#     G = torch.mm(features, features.t())  # compute the gram product

#     # we 'normalize' the values of the gram matrix
#     # by dividing by the number of element in each feature maps.
#     return G.div(a * b * c * d)


# class StyleLoss(nn.Module):

#     def __init__(self, target_feature):
#         super(StyleLoss, self).__init__()
#         self.target = gram_matrix(target_feature).detach()

#     def forward(self, input):
#         G = gram_matrix(input)
#         self.loss = F.mse_loss(G, self.target)
#         return input
    

# # create a module to normalize input image so we can easily put it in a
# # nn.Sequential
# class Normalization(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalization, self).__init__()
#         # .view the mean and std to make them [C x 1 x 1] so that they can
#         # directly work with image Tensor of shape [B x C x H x W].
#         # B is batch size. C is number of channels. H is height and W is width.
#         self.mean = mean.clone().detach().view(-1, 1, 1)
#         self.std = std.clone().detach().view(-1, 1, 1)
#         # self.mean = torch.tensor(mean).view(-1, 1, 1)
#         # self.std = torch.tensor(std).view(-1, 1, 1)

#     def forward(self, img):
#         # normalize img
#         return (img - self.mean) / self.std


# def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
#                                style_img, content_img, device,
#                                content_layers=['conv_4'],
#                                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
#     # normalization module
#     normalization = Normalization(normalization_mean, normalization_std).to(device)

#     # just in order to have an iterable access to or list of content/syle
#     # losses
#     content_losses = []
#     style_losses = []

#     # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
#     # to put in modules that are supposed to be activated sequentially
#     model = nn.Sequential(normalization)

#     i = 0  # increment every time we see a conv
#     for layer in cnn.children():
#         if isinstance(layer, nn.Conv2d):
#             i += 1
#             name = 'conv_{}'.format(i)
#         elif isinstance(layer, nn.ReLU):
#             name = 'relu_{}'.format(i)
#             # The in-place version doesn't play very nicely with the ContentLoss
#             # and StyleLoss we insert below. So we replace with out-of-place
#             # ones here.
#             layer = nn.ReLU(inplace=False)
#         elif isinstance(layer, nn.MaxPool2d):
#             name = 'pool_{}'.format(i)
#         elif isinstance(layer, nn.BatchNorm2d):
#             name = 'bn_{}'.format(i)
#         else:
#             raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

#         model.add_module(name, layer)

#         if name in content_layers:
#             # add content loss:
#             target = model(content_img).detach()
#             content_loss = ContentLoss(target)
#             model.add_module("content_loss_{}".format(i), content_loss)
#             content_losses.append(content_loss)

#         if name in style_layers:
#             # add style loss:
#             target_feature = model(style_img).detach()
#             style_loss = StyleLoss(target_feature)
#             model.add_module("style_loss_{}".format(i), style_loss)
#             style_losses.append(style_loss)

#     # now we trim off the layers after the last content and style losses
#     for i in range(len(model) - 1, -1, -1):
#         if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
#             break

#     model = model[:(i + 1)]

#     return model, style_losses, content_losses



# def get_input_optimizer(input_img):
#     # this line to show that input is a parameter that requires a gradient
#     optimizer = optim.LBFGS([input_img])
#     return optimizer


# def run_style_transfer(cnn, normalization_mean, normalization_std, device,
#                        content_img, style_img, input_img, num_steps=300,
#                        style_weight=1000000, content_weight=1, verbose:bool=True):
#     """Run the style transfer."""
#     if verbose:
#         print('Building the style transfer model..')
#     model, style_losses, content_losses = get_style_model_and_losses(cnn,
#         normalization_mean, normalization_std, style_img, content_img, device=device)

#     # We want to optimize the input and not the model parameters so we
#     # update all the requires_grad fields accordingly
#     input_img.requires_grad_(True)
#     model.requires_grad_(False)

#     optimizer = get_input_optimizer(input_img)

#     if verbose:
#         print('Optimizing..')
#     run = [0]
#     with tqdm.tqdm(total=num_steps) as pbar:
#         while run[0] <= num_steps:

#             def closure():
#                 # correct the values of updated input image
#                 with torch.no_grad():
#                     input_img.clamp_(0, 1)

#                 optimizer.zero_grad()
#                 model(input_img)
#                 style_score = 0
#                 content_score = 0

#                 for sl in style_losses:
#                     style_score += sl.loss
#                 for cl in content_losses:
#                     content_score += cl.loss

#                 style_score *= style_weight
#                 content_score *= content_weight

#                 loss = style_score + content_score
#                 loss.backward()

#                 run[0] += 1
#                 pbar.update(1)
#                 if verbose and run[0] % 50 == 0:
#                     print("run {}:".format(run))
#                     print('Style Loss : {:4f} Content Loss: {:4f}'.format(
#                         style_score.item(), content_score.item()))
#                     print()

#                 return style_score + content_score

#             optimizer.step(closure)

#     # a last correction...
#     with torch.no_grad():
#         input_img.clamp_(0, 1)

#     return input_img

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        output = tensor.clone()
        for t, m, s in zip(output, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return output
    
def imshow_tensor(tensor, title=None, figsize=(10,5)):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = transforms.ToPILImage()(image)

    fig, ax = plt.subplots(1,1, figsize=figsize)
    
    ax.imshow(image)

    if title is not None:
        ax.set_title(title)

    plt.pause(0.001) # pause a bit so that plots are updated

    return fig


def new_forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self[0](x))
        out['r12'] = F.relu(self[2](out['r11']))
        out['p1'] = self[4](out['r12'])
        out['r21'] = F.relu(self[5](out['p1']))
        out['r22'] = F.relu(self[7](out['r21']))
        out['p2'] = self[9](out['r22'])
        out['r31'] = F.relu(self[10](out['p2']))
        out['r32'] = F.relu(self[12](out['r31']))
        out['r33'] = F.relu(self[14](out['r32']))
        out['r34'] = F.relu(self[16](out['r33']))
        out['p3'] = self[18](out['r34'])
        out['r41'] = F.relu(self[19](out['p3']))
        out['r42'] = F.relu(self[21](out['r41']))
        out['r43'] = F.relu(self[23](out['r42']))
        out['r44'] = F.relu(self[25](out['r43']))
        out['p4'] = self[27](out['r44'])
        out['r51'] = F.relu(self[28](out['p4']))
        out['r52'] = F.relu(self[30](out['r51']))
        out['r53'] = F.relu(self[32](out['r52']))
        out['r54'] = F.relu(self[34](out['r53']))
        out['p5'] = self[36](out['r54'])
        return [out[key] for key in out_keys]

class PytorchNeuralStyleTransfer():
    """
    Custom implementation of https://github.com/leongatys/PytorchNeuralStyleTransfer/tree/master
    for replication of https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html
    with Open Amsterdam Data Set (OADS)
    """

    def __init__(self, img_size, device, mean=[0.3410, 0.3123, 0.2787], std=[1,1,1]):
        self.mean = mean
        self.std = std # oads std [0.2362, 0.2252, 0.2162]
        self.img_size = img_size
        self.device = device

        self.prep = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                transforms.Normalize(mean=self.mean, #subtract imagenet mean
                                    std=self.std),
                transforms.Lambda(lambda x: x.mul_(255)),
                ])
        # self.postpa = transforms.Compose([
        #             transforms.Lambda(lambda x: x.mul_(1./255)),
        #             transforms.Normalize(mean=self.mean, #add imagenet mean
        #                                     std=self.std),
        #             transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
        #                 ])
        
        # self.postpb = transforms.Compose([transforms.ToPILImage()])
        # self.back_transform = transforms.Compose([
        #             # transforms.Lambda(lambda x: x.mul_(1./255)),
        #             UnNormalize(mean=self.mean, std=self.std),
        #             transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
        #             transforms.ToPILImage()
        #         ])
        self.post = transforms.Compose([
            transforms.Lambda(lambda x: x.mul_(1./255)),
            # transforms.Normalize(mean=mean, #add imagenet mean
            #                         std=[1,1,1]),
            UnNormalize(mean=self.mean, std=[1,1,1]),
            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
            transforms.Lambda(lambda x:torch.where(x>1, 1., torch.where(x<0, 0., x.type(torch.DoubleTensor)))),
            transforms.ToPILImage(),
        ])

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(self.device)
        except AttributeError:
            self.vgg = models.vgg19(pretrained=True).features.to(self.device)

        bound_method = new_forward.__get__(self.vgg, self.vgg.__class__)
        setattr(self.vgg, 'forward', bound_method)

        for param in self.vgg.parameters():
            param.requires_grad = False

        # if torch.cuda.is_available():
        #     self.vgg.cuda()

        self.style_layers = ['r11','r21','r31','r41', 'r51'] 
        self.content_layers = ['r42']
        self.loss_layers = self.style_layers + self.content_layers
        self.loss_fns = [self.GramMSELoss()] * len(self.style_layers) + [nn.MSELoss()] * len(self.content_layers)
        if torch.cuda.is_available():
            self.loss_fns = [loss_fn.cuda() for loss_fn in self.loss_fns]
            
        #these are good weights settings:
        self.style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
        self.content_weights = [1e0]
        self.weights = self.style_weights + self.content_weights

        self.unnorm = UnNormalize(mean=self.mean, std=[1,1,1])
    
    def postp(self, tensor): # to clip results in the range [0,1]
        t = self.postpa(tensor)
        t[t>1] = 1    
        t[t<0] = 0
        img = self.postpb(t)
        return img
    
    

    # gram matrix and loss
    class GramMatrix(nn.Module):
        def forward(self, input):
            b,c,h,w = input.size()
            F = input.view(b, c, h*w)
            G = torch.bmm(F, F.transpose(1,2)) 
            G.div_(h*w)
            return G

    class GramMSELoss(nn.Module):
        def forward(self, input, target):
            out = nn.MSELoss()(PytorchNeuralStyleTransfer.GramMatrix()(input), target)
            return(out)
        
    # def load_images(self, image_dir, )
        
    def run(self, style_image, content_image, max_iter=500, show_iter=50, verbose:bool=True):
        opt_img = Variable(content_image.data.clone(), requires_grad=True)

        style_targets = [self.GramMatrix()(A) for A in self.vgg(style_image, self.style_layers)]
        content_targets = [A for A in self.vgg(content_image, self.content_layers)]
        self.targets = style_targets + content_targets
        
        optimizer = optim.LBFGS([opt_img])
        n_iter=[0]

        while n_iter[0] <= max_iter:

            def closure():
                optimizer.zero_grad()
                out = self.vgg(opt_img, self.loss_layers)
                layer_losses = [self.weights[a] * self.loss_fns[a](A, self.targets[a]) for a,A in enumerate(out)]
                # loss = sum(layer_losses)
                # loss = torch.tensor(layer_losses, requires_grad=True).sum()
                loss = torch.sum(torch.stack(layer_losses))
                loss.backward()
                n_iter[0]+=1
                #print loss
                if verbose and n_iter[0]%show_iter == (show_iter-1):
                    print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
        #             print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
                return loss
            
            optimizer.step(closure)

        return self.post(opt_img.data[0].cpu().squeeze())